"""
Shared NLLB-200 encode/decode helpers.

IMPORTANT — documented gotcha (see System/README.md "Implementation notes"):
the `transformers==5.5.0` `TokenizersBackend` fast tokenizer wrapper installed
in this environment does NOT honor `tokenizer.src_lang` / `tokenizer.tgt_lang`
the way classic `NllbTokenizer` did: calling `tok(text, src_lang=...)` silently
skips prepending the language-code token, and plain `tok(text)` appends a
spurious trailing `<unk>` after `</s>`. Verified directly by inspecting
`input_ids` -> token strings for several inputs. Confirmed generation still
"works" (forced_bos_token_id correctly drives the decoder side), but training
on encoder inputs that never carry the FLORES-200 src-language token would
silently diverge from how NLLB-200 was actually pretrained (source format is
documented as `[src_lang_code, tokens..., </s>]`).

Fix: bypass the buggy auto-injection entirely. Tokenize raw content with
`add_special_tokens=False` (verified clean, no stray tokens), then manually
build `[lang_code_id, *content_ids, eos_id]` ourselves. Used identically for
training-label construction and for inference, so train/inference formats
always match.
"""
import re
from collections import Counter
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lang_codes import to_nllb  # noqa: E402

MAX_LEN_DEFAULT = 96


def is_degenerate(text: str) -> bool:
    """Flags beam-search repetition-loop output (e.g. "X X X X X..."), the
    failure mode found in ~0.2% of an early prediction run -- since fixed at
    generation time (no_repeat_ngram_size/repetition_penalty below) but kept
    here as a shared, zero-cost check used in two places: (1) filtering the
    bronze synthetic tier before it becomes training data -- a degenerate
    "reference" would actively teach the model this failure mode, not just
    occasionally exhibit it; (2) a final QA safety net on the actual
    submission predictions."""
    words = text.split()
    if len(words) < 6:
        return False
    _, count = Counter(words).most_common(1)[0]
    if count >= 5 and count / len(words) > 0.35:
        return True
    if re.search(r"(\b\w+\b(?:\s+\b\w+\b){0,2})(?:\s+\1){2,}", text):
        return True
    return False


def build_ids(tok, text: str, dialect_or_code: str, max_len: int = MAX_LEN_DEFAULT):
    """Return input_ids list: [lang_id, content_tokens..., eos_id]."""
    lang_code = dialect_or_code if dialect_or_code.endswith("_Arab") or "_" in dialect_or_code and len(dialect_or_code) == 8 else to_nllb(dialect_or_code)
    content = tok(text, add_special_tokens=False, truncation=True, max_length=max_len - 2)["input_ids"]
    lang_id = tok.convert_tokens_to_ids(lang_code)
    eos_id = tok.eos_token_id
    return [lang_id] + content + [eos_id]


def pad_batch(id_lists, pad_id, pad_to_multiple_of=None):
    maxlen = max(len(x) for x in id_lists)
    if pad_to_multiple_of:
        import math
        maxlen = int(math.ceil(maxlen / pad_to_multiple_of) * pad_to_multiple_of)
    input_ids = torch.full((len(id_lists), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(id_lists), maxlen), dtype=torch.long)
    for i, ids in enumerate(id_lists):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :len(ids)] = 1
    return input_ids, attn


@torch.inference_mode()
def translate(model, tok, texts, src_dialect, tgt_dialect, device, max_new_tokens=64, num_beams=4,
              max_len=MAX_LEN_DEFAULT, no_repeat_ngram_size=3, repetition_penalty=1.3):
    """Translate a batch of `texts` from src_dialect to tgt_dialect, using the
    manual (bug-safe) encoding scheme.

    `no_repeat_ngram_size`/`repetition_penalty` guard against beam-search
    repetition degeneration (e.g. "X X X X X..." loops) -- found in ~0.2% of
    an earlier prediction run, concentrated in the least-supervised-with-real-
    data direction (into Egyptian). Standard, low-risk NMT decoding practice;
    values chosen conservatively (mild penalty, no_repeat_ngram_size=3) to
    avoid over-constraining otherwise-fluent short outputs.
    """
    src_code = to_nllb(src_dialect) if src_dialect in _KNOWN_DIALECTS else src_dialect
    tgt_code = to_nllb(tgt_dialect) if tgt_dialect in _KNOWN_DIALECTS else tgt_dialect
    id_lists = [build_ids(tok, t, src_code, max_len=max_len) for t in texts]
    input_ids, attn = pad_batch(id_lists, tok.pad_token_id)
    input_ids, attn = input_ids.to(device), attn.to(device)
    tgt_id = tok.convert_tokens_to_ids(tgt_code)
    out = model.generate(input_ids=input_ids, attention_mask=attn, forced_bos_token_id=tgt_id,
                          max_new_tokens=max_new_tokens, num_beams=num_beams,
                          no_repeat_ngram_size=no_repeat_ngram_size, repetition_penalty=repetition_penalty)
    return tok.batch_decode(out, skip_special_tokens=True)


_KNOWN_DIALECTS = {"MSA", "Palestinian", "Lebanese", "Moroccan", "Tunisian", "Egyptian", "Saudi"}
