"""Small decoding-hyperparameter sweep validated on gold_dev (the one dev
set proven to track real Codabench performance -- see RESEARCH_NOTES.md).
Tries a few (num_beams, length_penalty, repetition_penalty) combos and
reports gold_dev spBLEU/chrF++ for each, so the best config can be used for
final test predictions instead of untuned defaults."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch.distributed.tensor  # noqa: F401 -- see evaluate.py for why this is needed
from nllb_utils import build_ids, pad_batch, to_nllb
from spbleu_chrf import corpus_scores

DATA = Path(__file__).resolve().parent / "data"
MODEL_NAME = "facebook/nllb-200-distilled-600M"


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


@torch.inference_mode()
def translate(model, tok, texts, src_dialect, tgt_dialect, device, num_beams, length_penalty, repetition_penalty):
    src_code = to_nllb(src_dialect)
    tgt_code = to_nllb(tgt_dialect)
    id_lists = [build_ids(tok, t, src_code) for t in texts]
    input_ids, attn = pad_batch(id_lists, tok.pad_token_id)
    input_ids, attn = input_ids.to(device), attn.to(device)
    tgt_id = tok.convert_tokens_to_ids(tgt_code)
    out = model.generate(input_ids=input_ids, attention_mask=attn, forced_bos_token_id=tgt_id,
                          max_new_tokens=64, num_beams=num_beams, length_penalty=length_penalty,
                          no_repeat_ngram_size=3, repetition_penalty=repetition_penalty)
    return tok.batch_decode(out, skip_special_tokens=True)


def eval_gold_dev(tok, model, device, num_beams, length_penalty, repetition_penalty, batch_size=8):
    recs = read_jsonl(DATA / "gold_dev.jsonl")
    results = {}
    for direction, src_key, tgt_key, src_d, tgt_d in [
        ("MSA->Palestinian", "msa", "pal", "MSA", "Palestinian"),
        ("Palestinian->MSA", "pal", "msa", "Palestinian", "MSA"),
    ]:
        hyps, refs = [], []
        for i in range(0, len(recs), batch_size):
            batch = recs[i:i + batch_size]
            srcs = [r[src_key] for r in batch]
            outs = translate(model, tok, srcs, src_d, tgt_d, device, num_beams, length_penalty, repetition_penalty)
            hyps.extend(outs)
            refs.extend(r[tgt_key] for r in batch)
        results[direction] = corpus_scores(hyps, refs)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (mutually exclusive with --model)")
    ap.add_argument("--model", default=None, help="full local model dir, e.g. a full-FT checkpoint (no PEFT wrapping)")
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if args.model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, use_safetensors=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=True)
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.to(device).eval()

    configs = [
        (5, 1.0, 1.3),   # current default
        (4, 1.0, 1.3),
        (8, 1.0, 1.3),
        (5, 0.8, 1.3),
        (5, 1.2, 1.3),
        (5, 1.0, 1.1),
        (5, 1.0, 1.5),
    ]
    print(f"{'beams':>6} {'len_pen':>8} {'rep_pen':>8} {'avg_spBLEU':>12} {'avg_chrF++':>12}")
    for nb, lp, rp in configs:
        res = eval_gold_dev(tok, model, device, nb, lp, rp)
        avg_bleu = (res["MSA->Palestinian"]["spBLEU"] + res["Palestinian->MSA"]["spBLEU"]) / 2
        avg_chrf = (res["MSA->Palestinian"]["chrF++"] + res["Palestinian->MSA"]["chrF++"]) / 2
        print(f"{nb:>6} {lp:>8} {rp:>8} {avg_bleu:>12.3f} {avg_chrf:>12.3f}   {res}")


if __name__ == "__main__":
    main()
