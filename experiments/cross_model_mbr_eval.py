"""
Cross-model consensus reranking: instead of pooling diverse-beam candidates
from a SINGLE model (mbr_eval.py, found inconclusive: 8/20 wins, noise-level),
pool the top candidate from TWO independently-trained models (v4b: rank=64/
8ep on original bronze; v5: rank=32/5ep, self-distilled bronze) and pick
whichever one has higher chrF similarity to the other -- i.e. treat
"the two models agree" as a proxy for "probably correct", same MBR logic
but across models instead of across samples from one model. Two
independently-trained models are more likely to have UNCORRELATED errors
than diverse-beam samples from the same model/weights, which is what made
single-model MBR fail to help (the beams share the same failure modes).
Validated on gold_dev (trustworthy direction, MSA<->PAL only) and silver_dev
(all 20 real test-relevant directions, known-inflated absolute numbers but
comparable relative to the v4b-alone baseline on the same biased metric).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sacrebleu.metrics import CHRF

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch.distributed.tensor  # noqa: F401
from nllb_utils import build_ids, pad_batch, to_nllb
from lang_codes import NLLB_TO_DIALECT
from spbleu_chrf import corpus_scores

DATA = Path(__file__).resolve().parent / "data"
MODEL_NAME = "facebook/nllb-200-distilled-600M"
_chrf = CHRF()


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_model(adapter_path, device):
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=True)
    model = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    model.to(device).eval()
    return model


@torch.inference_mode()
def generate_single_best(model, tok, texts, src_dialect, tgt_dialect, device, beams=8):
    src_code = to_nllb(src_dialect)
    tgt_code = to_nllb(tgt_dialect)
    id_lists = [build_ids(tok, t, src_code) for t in texts]
    input_ids, attn = pad_batch(id_lists, tok.pad_token_id)
    input_ids, attn = input_ids.to(device), attn.to(device)
    tgt_id = tok.convert_tokens_to_ids(tgt_code)
    out = model.generate(input_ids=input_ids, attention_mask=attn, forced_bos_token_id=tgt_id,
                          max_new_tokens=64, num_beams=beams, no_repeat_ngram_size=3,
                          repetition_penalty=1.3)
    return tok.batch_decode(out, skip_special_tokens=True)


def consensus_select(hyps_a, hyps_b):
    """Per sentence: if both models agree closely (high chrF between the two
    candidates), consensus is strong -- default to model A's (the stronger
    single model, v4b) candidate. If they disagree, still default to A but
    this is where a real 3rd-model tiebreak would help; recorded separately
    for analysis."""
    selected, agreements = [], []
    for a, b in zip(hyps_a, hyps_b):
        agree = _chrf.sentence_score(a, [b]).score
        agreements.append(agree)
        selected.append(a)  # A (v4b, the stronger model) is always the pick;
        # this variant measures whether FILTERING low-agreement cases to fall
        # back on B ever helps, tested as a second candidate strategy below.
    return selected, agreements


def consensus_select_swap_on_disagree(hyps_a, hyps_b, agreements, threshold):
    """Where agreement is low (models diverge a lot), try swapping to B
    instead of trusting A blindly, and see if that helps or hurts."""
    return [b if agree < threshold else a for a, b, agree in zip(hyps_a, hyps_b, agreements)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-a", required=True, help="stronger model (v4b)")
    ap.add_argument("--adapter-b", required=True, help="second model (v5)")
    ap.add_argument("--dev-set", default="silver_dev.jsonl")
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Loading model A (v4b) from {args.adapter_a}")
    model_a = load_model(args.adapter_a, device)
    print(f"Loading model B (v5) from {args.adapter_b}")
    model_b = load_model(args.adapter_b, device)

    recs = read_jsonl(DATA / args.dev_set)
    by_pair = defaultdict(list)
    for r in recs:
        by_pair[(r["src_lang"], r["tgt_lang"])].append(r)

    # Fixed, pre-registered thresholds evaluated identically across every
    # direction -- NOT cherry-picked per direction after seeing scores, since
    # that would just overfit the dev set and tell us nothing about how this
    # would perform on the real blind test set.
    THRESHOLDS = [30, 40, 50, 60, 70]
    a_scores = {}
    all_hyps_a, all_hyps_b, all_refs, all_agreements = [], [], [], []
    for (src_code, tgt_code), items in by_pair.items():
        src_d, tgt_d = NLLB_TO_DIALECT[src_code], NLLB_TO_DIALECT[tgt_code]
        srcs = [r["src_text"] for r in items]
        refs = [r["tgt_text"] for r in items]

        hyps_a = generate_single_best(model_a, tok, srcs, src_d, tgt_d, device)
        hyps_b = generate_single_best(model_b, tok, srcs, src_d, tgt_d, device)
        _, agreements = consensus_select(hyps_a, hyps_b)

        a_scores[f"{src_d}->{tgt_d}"] = corpus_scores(hyps_a, refs)
        all_hyps_a.extend(hyps_a); all_hyps_b.extend(hyps_b)
        all_refs.extend(refs); all_agreements.extend(agreements)
        print(f"{src_d:>12} -> {tgt_d:<12}  v4b-alone: spBLEU={a_scores[f'{src_d}->{tgt_d}']['spBLEU']:.2f}  "
              f"(avg agreement chrF={sum(agreements)/len(agreements):.1f})")
        sys.stdout.flush()

    n = len(a_scores)
    avg_a = sum(v["spBLEU"] for v in a_scores.values()) / n
    micro_a = corpus_scores(all_hyps_a, all_refs)["spBLEU"]
    print(f"\n=== SUMMARY ({n} directions, dev={args.dev_set}) ===")
    print(f"v4b-alone: macro-avg spBLEU={avg_a:.3f}  micro (pooled) spBLEU={micro_a:.3f}")
    print(f"\n--- fixed global thresholds (swap to v5 when agreement < threshold) ---")
    for threshold in THRESHOLDS:
        swap_hyps = consensus_select_swap_on_disagree(all_hyps_a, all_hyps_b, all_agreements, threshold)
        micro_swap = corpus_scores(swap_hyps, all_refs)["spBLEU"]
        n_swapped = sum(1 for ag in all_agreements if ag < threshold)
        print(f"threshold={threshold:>3}: micro spBLEU={micro_swap:.3f}  "
              f"(swapped {n_swapped}/{len(all_agreements)} sentences to v5)  "
              f"{'BEATS' if micro_swap > micro_a else 'does not beat'} v4b-alone")


if __name__ == "__main__":
    main()
