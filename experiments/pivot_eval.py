"""
Test whether MSA-pivot decoding (source -> MSA -> target, two NLLB hops)
beats direct decoding (source -> target, one hop) for the v4 model.

Rationale (from literature: MSA-as-pivot gave +0.6-1.4 BLEU in prior
low-resource dialect MT work; our own gold_dev shows the model's MSA<->
Palestinian competence is strong -- spBLEU 27.8/41.1 -- built on 16,000 real
gold examples, while most dialect-to-dialect pairs only have mined/synthetic
signal). Tested on silver_dev (all 20 real directions among the 5 labeled
dialects) using the SAME (known-imperfect but consistent) metric for both
decoding strategies, so the comparison is informative even if absolute
numbers are inflated -- the bias should apply symmetrically to both.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch.distributed.tensor  # noqa: F401
from nllb_utils import translate
from lang_codes import NLLB_TO_DIALECT
from spbleu_chrf import corpus_scores

DATA = Path(__file__).resolve().parent / "data"
MODEL_NAME = "facebook/nllb-200-distilled-600M"


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def pivot_translate(model, tok, texts, src_dialect, tgt_dialect, device, batch_size=8):
    """src -> MSA -> tgt, two hops."""
    msa_hop = []
    for i in range(0, len(texts), batch_size):
        msa_hop.extend(translate(model, tok, texts[i:i + batch_size], src_dialect, "MSA", device, num_beams=5))
    final = []
    for i in range(0, len(msa_hop), batch_size):
        final.extend(translate(model, tok, msa_hop[i:i + batch_size], "MSA", tgt_dialect, device, num_beams=5))
    return final


def direct_translate(model, tok, texts, src_dialect, tgt_dialect, device, batch_size=8):
    out = []
    for i in range(0, len(texts), batch_size):
        out.extend(translate(model, tok, texts[i:i + batch_size], src_dialect, tgt_dialect, device, num_beams=5))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=True)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.to(device).eval()

    recs = read_jsonl(DATA / "silver_dev.jsonl")
    by_pair = defaultdict(list)
    for r in recs:
        by_pair[(r["src_lang"], r["tgt_lang"])].append(r)

    direct_scores, pivot_scores = {}, {}
    for (src_code, tgt_code), items in by_pair.items():
        src_d, tgt_d = NLLB_TO_DIALECT[src_code], NLLB_TO_DIALECT[tgt_code]
        if src_d == "MSA" or tgt_d == "MSA":
            continue  # pivoting through MSA is a no-op / degenerate for MSA-involving pairs
        srcs = [r["src_text"] for r in items]
        refs = [r["tgt_text"] for r in items]

        direct_hyps = direct_translate(model, tok, srcs, src_d, tgt_d, device)
        pivot_hyps = pivot_translate(model, tok, srcs, src_d, tgt_d, device)

        direct_scores[f"{src_d}->{tgt_d}"] = corpus_scores(direct_hyps, refs)
        pivot_scores[f"{src_d}->{tgt_d}"] = corpus_scores(pivot_hyps, refs)
        print(f"{src_d:>12} -> {tgt_d:<12}  direct: spBLEU={direct_scores[f'{src_d}->{tgt_d}']['spBLEU']:.2f} "
              f"chrF++={direct_scores[f'{src_d}->{tgt_d}']['chrF++']:.2f}   |   "
              f"pivot: spBLEU={pivot_scores[f'{src_d}->{tgt_d}']['spBLEU']:.2f} "
              f"chrF++={pivot_scores[f'{src_d}->{tgt_d}']['chrF++']:.2f}")
        sys.stdout.flush()

    n = len(direct_scores)
    avg_direct_bleu = sum(v["spBLEU"] for v in direct_scores.values()) / n
    avg_pivot_bleu = sum(v["spBLEU"] for v in pivot_scores.values()) / n
    avg_direct_chrf = sum(v["chrF++"] for v in direct_scores.values()) / n
    avg_pivot_chrf = sum(v["chrF++"] for v in pivot_scores.values()) / n
    n_pivot_wins = sum(1 for k in direct_scores if pivot_scores[k]["spBLEU"] > direct_scores[k]["spBLEU"])

    print(f"\n=== SUMMARY ({n} non-MSA directions) ===")
    print(f"direct: avg spBLEU={avg_direct_bleu:.3f} avg chrF++={avg_direct_chrf:.3f}")
    print(f"pivot:  avg spBLEU={avg_pivot_bleu:.3f} avg chrF++={avg_pivot_chrf:.3f}")
    print(f"pivot beats direct on spBLEU for {n_pivot_wins}/{n} directions")

    with open(Path(__file__).resolve().parent / "outputs" / "pivot_vs_direct_report.json", "w", encoding="utf-8") as f:
        json.dump({"direct": direct_scores, "pivot": pivot_scores}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
