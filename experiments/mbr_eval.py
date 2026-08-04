"""
Test Minimum-Bayes-Risk-style reranking: generate N diverse candidates per
source sentence, pick the one with highest average chrF similarity to the
OTHER candidates (the "consensus" candidate), instead of trusting a single
beam-search output. Idea: translation errors are idiosyncratic per-sample,
so outlier candidates are more likely wrong; the consensus candidate is a
safer bet. Compared against plain single-beam decoding on silver_dev (all 20
real test-relevant directions), same caveat as pivot_eval.py: absolute
numbers are known-inflated, but the comparison between two decoding
strategies on the same biased metric should still be informative.
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


@torch.inference_mode()
def generate_candidates(model, tok, texts, src_dialect, tgt_dialect, device, num_candidates=5):
    """num_return_sequences diverse-beam candidates per source sentence."""
    src_code = to_nllb(src_dialect)
    tgt_code = to_nllb(tgt_dialect)
    id_lists = [build_ids(tok, t, src_code) for t in texts]
    input_ids, attn = pad_batch(id_lists, tok.pad_token_id)
    input_ids, attn = input_ids.to(device), attn.to(device)
    tgt_id = tok.convert_tokens_to_ids(tgt_code)
    out = model.generate(input_ids=input_ids, attention_mask=attn, forced_bos_token_id=tgt_id,
                          max_new_tokens=64, num_beams=num_candidates, num_beam_groups=num_candidates,
                          diversity_penalty=0.8, num_return_sequences=num_candidates,
                          no_repeat_ngram_size=3, repetition_penalty=1.3)
    decoded = tok.batch_decode(out, skip_special_tokens=True)
    # reshape: [n_src * num_candidates] -> [n_src][num_candidates]
    return [decoded[i * num_candidates:(i + 1) * num_candidates] for i in range(len(texts))]


def mbr_select(candidates_per_src):
    """Pick, per source sentence, the candidate with highest average chrF to the others."""
    selected = []
    for cands in candidates_per_src:
        if len(cands) == 1:
            selected.append(cands[0])
            continue
        scores = []
        for i, c in enumerate(cands):
            others = cands[:i] + cands[i + 1:]
            avg = sum(_chrf.sentence_score(c, [o]).score for o in others) / len(others)
            scores.append(avg)
        best_idx = max(range(len(cands)), key=lambda i: scores[i])
        selected.append(cands[best_idx])
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--num-candidates", type=int, default=5)
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

    single_scores, mbr_scores = {}, {}
    for (src_code, tgt_code), items in by_pair.items():
        src_d, tgt_d = NLLB_TO_DIALECT[src_code], NLLB_TO_DIALECT[tgt_code]
        srcs = [r["src_text"] for r in items]
        refs = [r["tgt_text"] for r in items]

        cands = generate_candidates(model, tok, srcs, src_d, tgt_d, device, args.num_candidates)
        single_hyps = [c[0] for c in cands]  # first beam group's top candidate = plain single-beam-ish baseline
        mbr_hyps = mbr_select(cands)

        single_scores[f"{src_d}->{tgt_d}"] = corpus_scores(single_hyps, refs)
        mbr_scores[f"{src_d}->{tgt_d}"] = corpus_scores(mbr_hyps, refs)
        print(f"{src_d:>12} -> {tgt_d:<12}  single: spBLEU={single_scores[f'{src_d}->{tgt_d}']['spBLEU']:.2f}  |  "
              f"MBR: spBLEU={mbr_scores[f'{src_d}->{tgt_d}']['spBLEU']:.2f}")
        sys.stdout.flush()

    n = len(single_scores)
    avg_single = sum(v["spBLEU"] for v in single_scores.values()) / n
    avg_mbr = sum(v["spBLEU"] for v in mbr_scores.values()) / n
    n_mbr_wins = sum(1 for k in single_scores if mbr_scores[k]["spBLEU"] > single_scores[k]["spBLEU"])
    print(f"\n=== SUMMARY ({n} directions) ===")
    print(f"single-candidate: avg spBLEU={avg_single:.3f}")
    print(f"MBR-selected:     avg spBLEU={avg_mbr:.3f}")
    print(f"MBR beats single for {n_mbr_wins}/{n} directions")


if __name__ == "__main__":
    main()
