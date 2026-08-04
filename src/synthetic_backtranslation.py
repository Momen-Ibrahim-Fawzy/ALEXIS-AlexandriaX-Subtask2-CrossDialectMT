"""
Stage 3: synthetic parallel data for the two "cold start" dialects that have
ZERO public data of any kind -- Egyptian and Lebanese (EDA_INSIGHTS.md §5/§6).

Technique: knowledge distillation / self-training (Sennrich et al. 2016 style
backtranslation, applied as forward synthetic-target generation here since we
have no dialect text to backtranslate FROM): take real, financial-domain
sentences (the MSA + Palestinian sides of the gold pairs, plus the Moroccan/
Saudi/Tunisian monolingual pools) and translate them into Egyptian (arz_Arab)
and Lebanese (apc_Arab) with the zero-shot NLLB-200 backbone, which already has
dedicated pretraining for both codes. The resulting (real_source, synthetic_
target) pairs give the fine-tuning stage *some* domain- and register-adapted
signal for these two directions, instead of leaving them purely zero-shot.

This is explicitly synthetic/silver-minus data -- documented as such, never
presented as gold -- and is the standard, literature-supported fallback for a
language pair with no parallel data at all.

Output: data/synthetic_egy_leb_pairs.jsonl records:
  {src_dialect, tgt_dialect, src_text, tgt_text, method: "nllb_zeroshot_distill"}

Run: CUDA_VISIBLE_DEVICES=1 conda run -n mo python synthetic_backtranslation.py
"""
import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch.distributed.tensor  # noqa: F401 -- see evaluate.py for why this is needed with an adapter
from lang_codes import to_nllb
from nllb_utils import translate

OUT = Path(__file__).resolve().parent / "data"
MODEL_NAME = "facebook/nllb-200-distilled-600M"
TARGETS = ["Egyptian", "Lebanese"]
# v2: per-dialect cap, not a global shuffle-then-truncate. The v1 approach
# pooled all 5 source dialects and truncated to one global N, which -- since
# MSA/Palestinian's pools are far larger -- silently starved Moroccan/Saudi/
# Tunisian (536/552/148 sentences respectively) of BOTH forward coverage and
# any leftover for the reverse direction (Egyptian/Lebanese -> those 3
# dialects ended up with zero training pairs; see build_finetune_corpus.py's
# v2 fix for the allocation side of this same bug). Capping per-dialect
# instead guarantees every source dialect gets a meaningful, comparable
# amount translated, while still respecting genuinely small pools (Tunisian
# has only 999 sentences total) rather than inflating them artificially.
PER_DIALECT_CAP = 3000
BATCH_SIZE = 64
MAX_NEW_TOKENS = 64

RNG = random.Random(42)


def load_source_pool():
    """Real sentences to translate FROM: MSA+PAL gold-pair sides (highest quality,
    financial-domain, human-written) plus the Moroccan/Saudi/Tunisian monolingual
    pools, each tagged with its true dialect so the resulting pairs also teach
    {Moroccan,Saudi,Tunisian,MSA,Palestinian} -> {Egyptian,Lebanese}."""
    by_dialect = {}
    seen = set()

    def add(d, t):
        if t not in seen:
            seen.add(t)
            by_dialect.setdefault(d, []).append(t)

    with open(OUT / "gold_train.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            add("MSA", r["msa"])
            add("Palestinian", r["pal"])
    with open(OUT / "labeled_pool.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["dialect"] in ("Moroccan", "Saudi", "Tunisian"):
                add(r["dialect"], r["text"])

    items = []
    for d, texts in by_dialect.items():
        RNG.shuffle(texts)
        capped = texts[:PER_DIALECT_CAP]
        items.extend((d, t) for t in capped)
        print(f"  source pool[{d}]: {len(capped)} / {len(texts)} available")
    RNG.shuffle(items)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None,
                     help="v5 NOTE: regenerate Egyptian/Lebanese bronze targets using a fine-tuned "
                          "adapter (e.g. v4) instead of the raw zero-shot backbone. Rationale: v4's "
                          "overall Arabic translation competence improved substantially (real gold_dev "
                          "gains), and Egyptian/Lebanese -- ~39% of all test rows -- are the two "
                          "directions with zero real training data, entirely dependent on this self-"
                          "distillation step's quality. A better teacher here directly targets the "
                          "single largest remaining weak spot by test-row share.")
    ap.add_argument("--out-name", default="synthetic_egy_leb_pairs.jsonl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=True)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        print(f"Loaded adapter from {args.adapter} as the self-distillation teacher")
    model = model.to(device).eval()

    pool = load_source_pool()
    print(f"Source pool: {len(pool)} real sentences (MSA/Palestinian/Moroccan/Saudi/Tunisian)")

    records = []
    for tgt_dialect in TARGETS:
        # group by source dialect (each dialect needs its own src language code)
        by_dialect = {}
        for d, t in pool:
            by_dialect.setdefault(d, []).append(t)
        for src_dialect, texts in by_dialect.items():
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                outs = translate(model, tok, batch, src_dialect, tgt_dialect, device,
                                  max_new_tokens=MAX_NEW_TOKENS, num_beams=4)
                for s, o in zip(batch, outs):
                    records.append({
                        "src_dialect": src_dialect, "tgt_dialect": tgt_dialect,
                        "src_text": s, "tgt_text": o,
                        "method": "nllb_zeroshot_distill" if not args.adapter else "nllb_finetuned_selfdistill",
                    })
                if (i // BATCH_SIZE) % 10 == 0:
                    print(f"  {src_dialect} -> {tgt_dialect}: {i + len(batch)}/{len(texts)}")

    with open(OUT / args.out_name, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[synthetic] {len(records)} synthetic pairs written to {args.out_name}")


if __name__ == "__main__":
    main()
