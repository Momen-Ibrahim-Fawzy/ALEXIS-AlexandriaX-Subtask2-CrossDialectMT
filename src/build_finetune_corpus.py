"""
Stage 4: combine gold / silver / bronze tiers into the final many-to-many
fine-tuning corpus for NLLB-200, plus TWO held-out dev sets for real,
direction-specific evaluation (see "v2 dev-set redesign" below).

Tiers (see EDA_INSIGHTS.md §6 and each stage script's docstring for how they
were built):
  - gold   : data/gold_train.jsonl        (real MSA<->Palestinian, both directions)
  - silver : data/silver_pairs.jsonl      (intent-mined, 5 labeled dialects, all directed pairs)
  - bronze : data/synthetic_egy_leb_pairs.jsonl (NLLB self-distilled, Egyptian/Lebanese as target)
             + its reverse direction (Egyptian/Lebanese as source), included at lower weight
             since the "source" side there is itself machine-generated.

v2 dev-set redesign
--------------------
v1 only had validation signal for MSA<->Palestinian (gold_dev, not even a
required test direction) and Egyptian/Lebanese (soft_dev, a trivial-for-
baseline proxy). The 24 other real test-scoring directions (Moroccan<->Saudi,
Palestinian<->Tunisian, etc.) had NO validation signal at all. Silver mining
is now abundant enough (44K pairs after widening) to fix this: for every
silver-eligible directed pair, the top-N *highest-cosine-similarity* mined
pairs are held out as `silver_dev` -- real, native, human-written text on
both sides, near-paraphrase quality, leak-checked against the blind test file
-- giving real per-direction spBLEU/chrF++ for the pairs that actually matter
for scoring. Still not human-translated gold, so reported and labeled
separately, never conflated with `gold_dev`.

v2 quality filter
-------------------
Bronze pairs are model-generated (self-distilled from zero-shot NLLB); a
degenerate (repetition-loop) "reference" would actively teach the model that
failure mode rather than just occasionally producing it. Filtered out via
`nllb_utils.is_degenerate` before entering training or soft_dev.

Output: data/finetune_train.jsonl, data/silver_dev.jsonl, data/finetune_dev_soft.jsonl
Each record: {src_lang, tgt_lang, src_text, tgt_text, tier}
(src_lang/tgt_lang are NLLB FLORES-200 codes, ready for the trainer.)
"""
import json
import random
from pathlib import Path
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lang_codes import to_nllb
from nllb_utils import is_degenerate

TASK_ROOT = Path(__file__).resolve().parent
OUT = Path(__file__).resolve().parent / "data"
RNG = random.Random(42)

# Oversampling caps per tier so no single tier dominates the many-to-many
# mix. Originally capped low (2,500) reasoning that gold MSA<->PAL "isn't
# even a required test direction" and shouldn't swamp the pairs that matter
# -- but that reasoning implicitly trusted `silver_dev` to arbitrate what
# "matters." v4 revisits this: real Codabench feedback on v2/v3 showed
# `silver_dev` overestimates absolute quality ~3x (mined-pair reference bias)
# while `gold_dev` -- built from REAL human-translated data, just like this
# tier -- correctly predicted v3 > v2 in the real world. Hypothesis: more
# real parallel supervision (even in a non-test direction) teaches general
# translation competence/fidelity that transfers better than more mined data
# does, given mined-silver scaling (v2->v3, 1200->2000/pair) already showed
# only marginal real gains. Raised 2500->8000 (of 18,457 available real
# pairs) to test this directly, holding silver/bronze caps at v3's
# already-real-world-validated levels rather than changing multiple things
# at once.
MAX_GOLD = 8000
# v2: silver mining yield was widened 6.5x (mine_silver_pairs.py: cap 6->40,
# threshold 0.55->0.50). Caps here tightened from an initial v2 draft
# (2000/2000/1200) after bronze *generation alone* ran 60+min against a
# ~24min estimate under GPU contention on this shared machine -- training is
# more expensive per-example than inference, so a much bigger corpus risked a
# multi-hour run. v2 trained in 85min once contention eased, though, so v3
# raises these again -- verified against actual post-leak-filter supply
# (539-2578/pair for silver, exactly 3000 or 901/pair for bronze) rather than
# guessing: 2000 lets silver's 12 best-supplied pairs land at ~full use and
# the top 6 at ~80% use; bronze's 1700fwd+1260rev+40dev=3000 uses the ENTIRE
# generated pool for the 8 non-Tunisian pairs (v2 left ~560/pair unused).
MAX_SILVER_PER_PAIR = 2000
MAX_BRONZE_FORWARD_PER_PAIR = 1700
MAX_BRONZE_REVERSE_PER_PAIR = 1260  # reverse direction is noisier -> smaller cap

SILVER_DEV_HOLDOUT_PER_PAIR = 20      # highest-cos_sim pairs, reserved before the training cap
BRONZE_REVERSE_SOFT_DEV_HOLDOUT = 40  # per pair, held out from bronze reverse for soft-dev only


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_blind_test_sentences():
    import pandas as pd
    test = pd.read_csv(TASK_ROOT / "Data" / "subtask2_test_participants.csv")
    return set(test["Source sentence"].astype(str).str.strip())


def main():
    blind_test_sents = load_blind_test_sentences()
    records = []

    # --- gold ---
    gold = read_jsonl(OUT / "gold_train.jsonl")
    RNG.shuffle(gold)
    gold = gold[:MAX_GOLD]
    for r in gold:
        records.append({"src_lang": to_nllb("MSA"), "tgt_lang": to_nllb("Palestinian"),
                         "src_text": r["msa"], "tgt_text": r["pal"], "tier": "gold"})
        records.append({"src_lang": to_nllb("Palestinian"), "tgt_lang": to_nllb("MSA"),
                         "src_text": r["pal"], "tgt_text": r["msa"], "tier": "gold"})
    print(f"gold: {len(gold)} QID pairs -> {2*len(gold)} directed examples")

    # --- silver (+ silver_dev holdout) ---
    silver = read_jsonl(OUT / "silver_pairs.jsonl")
    by_pair = {}
    for r in silver:
        by_pair.setdefault((r["src_dialect"], r["tgt_dialect"]), []).append(r)
    n_silver, n_silver_dev, n_silver_leak_skipped = 0, 0, 0
    silver_dev = []
    for pair, items in by_pair.items():
        items.sort(key=lambda r: -r["cos_sim"])
        # leak check first (a leaked pair must never land in dev OR train with
        # its blind-test twin visible -- simplest safe rule: drop it entirely)
        clean = []
        for r in items:
            if r["src_text"] in blind_test_sents or r["tgt_text"] in blind_test_sents:
                n_silver_leak_skipped += 1
                continue
            clean.append(r)

        dev_slice = clean[:SILVER_DEV_HOLDOUT_PER_PAIR]
        train_slice = clean[SILVER_DEV_HOLDOUT_PER_PAIR:SILVER_DEV_HOLDOUT_PER_PAIR + MAX_SILVER_PER_PAIR]

        for r in dev_slice:
            silver_dev.append({"src_lang": to_nllb(r["src_dialect"]), "tgt_lang": to_nllb(r["tgt_dialect"]),
                                "src_text": r["src_text"], "tgt_text": r["tgt_text"], "cos_sim": r["cos_sim"],
                                "tier": "silver_dev"})
        n_silver_dev += len(dev_slice)

        for r in train_slice:
            records.append({"src_lang": to_nllb(r["src_dialect"]), "tgt_lang": to_nllb(r["tgt_dialect"]),
                             "src_text": r["src_text"], "tgt_text": r["tgt_text"], "tier": "silver"})
        n_silver += len(train_slice)
    print(f"silver: {n_silver} directed training examples across {len(by_pair)} pairs "
          f"(+ {n_silver_dev} held out as silver_dev, {n_silver_leak_skipped} dropped for blind-test overlap)")

    # --- bronze (forward: real source -> synthetic Egyptian/Lebanese) ---
    # v5: regenerated using v4 (fine-tuned) as the self-distillation teacher
    # instead of the raw zero-shot backbone -- v4's overall Arabic competence
    # improved substantially (real gold_dev gains), and this directly targets
    # Egyptian/Lebanese (~39% of all test rows, the two directions with zero
    # real training data of any kind).
    bronze = read_jsonl(OUT / "synthetic_egy_leb_pairs_v5.jsonl")
    n_bronze_raw = len(bronze)
    bronze = [r for r in bronze if not is_degenerate(r["tgt_text"])]
    n_bronze_degenerate = n_bronze_raw - len(bronze)
    by_pair_b = {}
    for r in bronze:
        by_pair_b.setdefault((r["src_dialect"], r["tgt_dialect"]), []).append(r)
    n_bronze_fwd, n_bronze_rev = 0, 0
    soft_dev = []
    for pair, items in by_pair_b.items():
        RNG.shuffle(items)
        fwd = items[:MAX_BRONZE_FORWARD_PER_PAIR]
        for r in fwd:
            records.append({"src_lang": to_nllb(r["src_dialect"]), "tgt_lang": to_nllb(r["tgt_dialect"]),
                             "src_text": r["src_text"], "tgt_text": r["tgt_text"], "tier": "bronze_fwd"})
        n_bronze_fwd += len(fwd)

        # reverse direction: synthetic-Egyptian/Lebanese -> real source, held-out slice for soft-dev
        rev_pool = items[MAX_BRONZE_FORWARD_PER_PAIR:]
        RNG.shuffle(rev_pool)
        dev_slice = rev_pool[:BRONZE_REVERSE_SOFT_DEV_HOLDOUT]
        train_slice = rev_pool[BRONZE_REVERSE_SOFT_DEV_HOLDOUT:BRONZE_REVERSE_SOFT_DEV_HOLDOUT + MAX_BRONZE_REVERSE_PER_PAIR]
        for r in train_slice:
            records.append({"src_lang": to_nllb(r["tgt_dialect"]), "tgt_lang": to_nllb(r["src_dialect"]),
                             "src_text": r["tgt_text"], "tgt_text": r["src_text"], "tier": "bronze_rev"})
        n_bronze_rev += len(train_slice)
        for r in dev_slice:
            soft_dev.append({"src_lang": to_nllb(r["src_dialect"]), "tgt_lang": to_nllb(r["tgt_dialect"]),
                              "src_text": r["src_text"], "tgt_text": r["tgt_text"], "tier": "bronze_soft_dev"})

    print(f"bronze forward: {n_bronze_fwd}, bronze reverse: {n_bronze_rev}, "
          f"soft-dev held out: {len(soft_dev)} ({n_bronze_degenerate} degenerate synthetic pairs dropped pre-training)")

    RNG.shuffle(records)
    with open(OUT / "finetune_train.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "finetune_dev_soft.jsonl", "w", encoding="utf-8") as f:
        for r in soft_dev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "silver_dev.jsonl", "w", encoding="utf-8") as f:
        for r in silver_dev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nTOTAL finetune_train.jsonl: {len(records)} examples")
    tier_counts = Counter(r["tier"] for r in records)
    for t, c in tier_counts.items():
        print(f"  {t}: {c}")
    dir_counts = Counter((r["src_lang"], r["tgt_lang"]) for r in records)
    print(f"\n{len(dir_counts)} distinct (src_lang,tgt_lang) directions covered:")
    for k, v in sorted(dir_counts.items()):
        print(f"  {k[0]} -> {k[1]}: {v}")

    dev_dir_counts = Counter((r["src_lang"], r["tgt_lang"]) for r in silver_dev)
    print(f"\nsilver_dev: {len(silver_dev)} examples across {len(dev_dir_counts)} directions "
          f"(min/max per direction: {min(dev_dir_counts.values()) if dev_dir_counts else 0}/"
          f"{max(dev_dir_counts.values()) if dev_dir_counts else 0})")

    # persisted (not just printed) so downstream stages -- the trainer and the
    # submission bundler -- can cite exact corpus composition in their outputs,
    # e.g. for a system-description paper.
    stats = {
        "total_examples": len(records),
        "tier_counts": dict(tier_counts),
        "n_directions": len(dir_counts),
        "direction_counts": {f"{k[0]}->{k[1]}": v for k, v in sorted(dir_counts.items())},
        "caps": {"MAX_GOLD": MAX_GOLD, "MAX_SILVER_PER_PAIR": MAX_SILVER_PER_PAIR,
                 "MAX_BRONZE_FORWARD_PER_PAIR": MAX_BRONZE_FORWARD_PER_PAIR,
                 "MAX_BRONZE_REVERSE_PER_PAIR": MAX_BRONZE_REVERSE_PER_PAIR,
                 "SILVER_DEV_HOLDOUT_PER_PAIR": SILVER_DEV_HOLDOUT_PER_PAIR},
        "soft_dev_size": len(soft_dev),
        "silver_dev_size": len(silver_dev),
        "silver_dev_n_directions": len(dev_dir_counts),
        "n_bronze_degenerate_filtered": n_bronze_degenerate,
        "n_silver_leak_skipped": n_silver_leak_skipped,
    }
    with open(OUT / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUT / 'corpus_stats.json'}")


if __name__ == "__main__":
    main()
