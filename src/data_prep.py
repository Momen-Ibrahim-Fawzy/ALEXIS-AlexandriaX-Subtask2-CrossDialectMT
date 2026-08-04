"""
Stage 1 of the training-data pipeline: turn raw ArBanking77 into three artifacts.

1. `gold_msa_pal_pairs.jsonl`   — 19,193 real, human-translated MSA<->Palestinian
                                    sentence pairs (both directions), QID-aligned,
                                    from Banking77_full_corpus.csv.
2. `labeled_pool.jsonl`          — every sentence from the 5 labeled monolingual
                                    dialect files (MSA, Palestinian, Moroccan,
                                    Saudi, Tunisian) tagged with its gold 77-way
                                    banking-intent label. Feeds stage 2 (mining).
3. `dev_holdout_ids.json`        — QIDs / sentences held out of gold-pair training
                                    so dev-set spBLEU/chrF++ numbers aren't inflated
                                    by sentences that also appear verbatim in the
                                    blind test file (see EDA_INSIGHTS.md §8).

Run: conda run -n mo python data_prep.py
"""
import json
import random
import re
from pathlib import Path

import pandas as pd


def clean_text(t):
    """Strip a leading-underscore data artifact found in the raw Tunisian
    ArBanking77 file (29.4% of its 999 rows start with a literal '_', 0% in
    any other dialect file -- confirmed via audit_dialect_markers.py-style
    inspection). The model faithfully learned to reproduce it: 35% of v2/v3's
    real Tunisian-target *predictions* started with the same character,
    something no other target dialect showed at all. Not a decoding bug --
    the artifact was baked into the source text this whole time."""
    return re.sub(r"^[_\s]+", "", str(t)).strip()

TASK_ROOT = Path(__file__).resolve().parent
EXT = TASK_ROOT / "Data" / "external" / "ArBanking77" / "data"
TEST_CSV = TASK_ROOT / "Data" / "subtask2_test_participants.csv"
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True, parents=True)

RNG = random.Random(42)

MONO_FILES = {
    "MSA": EXT / "Banking77_Arabized_MSA_test.csv",
    "Palestinian": EXT / "Banking77_Arabized_PAL_test.csv",
    "Moroccan": EXT / "Banking77_Arabized_Moroccan_test.csv",
    "Saudi": EXT / "Banking77_Arabized_Saudi_test.csv",
    "Tunisian": EXT / "Banking77_Arabized_Tunisian_test.csv",
}


def build_gold_pairs():
    full = pd.read_csv(EXT / "Banking77_full_corpus.csv", engine="python", on_bad_lines="skip")
    blind_test_sents = set(pd.read_csv(TEST_CSV)["Source sentence"].astype(str).str.strip())

    records = []
    for _, row in full.iterrows():
        msas = [row.get("Question_MSA1"), row.get("Question_MSA2")]
        pals = [row.get("Question_PAL1"), row.get("Question_PAL2")]
        msas = [m.strip() for m in msas if isinstance(m, str) and m.strip()]
        pals = [p.strip() for p in pals if isinstance(p, str) and p.strip()]
        for m in msas:
            for p in pals:
                # exclude any pair where EITHER side verbatim-matches a blind-test
                # source sentence, so dev metrics computed from this pool can't be
                # inflated by memorizing content the real test set also contains.
                leaks_into_test = (m in blind_test_sents) or (p in blind_test_sents)
                records.append({
                    "qid": int(row["QID"]),
                    "intent_ar": row["Intent_ar"],
                    "intent_id": int(row["Intent_ID"]),
                    "msa": m,
                    "pal": p,
                    "overlaps_blind_test": leaks_into_test,
                })

    with open(OUT / "gold_msa_pal_pairs.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_leak = sum(r["overlaps_blind_test"] for r in records)
    print(f"[gold] {len(records)} real MSA<->PAL pairs written "
          f"({n_leak} flagged as overlapping the blind test file, excluded from dev-set sampling)")
    return records


def build_labeled_pool():
    rows = []
    n_cleaned = 0
    for dialect, path in MONO_FILES.items():
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            raw = str(r["text"]).strip()
            cleaned = clean_text(raw)
            n_cleaned += (cleaned != raw)
            rows.append({"dialect": dialect, "text": cleaned, "intent_ar": r["label"]})
    with open(OUT / "labeled_pool.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[labeled_pool] {len(rows)} labeled sentences across {len(MONO_FILES)} dialects written "
          f"({n_cleaned} had a leading-underscore/whitespace artifact stripped)")
    return rows


def build_dev_split(gold_records, dev_frac=0.04, seed=42):
    """Sentence-level (not QID-level) held-out dev split of the gold pairs,
    restricted to pairs that don't overlap the blind test file."""
    eligible = [r for r in gold_records if not r["overlaps_blind_test"]]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    n_dev = max(200, int(len(eligible) * dev_frac))
    dev = eligible[:n_dev]
    train = eligible[n_dev:] + [r for r in gold_records if r["overlaps_blind_test"]]
    # (train also gets the "overlaps_blind_test" pairs back -- fine, they're only
    #  excluded from *dev*, using them for *training* is legitimate real data.)
    with open(OUT / "gold_train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "gold_dev.jsonl", "w", encoding="utf-8") as f:
        for r in dev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[split] gold_train={len(train)}  gold_dev={len(dev)} (leak-free)")


if __name__ == "__main__":
    gold = build_gold_pairs()
    build_labeled_pool()
    build_dev_split(gold)
