"""
Stage 2: mine "silver" cross-dialect pseudo-parallel pairs among the 5 dialects
that have real, human-written text with a recoverable banking-intent label
(MSA, Palestinian, Moroccan, Saudi, Tunisian — see EDA_INSIGHTS.md §5/§9).

Technique: bucket every sentence by its gold 77-way intent label, then within
each (source_dialect, target_dialect, intent) bucket, embed all candidate
sentences with BAAI/bge-m3 and greedily pair each source sentence with its
nearest-neighbor target sentence by cosine similarity (parallel-corpus-mining
style, restricted to same-intent buckets for precision). This is weak/silver
supervision -- both sides are real, native, human-written sentences, just not
translations of each other -- but nearest-neighbor-within-intent is far less
noisy than random pairing within the bucket.

Output: data/silver_pairs.jsonl records:
  {src_dialect, tgt_dialect, src_text, tgt_text, intent_ar, cos_sim}

Run: CUDA_VISIBLE_DEVICES=1 conda run -n mo python mine_silver_pairs.py
"""
import json
import itertools
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

OUT = Path(__file__).resolve().parent / "data"
LABELED_DIALECTS = ["MSA", "Palestinian", "Moroccan", "Saudi", "Tunisian"]

# Minimum cosine similarity to accept a pair (precision filter). Chosen
# conservatively: same-intent nearest neighbor almost always clears this,
# pairs that don't are usually a small/singleton intent bucket with no good match.
# v2: lowered slightly (0.55 -> 0.50) -- still a solid precision floor for
# intent-bucket-restricted mining (the literature's LASER/LaBSE-style mining
# typically runs 0.5-0.6 *without* a label restriction; ours has that extra
# precision guard already) -- to recover more of the smaller/harder buckets.
MIN_SIM = 0.50
# v1 used 6, which turned out to be the actual binding constraint for every
# "big" dialect pair (462 = 6*77 intents exactly) -- i.e. we were discarding
# real, human-written pairs purely because of this cap, not because they
# didn't exist. Intent buckets average ~46 sentences/dialect (3574/77); 40
# lets nearly all of them through while still capping any single oversized
# intent from dominating the tier.
MAX_PAIRS_PER_DIRECTED_PAIR_PER_INTENT = 40


def load_pool():
    rows = []
    with open(OUT / "labeled_pool.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BAAI/bge-m3 on {device} ...")
    model = SentenceTransformer("BAAI/bge-m3", device=device)

    rows = load_pool()
    print(f"Loaded {len(rows)} labeled sentences")

    # group by (dialect, intent) -> list of texts
    buckets = {}
    for r in rows:
        key = (r["dialect"], r["intent_ar"])
        buckets.setdefault(key, []).append(r["text"])
    # dedupe within bucket
    for k in buckets:
        buckets[k] = sorted(set(buckets[k]))

    # embed everything once, keyed by (dialect, intent, text) -> vector
    all_texts = []
    index_of = {}
    for (dialect, intent), texts in buckets.items():
        for t in texts:
            key = (dialect, intent, t)
            if key not in index_of:
                index_of[key] = len(all_texts)
                all_texts.append(t)
    print(f"Embedding {len(all_texts)} unique (dialect,intent,text) sentences ...")
    embs = model.encode(all_texts, batch_size=128, show_progress_bar=True,
                         normalize_embeddings=True, convert_to_numpy=True)

    def get_emb(dialect, intent, text):
        return embs[index_of[(dialect, intent, text)]]

    intents = sorted(set(intent for (_, intent) in buckets.keys()))
    pairs = []
    for intent in intents:
        for src_d, tgt_d in itertools.permutations(LABELED_DIALECTS, 2):
            src_texts = buckets.get((src_d, intent), [])
            tgt_texts = buckets.get((tgt_d, intent), [])
            if not src_texts or not tgt_texts:
                continue
            src_emb = np.stack([get_emb(src_d, intent, t) for t in src_texts])
            tgt_emb = np.stack([get_emb(tgt_d, intent, t) for t in tgt_texts])
            sims = src_emb @ tgt_emb.T  # cosine sim, already normalized
            best_j = sims.argmax(axis=1)
            best_sim = sims.max(axis=1)
            order = np.argsort(-best_sim)[:MAX_PAIRS_PER_DIRECTED_PAIR_PER_INTENT]
            for i in order:
                s = best_sim[i]
                if s < MIN_SIM:
                    continue
                pairs.append({
                    "src_dialect": src_d, "tgt_dialect": tgt_d,
                    "src_text": src_texts[i], "tgt_text": tgt_texts[best_j[i]],
                    "intent_ar": intent, "cos_sim": round(float(s), 4),
                })

    with open(OUT / "silver_pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[silver] {len(pairs)} mined cross-dialect pairs written to silver_pairs.jsonl")
    # quick per-directed-pair summary
    from collections import Counter
    c = Counter((p["src_dialect"], p["tgt_dialect"]) for p in pairs)
    for k, v in sorted(c.items()):
        print(f"  {k[0]:>12} -> {k[1]:<12} {v}")


if __name__ == "__main__":
    main()
