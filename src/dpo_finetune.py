"""
DPO-style preference fine-tuning (Rafailov et al. 2023) on top of the MLE
full-fine-tuned NLLB-600M generator, following the same recipe validated in
the sibling DialectSentEval 2026/Subtask 2 project's dpo_finetune.py (real
Codabench gains there, ~+1.5-2pp on their target metric once the reward was
well-calibrated -- see their SUBMISSIONS_LOG.md v3->v22 history).

v2 -- fixes two root causes diagnosed from v1's real, confirmed regression
(offline: gold_dev/silver_dev both down; real Codabench: 27.6->25.0 spBLEU):

  1. v1 trusted EVERY tier's tgt_text as a reliable "gold" reward reference,
     including bronze_fwd -- the 100%-synthetic, self-distilled tier that is
     the ONLY target-side data Egyptian/Lebanese have (verified: arz_Arab and
     apc_Arab have zero gold/silver rows in finetune_train.jsonl). This
     session already established that self-distilled bronze data hurts real
     Egyptian/Lebanese quality despite looking fine on offline dev metrics --
     v1's reward function ignored that finding and reinforced exactly this
     risk. Fix: the 4 dialects with real gold/silver data (Palestinian,
     Moroccan, Saudi, Tunisian) keep the v1 recipe (reward = chrF-to-
     reference + marker bonus, gold/silver tier only, bronze excluded).
     Egyptian/Lebanese switch to a reference-free reward: 0.6 * BGE-M3
     semantic similarity to the SOURCE sentence (faithfulness, independent of
     the untrustworthy bronze target text) + 0.4 * dialect-marker bonus.

  2. v1's reward was too redundant with what 20+ epochs of MLE cross-entropy
     already directly optimized -- gold was the top-reward pool member 99.3%
     of the time, so DPO mostly re-taught the model to prefer the exact
     reference text it was already trained toward, a weak/noisy signal
     (preference_accuracy plateaued at only ~0.68). Fix: raise DPO_BETA
     0.1->0.3 so the policy is kept closer to the known-good MLE reference,
     reducing how much damage a still-imperfect reward can do.

CRITICAL (learned the hard way in the sibling project): DPO log-prob math
MUST be computed in fp32. bf16's ~0.39% relative-precision floor is coarser
than a typical 1e-5-LR weight update (~0.02% relative) -- every update
silently rounds to zero in bf16, and a full multi-epoch run can complete
with preference_accuracy stuck at random chance before anyone notices.

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n mo python3 dpo_finetune.py --base-model ... --out-name nllb600m_dpo_v2
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nllb_utils import build_ids, pad_batch
from lang_codes import NLLB_TO_DIALECT, TEST_DIALECTS
from spbleu_chrf import get_metrics
from audit_dialect_markers import MARKERS, simple_tokenize

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "outputs"
CKPT_ROOT = OUT / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

DPO_SUBSET_PER_DIALECT = 700     # -> ~4200 total across the 6 graded dialects
NUM_SAMPLES = 4                  # sampled candidates per example, plus gold = pool size 5
DPO_EPOCHS = 7                   # matches the sibling project's validated v3/v6-era recipe
DPO_LR = 1e-5                    # much smaller than the 3e-5 used for MLE full-FT
DPO_BATCH_SIZE = 6
DPO_BETA = 0.3                   # v2: raised from v1's 0.1 -- keeps the policy closer to
                                  # the MLE reference, limiting how much damage an imperfect
                                  # reward can do (see module docstring, root cause #2)
CHRF_WEIGHT = 0.85
MARKER_WEIGHT = 0.15
SEMANTIC_WEIGHT = 0.6            # for reference-free dialects (Egyptian/Lebanese) only
UNREF_MARKER_WEIGHT = 0.4        # for reference-free dialects (Egyptian/Lebanese) only
# Dialects with real gold/silver target-side data (trustworthy reward reference).
# Egyptian (arz_Arab) and Lebanese (apc_Arab) are 100% bronze_fwd (synthetic,
# self-distilled) -- see module docstring, root cause #1 -- and use a
# reference-free reward instead (build_preference_pairs / REFERENCE_FREE_DIALECTS).
REFERENCE_FREE_DIALECTS = {"Egyptian", "Lebanese"}
MAX_SRC_LEN = 96
MAX_NEW_TOKENS = 64


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def build_dpo_subset():
    """Stratified sample: only the 6 dialects actually graded on the blind
    test as TARGET, equal cap per dialect so Egyptian/Lebanese (zero real
    training data) get the same representation as the labeled dialects.

    For the 4 dialects with real gold/silver data, bronze rows are excluded
    even where present (e.g. Palestinian's bronze_rev slice) -- v2 only
    trusts a tgt_text as a reward reference when it's real, non-synthetic
    text (see module docstring, root cause #1). Egyptian/Lebanese are 100%
    bronze_fwd with no alternative, so their tgt_text is never used as a
    reward reference at all (build_preference_pairs uses the reference-free
    path for REFERENCE_FREE_DIALECTS instead) -- their rows are kept here
    purely as a source of (src_text, src_lang, tgt_lang) to translate."""
    recs = read_jsonl(DATA / "finetune_train.jsonl")
    from lang_codes import DIALECT_TO_NLLB
    graded_codes = {DIALECT_TO_NLLB[d] for d in TEST_DIALECTS}
    reference_free_codes = {DIALECT_TO_NLLB[d] for d in REFERENCE_FREE_DIALECTS}
    by_lang = {}
    for r in recs:
        if r["tgt_lang"] not in graded_codes:
            continue
        if r["tgt_lang"] not in reference_free_codes and r["tier"] not in ("gold", "silver"):
            continue  # exclude bronze for dialects where a trustworthy reference exists
        by_lang.setdefault(r["tgt_lang"], []).append(r)
    rng = random.Random(SEED)
    subset = []
    for code, items in by_lang.items():
        rng.shuffle(items)
        subset.extend(items[:DPO_SUBSET_PER_DIALECT])
    # NOT shuffled further: sample_candidates() below requires each generation
    # batch to share one target language, so the subset stays grouped in
    # per-dialect blocks. Training-time batch order is randomized separately
    # by the DataLoader(shuffle=True) in main(), so this has no effect on
    # what the model actually trains on.
    return subset


@torch.no_grad()
def sample_candidates(model, tok, records, device, num_samples=NUM_SAMPLES, batch_size=16):
    """Diverse sampled candidates per example, using the same manual (bug-safe)
    src/tgt language-code encoding as train_full_ft.py / nllb_utils.translate.

    forced_bos_token_id must be constant per generate() call, so records are
    grouped by target language HERE (not assumed from caller ordering -- a
    per-dialect cap that isn't a multiple of batch_size would otherwise let a
    batch straddle two dialects' blocks and crash mid-run). Results are
    returned in the same order as the input `records`."""
    by_lang_idx = {}
    for idx, r in enumerate(records):
        by_lang_idx.setdefault(r["tgt_lang"], []).append(idx)

    results = [None] * len(records)
    for tgt_lang, idxs in by_lang_idx.items():
        tgt_id = tok.convert_tokens_to_ids(tgt_lang)
        for i in range(0, len(idxs), batch_size):
            batch_idxs = idxs[i:i + batch_size]
            batch = [records[k] for k in batch_idxs]
            id_lists = [build_ids(tok, r["src_text"], r["src_lang"], max_len=MAX_SRC_LEN) for r in batch]
            input_ids, attn = pad_batch(id_lists, tok.pad_token_id)
            input_ids, attn = input_ids.to(device), attn.to(device)
            gen = model.generate(
                input_ids=input_ids, attention_mask=attn, forced_bos_token_id=tgt_id,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=True, top_p=0.92, temperature=1.0,
                num_return_sequences=num_samples,
            )
            decoded = tok.batch_decode(gen, skip_special_tokens=True)
            for j, k in enumerate(batch_idxs):
                cands = decoded[j * num_samples:(j + 1) * num_samples]
                results[k] = [c if c.strip() else records[k]["tgt_text"] for c in cands]
    return results


def marker_bonus(text, target_dialect):
    toks = simple_tokenize(text)
    if not toks:
        return 0.0
    markers = set(MARKERS.get(target_dialect, []))
    hits = sum(1 for t in toks if t in markers)
    return min(hits / len(toks) * 10.0, 1.0)


_EMBEDDER = None


def _load_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("BAAI/bge-m3", device=str(DEVICE))
    return _EMBEDDER


def compute_semantic_scores(sources, cand_lists):
    """BGE-M3 cosine similarity between each source and its candidates --
    used ONLY for the reference-free dialects (Egyptian/Lebanese), where we
    don't trust the bronze tgt_text enough to score against it directly.
    Measures faithfulness to the SOURCE, independent of any target-side
    reference, so it can't reinforce the self-distillation risk v1 hit."""
    embedder = _load_embedder()
    flat_texts, spans = [], []
    for src, cands in zip(sources, cand_lists):
        start = len(flat_texts)
        flat_texts.append(src)
        flat_texts.extend(cands)
        spans.append((start, start + 1 + len(cands)))
    embeddings = embedder.encode(flat_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    all_scores = []
    for start, end in spans:
        src_vec = embeddings[start]
        cand_vecs = embeddings[start + 1:end]
        sims = cand_vecs @ src_vec  # both L2-normalized -> cosine similarity, in [-1, 1]
        all_scores.append(((sims + 1) / 2).tolist())  # rescale to [0, 1]
    return all_scores


def build_preference_pairs(records, sampled):
    """Two reward paths, branched per dialect (see module docstring):
      - referenced dialects (Palestinian/Moroccan/Saudi/Tunisian): pool =
        {gold} u {K sampled}, reward = chrF-to-gold + marker bonus.
      - reference-free dialects (Egyptian/Lebanese): pool = {K sampled} only
        (gold/bronze text never enters the reward), reward = BGE-M3
        semantic-similarity-to-SOURCE + marker bonus."""
    bleu, chrf = get_metrics()
    chosen, rejected = [None] * len(records), [None] * len(records)
    n_gold_chosen = 0
    n_referenced = 0

    ref_idxs, unref_idxs = [], []
    for i, rec in enumerate(records):
        if NLLB_TO_DIALECT[rec["tgt_lang"]] in REFERENCE_FREE_DIALECTS:
            unref_idxs.append(i)
        else:
            ref_idxs.append(i)

    for i in ref_idxs:
        rec, cands = records[i], sampled[i]
        gold = rec["tgt_text"]
        target_dialect = NLLB_TO_DIALECT[rec["tgt_lang"]]
        pool = [gold] + cands
        chrf_scores = [chrf.sentence_score(c, [gold]).score / 100.0 for c in pool]
        marker_scores = [marker_bonus(c, target_dialect) for c in pool]
        rewards = [CHRF_WEIGHT * cs + MARKER_WEIGHT * ms for cs, ms in zip(chrf_scores, marker_scores)]
        best_idx = max(range(len(pool)), key=lambda k: rewards[k])
        worst_idx = min(range(len(pool)), key=lambda k: rewards[k])
        chosen[i] = pool[best_idx]
        rejected[i] = pool[worst_idx]
        n_referenced += 1
        if best_idx == 0:
            n_gold_chosen += 1

    if unref_idxs:
        sources = [records[i]["src_text"] for i in unref_idxs]
        cand_lists = [sampled[i] for i in unref_idxs]
        sem_scores = compute_semantic_scores(sources, cand_lists)
        for i, sems in zip(unref_idxs, sem_scores):
            rec, cands = records[i], sampled[i]
            target_dialect = NLLB_TO_DIALECT[rec["tgt_lang"]]
            marker_scores = [marker_bonus(c, target_dialect) for c in cands]
            rewards = [SEMANTIC_WEIGHT * s + UNREF_MARKER_WEIGHT * m for s, m in zip(sems, marker_scores)]
            best_idx = max(range(len(cands)), key=lambda k: rewards[k])
            worst_idx = min(range(len(cands)), key=lambda k: rewards[k])
            chosen[i] = cands[best_idx]
            rejected[i] = cands[worst_idx]

    print(f"Preference pairs built. Referenced dialects: {n_referenced} pairs, gold was "
          f"chosen in {n_gold_chosen}/{n_referenced} ({n_gold_chosen/max(n_referenced,1):.1%}). "
          f"Reference-free dialects (Egyptian/Lebanese): {len(unref_idxs)} pairs, "
          f"scored by source-semantic-similarity + marker bonus only (no bronze text trusted).")
    return chosen, rejected


class PreferenceDataset(Dataset):
    def __init__(self, records, chosen, rejected, tok):
        self.records = records
        self.chosen = chosen
        self.rejected = rejected
        self.tok = tok

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        input_ids = build_ids(self.tok, rec["src_text"], rec["src_lang"], max_len=MAX_SRC_LEN)
        chosen_ids = build_ids(self.tok, self.chosen[idx], rec["tgt_lang"], max_len=MAX_NEW_TOKENS)
        rejected_ids = build_ids(self.tok, self.rejected[idx], rec["tgt_lang"], max_len=MAX_NEW_TOKENS)
        return {"input_ids": input_ids, "chosen_ids": chosen_ids, "rejected_ids": rejected_ids}


def collate(batch, pad_id):
    input_ids, input_attn = pad_batch([b["input_ids"] for b in batch], pad_id)
    chosen_ids, chosen_attn = pad_batch([b["chosen_ids"] for b in batch], pad_id)
    rejected_ids, rejected_attn = pad_batch([b["rejected_ids"] for b in batch], pad_id)
    return {
        "input_ids": input_ids, "input_attn": input_attn,
        "chosen_ids": chosen_ids, "chosen_attn": chosen_attn,
        "rejected_ids": rejected_ids, "rejected_attn": rejected_attn,
    }


def sequence_logprob(model, input_ids, input_attn, label_ids, pad_id):
    labels = label_ids.clone()
    labels[labels == pad_id] = -100
    out = model(input_ids=input_ids, attention_mask=input_attn, labels=labels)
    logits = out.logits.float()  # upcast before log_softmax -- see module docstring:
    # bf16 precision is coarser than a 1e-5-LR update, silent no-op otherwise.
    logp = F.log_softmax(logits, dim=-1)
    token_logp = torch.gather(logp, 2, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (labels != -100).float()
    return (token_logp * mask).sum(dim=1)


def dpo_loss(policy, ref, batch, pad_id):
    input_ids, input_attn = batch["input_ids"].to(DEVICE), batch["input_attn"].to(DEVICE)
    chosen_ids = batch["chosen_ids"].to(DEVICE)
    rejected_ids = batch["rejected_ids"].to(DEVICE)

    policy_chosen = sequence_logprob(policy, input_ids, input_attn, chosen_ids, pad_id)
    policy_rejected = sequence_logprob(policy, input_ids, input_attn, rejected_ids, pad_id)
    with torch.no_grad():
        ref_chosen = sequence_logprob(ref, input_ids, input_attn, chosen_ids, pad_id)
        ref_rejected = sequence_logprob(ref, input_ids, input_attn, rejected_ids, pad_id)

    policy_logratio = policy_chosen - policy_rejected
    ref_logratio = ref_chosen - ref_rejected
    logits = DPO_BETA * (policy_logratio - ref_logratio)
    loss = -F.logsigmoid(logits).mean()
    acc = (logits > 0).float().mean().item()
    return loss, acc


def main(base_model_dir, out_name="nllb600m_dpo_v1", epochs=DPO_EPOCHS, batch_size=DPO_BATCH_SIZE):
    seed_everything()
    print(f"Device: {DEVICE}, base model: {base_model_dir}, out: {out_name}")

    tok = AutoTokenizer.from_pretrained(base_model_dir)
    policy = AutoModelForSeq2SeqLM.from_pretrained(base_model_dir, use_safetensors=True).to(DEVICE)
    ref = AutoModelForSeq2SeqLM.from_pretrained(base_model_dir, use_safetensors=True).to(DEVICE)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    subset = build_dpo_subset()
    print(f"DPO preference-pair mining on {len(subset)} training examples "
          f"({NUM_SAMPLES} sampled candidates + gold each, stratified across "
          f"the {len(TEST_DIALECTS)} graded dialects)...")

    policy.eval()
    # sample_candidates requires homogeneous-target-language batches; subset is
    # already dialect-grouped by build_dpo_subset()'s construction order.
    sampled = sample_candidates(policy, tok, subset, DEVICE)
    chosen, rejected = build_preference_pairs(subset, sampled)

    global _EMBEDDER
    if _EMBEDDER is not None:
        del _EMBEDDER
        _EMBEDDER = None
        torch.cuda.empty_cache()  # free BGE-M3 before the policy/ref/optimizer occupy the GPU

    ds = PreferenceDataset(subset, chosen, rejected, tok)
    pad_id = tok.pad_token_id
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                         collate_fn=lambda b: collate(b, pad_id))

    optimizer = torch.optim.AdamW(policy.parameters(), lr=DPO_LR, weight_decay=0.0)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    policy.train()
    for epoch in range(epochs):
        total_loss, total_acc, n = 0.0, 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            loss, acc = dpo_loss(policy, ref, batch, pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_acc += acc
            n += 1
        print(f"[dpo] epoch {epoch+1}/{epochs} loss={total_loss/n:.4f} "
              f"preference_accuracy={total_acc/n:.3f} (fraction of batches where "
              f"policy correctly prefers chosen)", flush=True)

    out_dir = CKPT_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))
    print(f"Saved DPO-tuned model to {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out-name", default="nllb600m_dpo_v1")
    ap.add_argument("--epochs", type=int, default=DPO_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DPO_BATCH_SIZE)
    args = ap.parse_args()
    main(args.base_model, out_name=args.out_name, epochs=args.epochs, batch_size=args.batch_size)
