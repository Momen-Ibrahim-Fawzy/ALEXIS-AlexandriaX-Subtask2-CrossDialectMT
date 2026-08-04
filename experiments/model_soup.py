"""
Greedy model soup (Wortsman et al., arXiv:2203.05482) across the independently
-initialized full-FT lineages produced this session -- NOT checkpoint
averaging (epoch snapshots of ONE run, already tried and found negative).
Each candidate here started full fine-tuning from a DIFFERENT init:

  A: nllb600m_fullft_010continue_extended/checkpoint-8720
     init: v5-merged LoRA -> v4binit -> continued -> continued again.
     Current best, real Codabench spBLEU/chrF++ = 25.406 / 38.836.
  B: nllb600m_fullft_v4bcontinueinit/checkpoint-2180
     init: v4b-continue-merged LoRA. Real spBLEU/chrF++ = 23.378 / 37.470.
  C: nllb600m_fullft_v5init_lr5e5/model
     init: v5-merged LoRA, LR 5e-5, DIFFERENT finetune_train.jsonl snapshot.
     Real spBLEU/chrF++ = 22.655 / 37.228.
  D: nllb600m_fullft_v5init/model
     init: v5-merged LoRA, LR 2e-5. Never submitted -- offline-only.

Greedy procedure: start the soup at A (best known model). For each remaining
candidate in descending known-quality order, average it into the current
soup (weighted by how many models are already in it) and keep the merge
ONLY if it improves silver_dev MACRO spBLEU (the dev proxy closest to the
actual test distribution, per evaluate.py's own docstring) without
regressing gold_dev. Reject and discard otherwise.

Configure CANDIDATES below with paths to your own independently fine-tuned
checkpoints and their known real (or offline) scores -- see README.md.

Run: CUDA_VISIBLE_DEVICES=0 python model_soup.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nllb_utils import translate
from lang_codes import NLLB_TO_DIALECT
from spbleu_chrf import corpus_scores
from evaluate import eval_gold_dev, eval_by_direction, DATA

TOK_SOURCE = "facebook/nllb-200-distilled-600M"
REPO_ROOT = Path(__file__).resolve().parent

# (label, checkpoint_dir, known_score_or_None) -- ranked by known quality,
# highest first; unknowns are ranked last and evaluated offline before
# deciding their place in the greedy order. Replace with your own checkpoints.
CANDIDATES = [
    ("A_best_lineage", str(REPO_ROOT / "checkpoints" / "CHANGE_ME_A"), 25.406),
    ("B_second_lineage", str(REPO_ROOT / "checkpoints" / "CHANGE_ME_B"), 23.378),
    ("C_third_lineage", str(REPO_ROOT / "checkpoints" / "CHANGE_ME_C"), 22.655),
    ("D_fourth_lineage", str(REPO_ROOT / "checkpoints" / "CHANGE_ME_D"), None),
]

OUT_DIR = REPO_ROOT / "checkpoints" / "nllb600m_soup"
REPORT_PATH = REPO_ROOT / "outputs" / "model_soup_report.json"


def load_state(path):
    m = AutoModelForSeq2SeqLM.from_pretrained(path, use_safetensors=True)
    sd = {k: v.clone().float() for k, v in m.state_dict().items()}
    del m
    return sd


def score(state_dict, tok, device, label):
    model = AutoModelForSeq2SeqLM.from_pretrained(TOK_SOURCE, use_safetensors=True)
    model.load_state_dict({k: v.to(model.dtype) for k, v in state_dict.items()})
    model.to(device).eval()

    gold = eval_gold_dev(tok, model, device)
    silver = eval_by_direction(tok, model, device, DATA / "silver_dev.jsonl")
    gold_avg_spbleu = sum(v["spBLEU"] for v in gold.values()) / len(gold) if gold else None
    silver_macro = silver.get("MACRO_AVG", {})
    result = {
        "label": label,
        "gold_dev": gold,
        "gold_dev_avg_spBLEU": round(gold_avg_spbleu, 3) if gold_avg_spbleu else None,
        "silver_dev_MACRO_AVG": silver_macro,
    }
    print(f"[{label}] gold_dev avg spBLEU={result['gold_dev_avg_spBLEU']}  "
          f"silver_dev MACRO spBLEU={silver_macro.get('spBLEU')} chrF++={silver_macro.get('chrF++')}")
    del model
    torch.cuda.empty_cache()
    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TOK_SOURCE)
    report = {"candidates": [], "greedy_steps": []}

    print("=== Step 1: score each candidate individually (offline) ===")
    states = {}
    indiv_scores = {}
    for label, path, real_spbleu in CANDIDATES:
        print(f"\nLoading {label} from {path}")
        sd = load_state(path)
        states[label] = sd
        res = score(sd, tok, device, label)
        res["real_codabench_spBLEU"] = real_spbleu
        indiv_scores[label] = res
        report["candidates"].append(res)

    # sort candidates by known real score (unknowns last, ranked by offline silver_dev after)
    ranked = sorted(
        CANDIDATES,
        key=lambda c: (c[2] is None, -(c[2] or 0)),
    )
    # if D has no real score, re-rank it by its offline silver_dev macro spBLEU among the unknowns
    ranked_labels = [c[0] for c in ranked]
    print(f"\nGreedy order: {ranked_labels}")

    print("\n=== Step 2: greedy soup ===")
    base_label = ranked_labels[0]
    soup_state = {k: v.clone() for k, v in states[base_label].items()}
    soup_members = [base_label]
    current = indiv_scores[base_label]
    current_silver = current["silver_dev_MACRO_AVG"].get("spBLEU", 0)
    current_gold = current["gold_dev_avg_spBLEU"] or 0
    print(f"Soup base: {base_label} (silver spBLEU={current_silver}, gold spBLEU={current_gold})")

    for label in ranked_labels[1:]:
        k = len(soup_members)
        trial_state = {
            key: (soup_state[key] * k + states[label][key]) / (k + 1)
            for key in soup_state
        }
        trial = score(trial_state, tok, device, f"trial({'+'.join(soup_members)}+{label})")
        trial_silver = trial["silver_dev_MACRO_AVG"].get("spBLEU", 0)
        trial_gold = trial["gold_dev_avg_spBLEU"] or 0
        step = {
            "tried_adding": label, "soup_before": list(soup_members),
            "silver_before": current_silver, "silver_trial": trial_silver,
            "gold_before": current_gold, "gold_trial": trial_gold,
        }
        if trial_silver > current_silver and trial_gold >= current_gold - 0.5:
            step["decision"] = "ACCEPTED"
            soup_state = trial_state
            soup_members.append(label)
            current_silver, current_gold = trial_silver, trial_gold
            print(f"  ACCEPTED {label}: silver {step['silver_before']} -> {trial_silver}, "
                  f"gold {step['gold_before']} -> {trial_gold}")
        else:
            step["decision"] = "REJECTED"
            print(f"  REJECTED {label}: silver {step['silver_before']} -> {trial_silver} "
                  f"(would need improvement + gold within -0.5), gold {step['gold_before']} -> {trial_gold}")
        report["greedy_steps"].append(step)

    report["final_soup_members"] = soup_members
    report["final_silver_spBLEU"] = current_silver
    report["final_gold_spBLEU"] = current_gold
    print(f"\n=== Final soup: {soup_members} ===")
    print(f"silver_dev MACRO spBLEU={current_silver}  gold_dev avg spBLEU={current_gold}")

    if len(soup_members) > 1:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        final_model = AutoModelForSeq2SeqLM.from_pretrained(TOK_SOURCE, use_safetensors=True)
        final_model.load_state_dict({k: v.to(final_model.dtype) for k, v in soup_state.items()})
        final_model.save_pretrained(str(OUT_DIR), safe_serialization=True)
        tok.save_pretrained(str(OUT_DIR))
        print(f"Saved souped model to {OUT_DIR}")
        report["saved_to"] = str(OUT_DIR)
    else:
        print("No candidate improved on the base model -- soup rejected everything, nothing saved. "
              "The base model alone (already submission 018) remains the best option.")
        report["saved_to"] = None

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
