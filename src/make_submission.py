"""
Stage 8: validate a predictions CSV against the Codabench submission spec,
zip it, and bundle it with a self-contained "system card" documenting exactly
how that specific submission was produced -- model, data, hyperparameters,
dev-set scores, timestamp. Every submission gets its own numbered, timestamped
folder under outputs/submissions/ so a multi-submission run (this task allows
up to 10 total, 3/day) leaves a full paper-ready trail instead of overwriting
itself.

Spec (decoded from the live Codabench "Participate" page, see
EDA_INSIGHTS.md §1): CSV with exact headers `Source dialect, Source sentence,
Target dialect, Target sentence`, compressed as a ZIP file.

Usage:
  conda run -n mo python make_submission.py <predictions_csv> \
      [--tag NAME] [--adapter PATH] [--notes "free text"]

Writes outputs/submissions/<NNN>_<tag>_<UTCtimestamp>/
    subtask2_predictions.csv
    subtask2_predictions.zip   <- upload this one to Codabench
    system_card.md
    system_card.json
also appends one row to outputs/submissions/SUBMISSIONS_INDEX.csv so every
attempt is visible in one place.
"""
import argparse
import csv
import datetime
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

TASK_ROOT = Path(__file__).resolve().parent
TEST_CSV = TASK_ROOT / "Data" / "subtask2_test_participants.csv"
SYS_ROOT = Path(__file__).resolve().parent
DEFAULT_PRED = SYS_ROOT / "outputs" / "predictions" / "subtask2_predictions.csv"
SUB_ROOT = SYS_ROOT / "outputs" / "submissions"
DATA_DIR = SYS_ROOT / "data"

REQUIRED_COLS = ["Source dialect", "Source sentence", "Target dialect", "Target sentence"]


def next_run_number():
    SUB_ROOT.mkdir(parents=True, exist_ok=True)
    existing = [p for p in SUB_ROOT.iterdir() if p.is_dir()]
    nums = []
    for p in existing:
        head = p.name.split("_")[0]
        if head.isdigit():
            nums.append(int(head))
    return (max(nums) + 1) if nums else 1


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_all_eval_reports():
    """Every outputs/dev_eval_report_<tag>.json found, keyed by <tag> -- as many
    candidate models as have been evaluated (baseline, v1, v2_best,
    v2_averaged, ...), not just two hardcoded ones. Lets the system card show
    the full comparison table regardless of how many iterations a task went
    through, without needing to touch this script each time."""
    reports = {}
    for p in sorted((SYS_ROOT / "outputs").glob("dev_eval_report_*.json")):
        tag = p.stem[len("dev_eval_report_"):]
        reports[tag] = load_json(p)
    return reports


def build_system_card(run_dir_name, tag, notes, adapter_path, pred_path, n_rows, per_pair_counts):
    corpus_stats = load_json(DATA_DIR / "corpus_stats.json")
    run_config = load_json(Path(adapter_path).parent / "run_config.json") if adapter_path else None
    eval_reports = load_all_eval_reports()

    card = {
        "submission_id": run_dir_name,
        "tag": tag,
        "notes": notes,
        "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "task": "AlexandriaX-2026 Subtask 2 -- DialectMTBench (Cross-Dialect Arabic MT)",
        "codabench_competition": "https://www.codabench.org/competitions/16339/",
        "backbone_model": "facebook/nllb-200-distilled-600M",
        "adapter_used": str(adapter_path) if adapter_path else "none (zero-shot baseline)",
        "n_predictions": n_rows,
        "predictions_per_source_target_pair": per_pair_counts,
        "training_corpus_composition": corpus_stats,
        "training_run_config": run_config,
        "dev_eval_reports": eval_reports,
        "data_sources": {
            "train_val_proxy": "SinaLab/ArBanking77 (public GitHub release), "
                                "https://github.com/SinaLab/ArBanking77",
            "test": "Data/subtask2_test_participants.csv (organizer-released blind test)",
        },
        "reference_docs": {
            "eda_insights": "EDA/EDA_INSIGHTS.md",
            "research_notes": "System/RESEARCH_NOTES.md",
            "system_readme": "System/README.md",
        },
    }
    return card


def system_card_markdown(card):
    lines = [f"# System card — submission `{card['submission_id']}`", ""]
    lines.append(f"- **Created (UTC):** {card['created_at_utc']}")
    lines.append(f"- **Tag:** {card['tag']}")
    if card["notes"]:
        lines.append(f"- **Notes:** {card['notes']}")
    lines.append(f"- **Task:** {card['task']}")
    lines.append(f"- **Backbone model:** {card['backbone_model']}")
    lines.append(f"- **Adapter:** {card['adapter_used']}")
    lines.append(f"- **Predictions:** {card['n_predictions']} rows")
    lines.append("")

    rc = card.get("training_run_config")
    if rc:
        lines.append("## Training configuration")
        lines.append(f"- Method: {rc.get('method')}")
        lc = rc.get("lora_config", {})
        lines.append(f"- LoRA: r={lc.get('r')}, alpha={lc.get('lora_alpha')}, "
                      f"dropout={lc.get('lora_dropout')}, targets={lc.get('target_modules')}")
        lines.append(f"- Trainable / total params: {rc.get('trainable_params'):,} / {rc.get('total_params'):,}")
        ta = rc.get("training_args", {})
        lines.append(f"- Effective batch size: {ta.get('effective_batch_size')} "
                      f"(per-device={ta.get('per_device_train_batch_size')} x "
                      f"grad-accum={ta.get('gradient_accumulation_steps')})")
        lines.append(f"- LR={ta.get('learning_rate')}, epochs={ta.get('num_train_epochs')}, "
                      f"bf16={ta.get('bf16')}, max_seq_len={ta.get('max_seq_len')}")
        lines.append(f"- Final training loss: {rc.get('final_train_loss')}")
        lines.append(f"- Trained at (UTC): {rc.get('trained_at_utc')}")
        lines.append("")

    cs = card.get("training_corpus_composition")
    if cs:
        lines.append("## Training corpus composition")
        lines.append(f"- Total examples: {cs.get('total_examples')}")
        lines.append(f"- Tier counts: {cs.get('tier_counts')}")
        lines.append(f"- Distinct (src,tgt) directions covered: {cs.get('n_directions')}")
        lines.append("")

    reports = card.get("dev_eval_reports") or {}
    if reports:
        lines.append("## Dev evaluation -- all candidate models compared")
        lines.append(
            "> **Caveat on `soft_dev`:** its Egyptian/Lebanese \"references\" were themselves "
            "generated by the zero-shot backbone, so a from-scratch zero-shot eval trivially "
            "reproduces them exactly (100.0/100.0) -- an artifact, not a real quality result. "
            "The metric only becomes meaningful once a model's weights actually differ from the "
            "one that generated those references (any fine-tuned candidate). `gold_dev` (real, "
            "human-translated MSA<->Palestinian) and `silver_dev` (mined, real-native-text, "
            "leak-checked, covering all 20 directions among the 5 labeled dialects -- most of "
            "the actual test-scoring pairs) are unaffected and are the trustworthy numbers "
            "throughout. See EDA_INSIGHTS.md / System/README.md for the full explanation."
        )
        lines.append("")
        lines.append("| Model | gold MSA→PAL spBLEU | gold PAL→MSA spBLEU | silver_dev macro spBLEU/chrF++ | soft_dev macro spBLEU/chrF++ |")
        lines.append("|---|---|---|---|---|")
        for eval_tag, ev in reports.items():
            gd = ev.get("gold_dev", {})
            sd = ev.get("silver_dev", {}).get("MACRO_AVG", {})
            bd = ev.get("soft_dev", {}).get("MACRO_AVG", {})
            m2p = gd.get("MSA->Palestinian", {}).get("spBLEU", "-")
            p2m = gd.get("Palestinian->MSA", {}).get("spBLEU", "-")
            sd_s = f"{sd.get('spBLEU', '-')} / {sd.get('chrF++', '-')}" if sd else "-"
            bd_s = f"{bd.get('spBLEU', '-')} / {bd.get('chrF++', '-')}" if bd else "-"
            lines.append(f"| {eval_tag} | {m2p} | {p2m} | {sd_s} | {bd_s} |")
        lines.append("")

        for eval_tag, ev in reports.items():
            if not ev:
                continue
            lines.append(f"### Full report: `{eval_tag}`")
            lines.append("```json")
            lines.append(json.dumps({k: v for k, v in ev.items() if k != "qualitative_samples"},
                                     ensure_ascii=False, indent=2))
            lines.append("```")
            if ev.get("qualitative_samples"):
                lines.append("Qualitative samples (MSA source, all 6 test dialects):")
                for s in ev["qualitative_samples"]:
                    lines.append(f"- **{s['tgt_dialect']}**: {s['hyp']}")
            lines.append("")

    lines.append("## Predictions per (source dialect -> target dialect) pair")
    lines.append("")
    lines.append("| Source | Target | n |")
    lines.append("|---|---|---|")
    for key, n in sorted(card["predictions_per_source_target_pair"].items()):
        s, t = key.split(" -> ")
        lines.append(f"| {s} | {t} | {n} |")
    lines.append("")

    lines.append("## Data sources")
    for k, v in card["data_sources"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Reference docs (full reasoning trail)")
    for k, v in card["reference_docs"].items():
        lines.append(f"- `{v}`")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions_csv", nargs="?", default=str(DEFAULT_PRED))
    ap.add_argument("--tag", default="run", help="short label, e.g. 'lora_v1', 'zeroshot_baseline'")
    ap.add_argument("--adapter", default=None, help="path to the LoRA adapter used, if any")
    ap.add_argument("--notes", default="", help="free-text notes for the system card")
    args = ap.parse_args()

    pred_path = Path(args.predictions_csv)
    assert pred_path.exists(), f"predictions file not found: {pred_path}"

    pred = pd.read_csv(pred_path)
    pred.columns = [c.strip() for c in pred.columns]
    test = pd.read_csv(TEST_CSV)
    test.columns = [c.strip() for c in test.columns]

    # --- validation against the exact Codabench spec ---
    assert list(pred.columns) == REQUIRED_COLS, f"columns must be exactly {REQUIRED_COLS}, got {list(pred.columns)}"
    assert len(pred) == len(test), f"row count mismatch: predictions={len(pred)} test={len(test)}"
    assert (pred["Source sentence"].astype(str).values == test["Source sentence"].astype(str).values).all(), \
        "row order / source sentences don't match the test file -- did you keep original order?"
    assert (pred["Source dialect"].values == test["Source dialect"].values).all()
    assert (pred["Target dialect"].values == test["Target dialect"].values).all()
    n_empty = (pred["Target sentence"].astype(str).str.strip() == "").sum()
    assert n_empty == 0, f"{n_empty} rows have an empty Target sentence"
    print(f"Validation OK: {len(pred)} rows, columns {REQUIRED_COLS}, no empty predictions.")

    # --- versioned bundle ---
    run_num = next_run_number()
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir_name = f"{run_num:03d}_{args.tag}_{ts}"
    run_dir = SUB_ROOT / run_dir_name
    run_dir.mkdir(parents=True)

    csv_out = run_dir / "subtask2_predictions.csv"
    pred.to_csv(csv_out, index=False, encoding="utf-8-sig")
    zip_out = run_dir / "subtask2_predictions.zip"
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_out, arcname=csv_out.name)

    per_pair = pred.groupby(["Source dialect", "Target dialect"]).size()
    per_pair_counts = {str(k): int(v) for k, v in per_pair.items()}

    card = build_system_card(run_dir_name, args.tag, args.notes, args.adapter, csv_out, len(pred), per_pair)
    card["predictions_per_source_target_pair"] = {f"{k[0]} -> {k[1]}": v for k, v in per_pair.items()}
    (run_dir / "system_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2, default=str),
                                                encoding="utf-8")
    (run_dir / "system_card.md").write_text(system_card_markdown(card), encoding="utf-8")

    # append to a flat index so all submissions are visible at a glance
    index_path = SUB_ROOT / "SUBMISSIONS_INDEX.csv"
    is_new = not index_path.exists()
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["run_id", "tag", "created_at_utc", "adapter", "n_predictions", "notes"])
        w.writerow([run_dir_name, args.tag, card["created_at_utc"], args.adapter or "", len(pred), args.notes])

    print(f"\nWrote {csv_out}")
    print(f"Wrote {zip_out}  <-- upload this file to Codabench")
    print(f"Wrote {run_dir / 'system_card.md'} / .json  (full reproducibility record)")
    print(f"Updated {index_path}")
    print(f"\n=== SUBMISSION READY: {zip_out} ===")


if __name__ == "__main__":
    main()
