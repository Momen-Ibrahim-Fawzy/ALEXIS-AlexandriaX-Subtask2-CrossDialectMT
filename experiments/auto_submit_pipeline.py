"""
Watches a training run's checkpoint directory and, for every NEW epoch
checkpoint that appears, generates predictions / runs the quality audits /
packages a submission on a separate GPU from whatever GPU training itself
is using, so packaging never competes with training for compute.

Keeps a small state file so it never reprocesses a checkpoint, and appends
each new submission to a shared SUBMISSIONS_INDEX.csv with an
auto-incrementing run id.

Configure via environment variables (see README.md):
  CKPT_DIR              directory of checkpoint-<step> subdirs to watch (required)
  LINEAGE_TAG           short label for this run, used in submission names (required)
  OUTPUTS_DIR           base dir for predictions/ and logs/ (default: ./outputs)
  SUBMISSIONS_DIR       where packaged submissions are collected (default: ./submissions)
  LINEAGE_START_EPOCH   offset if this run continues a numbered lineage (default: 1)
  PACKAGING_GPU         CUDA device index for packaging (default: 0)

Run: python auto_submit_pipeline.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", REPO_ROOT / "outputs"))
SRC_DIR = REPO_ROOT
PRED_DIR = OUTPUTS_DIR / "predictions"
LOG_DIR = OUTPUTS_DIR / "logs"
SUBMISSIONS_DIR = Path(os.environ.get("SUBMISSIONS_DIR", REPO_ROOT / "submissions"))
INDEX_CSV = SUBMISSIONS_DIR / "SUBMISSIONS_INDEX.csv"

CKPT_DIR = Path(os.environ["CKPT_DIR"])
LINEAGE_TAG = os.environ["LINEAGE_TAG"]
LINEAGE_START_EPOCH = int(os.environ.get("LINEAGE_START_EPOCH", "1"))
STATE_FILE = LOG_DIR / f"auto_submit_state_{LINEAGE_TAG}.json"
TRAIN_LOG = LOG_DIR / f"train_{LINEAGE_TAG}.log"
GPU = os.environ.get("PACKAGING_GPU", "0")

PY = sys.executable


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"processed": [], "next_submission_num": next_submission_num()}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def next_submission_num():
    nums = []
    if SUBMISSIONS_DIR.exists():
        for p in SUBMISSIONS_DIR.iterdir():
            if p.is_dir() and p.name[:3].isdigit():
                nums.append(int(p.name[:3]))
    return (max(nums) + 1) if nums else 1


def run(cmd, cwd=None, env_extra=None):
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def get_eval_loss(ckpt_dir):
    ts_path = ckpt_dir / "trainer_state.json"
    if not ts_path.exists():
        return None, None
    d = json.loads(ts_path.read_text())
    for e in reversed(d.get("log_history", [])):
        if "eval_loss" in e:
            return e["eval_loss"], e.get("epoch")
    return None, None


def quality_check(csv_path):
    import csv as csv_mod
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv_mod.DictReader(f))
    col = "Target sentence"
    n_underscore = sum(1 for r in rows if str(r.get(col, "")).lstrip().startswith("_"))
    n_empty = sum(1 for r in rows if not str(r.get(col, "")).strip())

    def is_degenerate(t):
        toks = t.split()
        if len(toks) < 6:
            return False
        return any(len(set(toks[i:i + 6])) == 1 for i in range(len(toks) - 5))

    n_degen = sum(1 for r in rows if is_degenerate(str(r.get(col, ""))))
    return {"total": len(rows), "underscore": n_underscore, "empty": n_empty, "degenerate": n_degen}


def process_checkpoint(ckpt_dir, state):
    step = ckpt_dir.name.split("-")[1]
    eval_loss, own_epoch = get_eval_loss(ckpt_dir)
    # trainer_state.json can lag behind the checkpoint directory's own creation
    # (HF Trainer writes model/optimizer files before the state json), so a
    # checkpoint caught the instant it appears can still be missing eval data.
    # Retry briefly rather than packaging a submission with eval_loss=None.
    wait_elapsed = 0
    while own_epoch is None and wait_elapsed < 60:
        time.sleep(3)
        wait_elapsed += 3
        eval_loss, own_epoch = get_eval_loss(ckpt_dir)
    if own_epoch is None:
        log(f"WARNING: {ckpt_dir.name} still has no eval_loss after {wait_elapsed}s wait -- proceeding anyway.")
    lineage_epoch = LINEAGE_START_EPOCH + int(own_epoch) - 1 if own_epoch else "?"
    log(f"=== New checkpoint {ckpt_dir.name} (own epoch {own_epoch}, lineage epoch {lineage_epoch}) eval_loss={eval_loss} ===")

    pred_csv = PRED_DIR / f"subtask2_predictions_{LINEAGE_TAG}_step{step}.csv"
    log("Generating predictions...")
    rc, out, err = run([
        PY, "-u", "generate_predictions.py",
        "--model", str(ckpt_dir),
        "--out-name", pred_csv.name,
        "--num-beams", "4",
    ], cwd=str(SRC_DIR))
    if rc != 0 or not pred_csv.exists():
        log(f"FAILED prediction generation (rc={rc}). stderr tail:\n{err[-1500:]}")
        return False

    qc = quality_check(pred_csv)
    log(f"Quality check: {qc}")
    if qc["empty"] > 0 or qc["degenerate"] > 5:
        log(f"QUALITY GATE FAILED (empty={qc['empty']}, degenerate={qc['degenerate']}) -- skipping packaging for this checkpoint.")
        return False

    log("Running dialect-marker audit...")
    rc, out, err = run([PY, "-u", "audit_dialect_markers.py", str(pred_csv)], cwd=str(SRC_DIR))
    log(out[-1200:] if out else "(no audit output)")

    num = state["next_submission_num"]
    tag = f"fullft_{LINEAGE_TAG}_ep{lineage_epoch}"
    notes = (
        f"Auto-packaged by the per-epoch pipeline. Continues the 010-continue lineage "
        f"(current best real result at the time: 24.440/38.259, submission 014) with a "
        f"generous 10-epoch budget and early-stopping patience=2 on eval_loss, LR=3e-5, "
        f"running solo on GPU1 while this packaging step runs on GPU0. This is lineage "
        f"epoch {lineage_epoch} (own-run epoch {own_epoch} of the extended run). "
        f"OFFLINE eval_loss={eval_loss}."
    )
    log(f"Packaging as submission {num:03d} ({tag})...")
    rc, out, err = run([
        PY, "-u", "make_submission.py", str(pred_csv),
        "--tag", tag, "--notes", notes,
    ], cwd=str(SRC_DIR))
    if rc != 0:
        log(f"FAILED make_submission.py (rc={rc}). stderr tail:\n{err[-1500:]}")
        return False
    log(out[-1500:])

    # find the folder make_submission.py just created and move/rename it
    subs_local = SRC_DIR.parent / "outputs" / "submissions"
    candidates = sorted(subs_local.glob(f"*{tag}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        log("Could not locate the freshly created submission folder -- leaving as-is.")
        return False
    created = candidates[0]
    final_name = f"{num:03d}_{tag}_{created.name.split('_')[-1]}"
    final_path = SUBMISSIONS_DIR / final_name
    created.rename(final_path)
    log(f"Moved to {final_path}")

    # append to the shared index
    import csv as csv_mod
    row = [
        final_name, tag, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        f"N/A (full fine-tune) -- {LINEAGE_TAG}, lineage epoch {lineage_epoch}",
        str(qc["total"]),
        f"Auto-packaged, eval_loss={eval_loss}, lineage epoch {lineage_epoch}.",
        "", "",
    ]
    with open(INDEX_CSV, "a", newline="", encoding="utf-8") as f:
        csv_mod.writer(f).writerow(row)
    log(f"READY: {final_path / 'subtask2_predictions.zip'}")

    state["next_submission_num"] = num + 1
    return True


def main():
    state = load_state()
    log(f"Auto-submit pipeline started. Watching {CKPT_DIR}. Next submission number: {state['next_submission_num']}")
    while True:
        if CKPT_DIR.exists():
            ckpts = sorted(
                [p for p in CKPT_DIR.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
                key=lambda p: int(p.name.split("-")[1]),
            )
            for ckpt in ckpts:
                if ckpt.name in state["processed"]:
                    continue
                ok = process_checkpoint(ckpt, state)
                state["processed"].append(ckpt.name)
                save_state(state)
                if not ok:
                    log(f"Checkpoint {ckpt.name} processed with issues -- see log above.")

        # stop watching once the training process itself is gone AND we've
        # drained every checkpoint currently on disk
        training_alive = subprocess.run(
            ["pgrep", "-f", f"train_full_ft.py.*--out-name {CKPT_DIR.name}"],
            capture_output=True,
        ).returncode == 0
        if not training_alive:
            log("Training process no longer running. Draining any final checkpoints, then exiting.")
            time.sleep(10)
            ckpts = sorted(
                [p for p in CKPT_DIR.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
                key=lambda p: int(p.name.split("-")[1]),
            ) if CKPT_DIR.exists() else []
            for ckpt in ckpts:
                if ckpt.name in state["processed"]:
                    continue
                process_checkpoint(ckpt, state)
                state["processed"].append(ckpt.name)
                save_state(state)
            log("Pipeline finished -- training has ended and all checkpoints processed.")
            break

        time.sleep(45)


if __name__ == "__main__":
    main()
