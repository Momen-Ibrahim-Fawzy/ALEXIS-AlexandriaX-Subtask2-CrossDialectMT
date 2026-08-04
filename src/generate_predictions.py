"""
Stage 7: run the final model over the real blind test file and produce
predictions in the exact column format Codabench expects (per the task's
"Submission Format" section, decoded from the live competition page):

    Source dialect, Source sentence, Target dialect, Target sentence

One row per input row, same order, `Target sentence` filled with our
translation. Written to outputs/predictions/subtask2_predictions.csv (not yet
zipped -- see make_submission.py).

Run: CUDA_VISIBLE_DEVICES=1 conda run -n mo python generate_predictions.py [--adapter PATH]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.distributed.tensor  # noqa: F401 -- peft 0.20's PeftModel.from_pretrained checks
                                  # isinstance(x, torch.distributed.tensor.DTensor) but doesn't
                                  # import the submodule itself; in a non-distributed single-
                                  # process script it's never loaded otherwise, so the attribute
                                  # access raises AttributeError without this explicit import.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nllb_utils import translate, is_degenerate

TASK_ROOT = Path(__file__).resolve().parent
TEST_CSV = TASK_ROOT / "Data" / "subtask2_test_participants.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "predictions"
MODEL_NAME = "facebook/nllb-200-distilled-600M"
BATCH_SIZE = 8  # kept small -- this shared GPU's free memory fluctuates sharply
                # between other tenants' jobs; smaller batches survive contention
                # that killed an earlier batch=32 attempt with a CUDA OOM.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default=MODEL_NAME, help="NLLB backbone")
    ap.add_argument("--out-name", default="subtask2_predictions.csv")
    ap.add_argument("--num-beams", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, use_safetensors=True)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print(f"Loaded LoRA adapter from {args.adapter}")
    else:
        print("No adapter given -- running the zero-shot baseline.")
    model.to(device).eval()

    df = pd.read_csv(TEST_CSV)
    df.columns = [c.strip() for c in df.columns]
    df["Target sentence"] = ""

    n_flagged_total = 0
    n_fixed_total = 0
    # translate grouped by (source dialect, target dialect) for batching efficiency
    for (src_d, tgt_d), group in df.groupby(["Source dialect", "Target dialect"]):
        idx = group.index.tolist()
        texts = group["Source sentence"].astype(str).tolist()
        preds = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            outs = translate(model, tok, batch, src_d, tgt_d, device, num_beams=args.num_beams)
            preds.extend(outs)
        df.loc[idx, "Target sentence"] = preds
        print(f"{src_d:>12} -> {tgt_d:<12}: {len(texts)} rows translated")

    # QA safety net: retry any degenerate row one at a time with a harsher
    # repetition penalty + no beam search (greedy tends to escape the loop
    # that caused the degeneration in the first place). Never leaves a
    # flagged row unfixed without a fallback -- if the retry is STILL
    # degenerate, keep the shorter of the two candidates.
    flagged_idx = [i for i in df.index if is_degenerate(df.loc[i, "Target sentence"])]
    n_flagged_total = len(flagged_idx)
    if flagged_idx:
        print(f"\nQA pass: {n_flagged_total} degenerate output(s) flagged, retrying individually ...")
        for i in flagged_idx:
            src_d, tgt_d = df.loc[i, "Source dialect"], df.loc[i, "Target dialect"]
            src_text = str(df.loc[i, "Source sentence"])
            retry = translate(model, tok, [src_text], src_d, tgt_d, device,
                               num_beams=1, repetition_penalty=1.8, no_repeat_ngram_size=2)[0]
            if not is_degenerate(retry):
                df.loc[i, "Target sentence"] = retry
                n_fixed_total += 1
                print(f"  fixed  [{src_d}->{tgt_d}] {src_text[:40]!r} -> {retry[:60]!r}")
            else:
                original = df.loc[i, "Target sentence"]
                df.loc[i, "Target sentence"] = min([original, retry], key=len)
                print(f"  WARN still degenerate after retry [{src_d}->{tgt_d}]; kept shorter candidate: "
                      f"{df.loc[i, 'Target sentence'][:60]!r}")
    print(f"\nQA summary: {n_flagged_total} flagged, {n_fixed_total} fixed by retry, "
          f"{n_flagged_total - n_fixed_total} kept as shorter-candidate fallback.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / args.out_name
    df = df[["Source dialect", "Source sentence", "Target dialect", "Target sentence"]]
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(df)} predictions to {out_path}")


if __name__ == "__main__":
    main()
