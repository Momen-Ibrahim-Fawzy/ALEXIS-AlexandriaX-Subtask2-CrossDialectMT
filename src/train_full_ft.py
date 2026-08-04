"""
v-fullft experiment: full fine-tuning (not LoRA) as a genuinely different
capacity lever, after three consecutive LoRA data-recipe iterations (v3, v4,
v5) each produced real but shrinking real-world gains (+3.8%, +2.7%, +1.8%
relative spBLEU) -- a decelerating curve suggesting LoRA's frozen-base-model
capacity, not the data recipe, may now be the binding constraint.

Initialized from v5 (LoRA-merged into the base weights) rather than from
scratch, specifically to reduce catastrophic-forgetting risk to Egyptian/
Lebanese (~39% of test rows, zero real training data, entirely dependent on
the base model's pretrained zero-shot competence + synthetic self-distilled
signal) -- starting from an already-adapted checkpoint and doing a SHORT,
low-LR continuation is a much smaller step than training all 600M parameters
from the raw pretrained backbone, while still unlocking capacity LoRA can't
reach.

Run: CUDA_VISIBLE_DEVICES=0 conda run -n mo python train_full_ft.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback,
)
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nllb_utils import build_ids

MODEL_NAME = "facebook/nllb-200-distilled-600M"
DATA = Path(__file__).resolve().parent / "data"
LOG_DIR = Path(__file__).resolve().parent / "outputs" / "logs"
CKPT_ROOT = Path(__file__).resolve().parent / "outputs" / "checkpoints"
MAX_LEN = 96


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def build_dataset(tok, records):
    def _map(ex):
        input_ids = build_ids(tok, ex["src_text"], ex["src_lang"], max_len=MAX_LEN)
        labels = build_ids(tok, ex["tgt_text"], ex["tgt_lang"], max_len=MAX_LEN)
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    ds = Dataset.from_list(records)
    ds = ds.map(_map, remove_columns=ds.column_names, desc="tokenizing")
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-adapter", default=None, help="LoRA adapter to merge in as the starting point")
    ap.add_argument("--continue-from-model", default=None,
                     help="continue training an existing FULL (non-LoRA) fine-tuned checkpoint dir "
                          "(mutually exclusive with --init-adapter) -- e.g. to keep training a full-FT "
                          "run whose eval_loss was still dropping when it hit its epoch budget")
    ap.add_argument("--out-name", default="nllb600m_fullft_v5init")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--resume", action="store_true",
                     help="resume the SAME --epochs run from the latest checkpoint already in "
                          "--out-name's output dir (optimizer/scheduler/RNG state included), "
                          "rather than starting a fresh run")
    ap.add_argument("--early-stopping-patience", type=int, default=None,
                     help="stop once eval_loss fails to improve for this many consecutive "
                          "evals (each eval = one epoch here). Pair with a generous --epochs "
                          "budget so the run stops itself instead of needing to be watched; "
                          "load_best_model_at_end=True guarantees the final saved model is "
                          "the best epoch seen, not just the last one.")
    args = ap.parse_args()
    assert bool(args.init_adapter) != bool(args.continue_from_model), \
        "pass exactly one of --init-adapter or --continue-from-model"
    OUT_DIR = CKPT_ROOT / args.out_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, out: {OUT_DIR}, lr: {args.lr}, epochs: {args.epochs}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if args.continue_from_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.continue_from_model, use_safetensors=True)
        print(f"Continuing full-FT from {args.continue_from_model}")
    else:
        base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=True)
        model = PeftModel.from_pretrained(base, args.init_adapter).merge_and_unload()
        print(f"Initialized from merged {args.init_adapter}")
    # merge_and_unload() returns the base model with every parameter still
    # frozen (requires_grad=False) -- PeftModel.from_pretrained froze the
    # base model when wrapping it, and merging/unloading never undoes that.
    # Without this, the "full fine-tune" silently trains zero parameters:
    # loss logs fine, grad_norm reads 0.0 on every step, and the saved
    # checkpoint is byte-identical to the init weights (confirmed via
    # md5sum across epoch checkpoints on a prior run). A plain from_pretrained
    # load (the --continue-from-model path) doesn't have this issue since
    # there's no PeftModel wrapping involved, but setting this explicitly
    # either way costs nothing and removes any doubt.
    model.requires_grad_(True)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable == n_params, f"expected all {n_params:,} params trainable, got {n_trainable:,}"
    print(f"Full fine-tune: all {n_params:,} parameters trainable (vs LoRA's ~1-3%)")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.to(device)

    train_records = read_jsonl(DATA / "finetune_train.jsonl")
    print(f"Loaded {len(train_records)} training examples")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_records))
    n_eval = min(400, max(50, len(train_records) // 100))
    eval_idx = set(idx[:n_eval].tolist())
    eval_records = [r for i, r in enumerate(train_records) if i in eval_idx]
    fit_records = [r for i, r in enumerate(train_records) if i not in eval_idx]

    train_ds = build_dataset(tok, fit_records)
    eval_ds = build_dataset(tok, eval_records)

    collator = DataCollatorForSeq2Seq(tok, model=model, padding=True, label_pad_token_id=-100)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUT_DIR),
        # Lowered 6->2 (grad_accum 6->16, same ~32 effective batch) after an
        # OOM: batch=6 was tuned while requires_grad was silently all-False
        # (see the merge_and_unload note above), which made optimizer.step()
        # a no-op and meant NO Adam state was ever allocated. Now that
        # gradients genuinely flow, real fp32 AdamW state for 615M params
        # needs real memory, and this GPU is shared with bursty vLLM tenants
        # (free memory swung from 10.5GB to 522MB within seconds during
        # observation) -- smaller batch leaves headroom for those spikes.
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,           # much lower than LoRA's 1e-4 -- full-FT, higher forgetting risk
        num_train_epochs=args.epochs,    # short continuation, not from-scratch training
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,               # extra safety against destructive updates
        logging_dir=str(LOG_DIR),
        logging_steps=50,
        eval_strategy="epoch",  # must match save_strategy for load_best_model_at_end; fine for a 2-epoch run
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_bf16_supported() if device == "cuda" else False,
        predict_with_generate=False,
        report_to=[],
        dataloader_num_workers=16,  # raised 2->16: machine has 192 cores / 387GB RAM free, GPU util
                                    # stayed flat as batch size increased -> CPU-side collation was
                                    # starving the GPU, not a memory/compute limit
        remove_unused_columns=False,
    )

    callbacks = []
    if args.early_stopping_patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        print(f"Early stopping enabled: patience={args.early_stopping_patience} evals on eval_loss")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tok,
        callbacks=callbacks,
    )

    train_result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    model.save_pretrained(str(OUT_DIR / "model"), safe_serialization=True)
    tok.save_pretrained(str(OUT_DIR / "model"))
    print(f"Saved full-FT model to {OUT_DIR / 'model'}")

    import datetime
    run_config = {
        "base_model": MODEL_NAME,
        "method": "full fine-tune, initialized from merged LoRA adapter" if args.init_adapter
                   else "full fine-tune, continued from existing full-FT checkpoint",
        "init_adapter": args.init_adapter,
        "continue_from_model": args.continue_from_model,
        "trainable_params": n_params,
        "training_args": {
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
            "learning_rate": args.lr,
            "num_train_epochs": args.epochs,
        },
        "n_train_examples": len(fit_records),
        "final_train_loss": train_result.training_loss if train_result else None,
        "trained_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "model_path": str(OUT_DIR / "model"),
    }
    with open(OUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2, default=str)
    print(f"Wrote {OUT_DIR / 'run_config.json'}")


if __name__ == "__main__":
    main()
