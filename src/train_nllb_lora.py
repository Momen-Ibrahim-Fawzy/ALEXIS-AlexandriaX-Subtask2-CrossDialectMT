"""
Stage 5: LoRA fine-tune facebook/nllb-200-distilled-600M on the combined
gold+silver+bronze many-to-many corpus (data/finetune_train.jsonl).

Why LoRA over full fine-tuning (documented decision, see System/README.md):
  - Only ~19k gold + tens of thousands of silver/bronze examples are available
    -- a genuinely low-resource regime for a 600M-param multilingual model that
    already covers 200 languages including all 7 varieties this task needs
    natively. Full fine-tuning on this little data risks catastrophic
    forgetting of the pretrained dialectal competence for Egyptian/Lebanese
    (which get NO real training signal at all -- see EDA_INSIGHTS.md §5/§6),
    which would be actively harmful. LoRA leaves the base weights frozen and
    only learns small low-rank deltas, which the literature on low-resource /
    dialect NLLB adaptation consistently reports as matching-or-beating full
    fine-tuning while being far cheaper and more stable in this regime.

Run: CUDA_VISIBLE_DEVICES=1 conda run -n mo python train_nllb_lora.py
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
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nllb_utils import build_ids

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
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M",
                     help="NLLB backbone, e.g. facebook/nllb-200-distilled-1.3B for the larger-capacity run")
    ap.add_argument("--out-name", default="nllb600m_lora_dialectmt_v2",
                     help="subdir under outputs/checkpoints/ -- keep distinct per run so nothing gets overwritten")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--resume", action="store_true",
                     help="resume from the latest checkpoint already in --out-name's output dir, if any "
                          "(e.g. v3 was interrupted mid-epoch-4 by a machine migration -- this picks up "
                          "from checkpoint-5484/epoch-3 instead of restarting from scratch)")
    ap.add_argument("--init-adapter", default=None,
                     help="continue training an EXISTING adapter's weights for --epochs MORE epochs, "
                          "instead of building a fresh LoRA init. Use when a prior run's eval_loss was "
                          "still trending down at its last epoch (e.g. v4b: 0.916->0.911 epoch7->8, "
                          "never plateaued) -- a sign it was cut off before convergence, not overfitting.")
    cli_args = ap.parse_args()
    MODEL_NAME = cli_args.model
    OUT_DIR = CKPT_ROOT / cli_args.out_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, model: {MODEL_NAME}, out: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    # use_safetensors=True forced explicitly: this cache has both a .bin and a
    # .safetensors snapshot (two different upstream commits), and newer
    # transformers refuses to torch.load() the .bin one on torch<2.6 (CVE-2025-32434).
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=True)

    if cli_args.init_adapter:
        model = PeftModel.from_pretrained(model, cli_args.init_adapter, is_trainable=True)
        print(f"Continuing training from existing adapter: {cli_args.init_adapter}")
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            # v2: rank doubled (16->32, alpha kept at 2x r) now that the corpus is
            # ~3x larger (v1: 22.7k, v2: ~90k+) -- more capacity to match more data.
            r=cli_args.lora_r, lora_alpha=cli_args.lora_r * 2, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.config.use_cache = False  # required alongside gradient checkpointing
    model.enable_input_require_grads()
    model.to(device)

    train_records = read_jsonl(DATA / "finetune_train.jsonl")
    print(f"Loaded {len(train_records)} training examples")

    # small internal eval slice (loss-only sanity check; real quality eval is
    # done separately in evaluate.py against gold_dev + finetune_dev_soft)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_records))
    n_eval = min(400, max(50, len(train_records) // 100))
    eval_idx = set(idx[:n_eval].tolist())
    eval_records = [r for i, r in enumerate(train_records) if i in eval_idx]
    fit_records = [r for i, r in enumerate(train_records) if i not in eval_idx]

    train_ds = build_dataset(tok, fit_records)
    eval_ds = build_dataset(tok, eval_records)

    collator = DataCollatorForSeq2Seq(tok, model=model, padding=True, label_pad_token_id=-100)

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUT_DIR),
        # This machine's GPUs are shared with other tenants (confirmed via
        # nvidia-smi: VLLM engines / Triton servers routinely eat 15-20GiB each
        # with no notice) -- an earlier run at batch=32 OOM'd mid-training when
        # contention spiked. Smaller per-device batch + gradient accumulation
        # + gradient checkpointing keeps our peak footprint low enough to
        # survive that fluctuation while keeping the same effective batch size.
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        # v4 NOTE: added use_reentrant=False after the l402 machine's torch
        # 2.6.0 upgrade (done for an unrelated PeftModel.from_pretrained/
        # DTensor eval-time bug) broke training with "element 0 of tensors
        # does not require grad and does not have a grad_fn" -- reentrant
        # checkpointing (the old default) has known gradient-flow issues
        # through a frozen base model + PEFT adapter on newer torch;
        # non-reentrant is the standard fix and doesn't need a torch pin.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1e-4,
        # v2: 3->5 epochs -- v1's eval_loss was still trending down at epoch 3
        # end (1.42->1.19, not plateaued), i.e. it was undertrained, not at
        # risk of overfitting yet. load_best_model_at_end below protects
        # against going too far this time.
        num_train_epochs=cli_args.epochs,
        warmup_ratio=0.05,
        weight_decay=0.01,
        # v2 NOTE: label_smoothing_factor was tried here (standard NMT
        # mitigation for exactly the repetition-degeneration failure mode
        # found in v1's predictions) but REMOVED -- it crashes immediately
        # with "You cannot specify both decoder_input_ids and
        # decoder_inputs_embeds at the same time" in this transformers/peft
        # version combo. Root cause: label_smoothing_factor>0 makes HF
        # Trainer.compute_loss pop "labels" and call model(**inputs) without
        # them (loss computed separately via LabelSmoother), which conflicts
        # with DataCollatorForSeq2Seq's model-aware decoder_input_ids
        # pre-population in this PEFT-wrapped-M2M100 forward path. The
        # generation-time repetition guard (no_repeat_ngram_size /
        # repetition_penalty in nllb_utils.py) plus the bronze-tier
        # degenerate-output filter (build_finetune_corpus.py) cover the same
        # failure mode without this crash; not worth debugging further under
        # the time budget. Revisit if upgrading transformers/peft.
        logging_dir=str(LOG_DIR),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        # v2: pick the checkpoint with the best real held-out loss rather than
        # whatever epoch happens to finish last -- guards against the extra
        # 2 epochs overfitting on the smaller tiers (e.g. Tunisian silver).
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_bf16_supported() if device == "cuda" else False,
        predict_with_generate=False,
        report_to=[],
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tok,
    )

    resume_ckpt = True if cli_args.resume else None
    if cli_args.resume:
        print(f"Resuming from latest checkpoint in {OUT_DIR} (if any)")
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    model.save_pretrained(str(OUT_DIR / "adapter"))
    tok.save_pretrained(str(OUT_DIR / "adapter"))
    print(f"Saved LoRA adapter to {OUT_DIR / 'adapter'}")

    # persisted run config -- feeds System/src/make_submission.py's system-card
    # generator so every submission is traceable back to exactly how its
    # underlying model was trained (for the paper's reproducibility section).
    import datetime
    corpus_stats_path = DATA / "corpus_stats.json"
    corpus_stats = json.loads(corpus_stats_path.read_text(encoding="utf-8")) if corpus_stats_path.exists() else None
    run_config = {
        "base_model": MODEL_NAME,
        "method": "LoRA (peft)",
        "lora_config": {"r": lora_cfg.r, "lora_alpha": lora_cfg.lora_alpha,
                         "lora_dropout": lora_cfg.lora_dropout,
                         "target_modules": list(lora_cfg.target_modules)},
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "training_args": {
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "gradient_checkpointing": args.gradient_checkpointing,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "label_smoothing_factor": args.label_smoothing_factor,
            "load_best_model_at_end": args.load_best_model_at_end,
            "bf16": args.bf16,
            "max_seq_len": MAX_LEN,
        },
        "n_train_examples": len(fit_records),
        "n_internal_eval_examples": len(eval_records),
        "final_train_loss": train_result.training_loss if train_result else None,
        "corpus_stats": corpus_stats,
        "trained_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "adapter_path": str(OUT_DIR / "adapter"),
    }
    with open(OUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2, default=str)
    print(f"Wrote {OUT_DIR / 'run_config.json'}")


if __name__ == "__main__":
    main()
