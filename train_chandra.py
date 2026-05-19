#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chandra_finetune import DEFAULT_MODEL_NAME
from chandra_finetune.data import (
    load_chandra_dataset,
    load_lazy_training_dataset,
    samples_to_training_records,
)
from chandra_finetune.modeling import LoraSettings, load_training_model, set_training_mode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Chandra with Unsloth LoRA.")
    parser.add_argument("--dataset", required=True, help="Training dataset artifact: Arrow dir, .pkl, .json, or .jsonl.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base model checkpoint.")
    parser.add_argument("--output-dir", default="outputs/chandra_lora", help="Directory for LoRA adapter output.")
    parser.add_argument("--eval-dataset", help="Optional held-out dataset path for trainer evaluation.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for LoRA and trainer config.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional eval cap for debugging.")

    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", default="unsloth", help='Use "unsloth", "true", "false", or "none".')

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-bias", default="none")
    parser.add_argument("--use-rslora", action="store_true")
    parser.add_argument("--finetune-vision-layers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--finetune-language-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-attention-modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-mlp-modules", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=30, help="Use -1 with --num-train-epochs for full epochs.")
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a Trainer checkpoint directory to resume model, optimizer, scheduler, and trainer state.",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--eval-strategy", default=None, help='Evaluation strategy, e.g. "no", "steps", or "epoch".')
    parser.add_argument("--save-strategy", default=None, help='Checkpoint save strategy, e.g. "steps" or "epoch".')
    parser.add_argument("--load-best-model-at-end", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--metric-for-best-model", default="eval_loss")
    parser.add_argument("--greater-is-better", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # ── Load training data (lazy for Arrow, eager for pkl/json) ──────────
    train_dataset = load_lazy_training_dataset(
        args.dataset, max_samples=args.max_samples,
    )
    if train_dataset is not None:
        # Arrow path: lazy loading – images stay on disk, decoded per-batch
        num_train = len(train_dataset)
        print(f"[lazy] Loaded Arrow training dataset with {num_train} samples (images decoded on-the-fly).")
    else:
        # Fallback: pkl / json / jsonl – eager load (small datasets)
        samples = load_chandra_dataset(args.dataset)
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
        if not samples:
            raise ValueError("The training dataset is empty.")
        num_train = len(samples)
        train_dataset = samples_to_training_records(samples)
        print(f"[eager] Loaded {num_train} training samples into RAM.")

    # ── Load eval data ───────────────────────────────────────────────────
    eval_dataset = None
    num_eval = 0
    if args.eval_dataset:
        eval_lazy = load_lazy_training_dataset(
            args.eval_dataset, max_samples=args.max_eval_samples,
        )
        if eval_lazy is not None:
            eval_dataset = eval_lazy
            num_eval = len(eval_dataset)
            print(f"[lazy] Loaded Arrow eval dataset with {num_eval} samples.")
        else:
            eval_samples = load_chandra_dataset(args.eval_dataset)
            if args.max_eval_samples is not None:
                eval_samples = eval_samples[: args.max_eval_samples]
            num_eval = len(eval_samples)
            eval_dataset = samples_to_training_records(eval_samples) if eval_samples else None
            print(f"[eager] Loaded {num_eval} eval samples into RAM.")

    lora = LoraSettings(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        random_state=args.seed,
        use_rslora=args.use_rslora,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
    )

    model, tokenizer = load_training_model(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing=_parse_gradient_checkpointing(args.gradient_checkpointing),
        lora=lora,
    )
    set_training_mode(model)

    try:
        from trl import SFTConfig, SFTTrainer
        from transformers import EarlyStoppingCallback
        from unsloth.trainer import UnslothVisionDataCollator
    except ImportError as exc:
        raise RuntimeError(
            "Training requires trl and unsloth. Install dependencies from README_CHANDRA_FINETUNE.md."
        ) from exc

    trainer_kwargs = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "optim": args.optim,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "report_to": args.report_to,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": args.max_length,
    }
    if args.num_train_epochs is not None:
        trainer_kwargs["num_train_epochs"] = args.num_train_epochs
    if args.eval_strategy is not None:
        trainer_kwargs["eval_strategy"] = args.eval_strategy
    if args.save_strategy is not None:
        trainer_kwargs["save_strategy"] = args.save_strategy
    if args.load_best_model_at_end:
        trainer_kwargs["load_best_model_at_end"] = args.load_best_model_at_end
        trainer_kwargs["metric_for_best_model"] = args.metric_for_best_model
        trainer_kwargs["greater_is_better"] = args.greater_is_better

    callbacks = []
    if args.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**trainer_kwargs),
        callbacks=callbacks or None,
    )

    print(f"Training with {num_train} train samples and {num_eval} eval samples.")
    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _write_run_metadata(output_dir, args, trainer_stats.metrics)
    print(f"Saved LoRA adapter and tokenizer to {output_dir}")


def _parse_gradient_checkpointing(value: str) -> str | bool | None:
    normalized = value.lower()
    if normalized == "none":
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return value


def _write_run_metadata(output_dir: Path, args: argparse.Namespace, metrics: dict) -> None:
    payload = {
        "args": vars(args),
        "metrics": metrics,
    }
    with (output_dir / "training_run.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
