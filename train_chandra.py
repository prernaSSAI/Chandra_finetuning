#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from chandra_finetune import DEFAULT_MODEL_NAME
from chandra_finetune.data import (
    ChandraSample,
    load_chandra_dataset,
    load_lazy_training_dataset,
    normalize_sample,
    samples_to_training_records,
)
from chandra_finetune.generation import GenerationSettings, generate_text
from chandra_finetune.metrics import table_teds_score
from chandra_finetune.modeling import LoraSettings, load_training_model, set_training_mode


DEFAULT_BEST_METRIC = "table_teds"
DEFAULT_GREATER_IS_BETTER = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Chandra with Unsloth LoRA.")
    parser.add_argument("--dataset", required=True, help="Training dataset artifact: Arrow dir, .pkl, .json, or .jsonl.")
    parser.add_argument("--model-name", "--model-path", dest="model_name", default=DEFAULT_MODEL_NAME, help="Base model checkpoint.")
    parser.add_argument("--output-dir", default="outputs/chandra_lora", help="Directory for LoRA adapter output.")
    parser.add_argument("--eval-dataset", help="Optional held-out dataset path for Table-TEDS validation.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for LoRA and trainer config.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional cap for generated Table-TEDS evaluation.")

    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", default="unsloth", help='Use "unsloth", "true", "false", or "none".')

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-bias", "--bias", dest="lora_bias", default="none")
    parser.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--finetune-vision-layers", "--vision-layers", dest="finetune_vision_layers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--finetune-language-layers", "--language-layers", dest="finetune_language_layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-attention-modules", "--attention-modules", dest="finetune_attention_modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-mlp-modules", "--mlp-modules", dest="finetune_mlp_modules", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--per-device-train-batch-size", "--batch-size-per-device", dest="per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum optimizer steps. Keep -1 to train for the requested number of epochs.",
    )
    parser.add_argument("--num-train-epochs", "--epochs", dest="num_train_epochs", type=float, default=50.0)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a Trainer checkpoint directory to resume model, optimizer, scheduler, and trainer state.",
    )

    parser.add_argument(
    "--resume-best-table-teds",
    type=float,
    default=None,
    help="Best validation Table-TEDS score from the previous interrupted run.",
)
    parser.add_argument(
        "--resume-best-epoch",
        type=float,
        default=None,
        help="Epoch where the previous best Table-TEDS was achieved.",
    )
    parser.add_argument(
        "--resume-best-step",
        type=int,
        default=None,
        help="Global step where the previous best Table-TEDS was achieved.",
    )

    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--eval-strategy", default="epoch", help='Evaluation strategy, e.g. "no", "steps", or "epoch".')
    parser.add_argument(
        "--save-strategy",
        default="epoch",
        help='Trainer checkpoint save strategy, e.g. "no", "steps", or "epoch". Best/last adapters are always saved separately.',
    )
    parser.add_argument(
        "--load-best-model-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Hugging Face Trainer best-checkpoint loading when possible. The custom Table-TEDS callback still saves the best LoRA adapter.",
    )
    parser.add_argument(
        "--metric-for-best-model",
        default=DEFAULT_BEST_METRIC,
        help='Metric used by Trainer best-checkpoint logic. Defaults to "table_teds".',
    )
    parser.add_argument(
        "--greater-is-better",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GREATER_IS_BETTER,
        help="Table-TEDS is a score, so higher is better.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        "--patience",
        dest="early_stopping_patience",
        type=int,
        default=15,
        help="Stop after this many validation epochs without Table-TEDS improvement.",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help="Minimum Table-TEDS increase required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--early-stopping-min-steps",
        type=int,
        default=0,
        help="Do not stop early before this many optimizer steps.",
    )
    parser.add_argument("--eval-generation-max-new-tokens", type=int, default=112384)
    parser.add_argument("--eval-generation-temperature", type=float, default=0.0)
    parser.add_argument("--eval-generation-top-p", type=float, default=1.0)
    parser.add_argument("--eval-generation-top-k", type=int, default=0)
    parser.add_argument("--eval-generation-repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--table-teds-exclude-first-table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass exclude_first_table to table_teds_score().",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # ── Load training data (lazy for Arrow, eager for pkl/json) ──────────
    train_dataset = load_lazy_training_dataset(
        args.dataset, max_samples=args.max_samples,
    )
    if train_dataset is not None:
        num_train = len(train_dataset)
        print(f"[lazy] Loaded Arrow training dataset with {num_train} samples (images decoded on-the-fly).")
    else:
        samples = load_chandra_dataset(args.dataset)
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
        if not samples:
            raise ValueError("The training dataset is empty.")
        num_train = len(samples)
        train_dataset = samples_to_training_records(samples)
        print(f"[eager] Loaded {num_train} training samples into RAM.")

    # ── Load eval data for both Trainer eval-loss and generated Table-TEDS ─
    eval_dataset = None
    eval_samples_for_teds: list[ChandraSample] = []
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
        eval_samples_for_teds = _load_eval_samples_for_table_teds(args.eval_dataset, max_samples=args.max_eval_samples)

    train_only = eval_dataset is None
    if train_only:
        if args.eval_strategy not in (None, "no"):
            raise ValueError(
                "No eval dataset was loaded, so --eval-strategy must be omitted or set to 'no'. "
                "Pass --eval-dataset to enable generated Table-TEDS model selection."
            )
        args.eval_strategy = "no"
        args.load_best_model_at_end = False
        print("No eval dataset was loaded; Table-TEDS best-model selection and early stopping are disabled.")
    else:
        _validate_table_teds_best_model_args(args)

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
        from transformers import TrainerCallback
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
        "eval_strategy": args.eval_strategy,
        "save_strategy": args.save_strategy,
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": args.greater_is_better,
    }
    if args.num_train_epochs is not None:
        trainer_kwargs["num_train_epochs"] = args.num_train_epochs

    best_table_teds_dir = Path(args.output_dir) / ".best_table_teds_tmp"

    if best_table_teds_dir.exists() and args.resume_from_checkpoint is None:
        shutil.rmtree(best_table_teds_dir)
    elif best_table_teds_dir.exists() and args.resume_from_checkpoint is not None:
        print(f"Resuming: preserving existing best Table-TEDS adapter at {best_table_teds_dir}")
    table_teds_monitor = _build_table_teds_monitor_callback(
    TrainerCallback,
    tokenizer=tokenizer,
    eval_samples=eval_samples_for_teds,
    best_model_dir=best_table_teds_dir,
    patience=args.early_stopping_patience,
    threshold=args.early_stopping_threshold,
    min_steps=args.early_stopping_min_steps,
    generation_settings=GenerationSettings(
        max_new_tokens=args.eval_generation_max_new_tokens,
        temperature=args.eval_generation_temperature,
        top_p=args.eval_generation_top_p,
        top_k=args.eval_generation_top_k,
        repetition_penalty=args.eval_generation_repetition_penalty,
    ),
    exclude_first_table=args.table_teds_exclude_first_table,
    initial_best_score=args.resume_best_table_teds,
    initial_best_epoch=args.resume_best_epoch,
    initial_best_step=args.resume_best_step,
)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**trainer_kwargs),
        callbacks=[table_teds_monitor],
    )

    print(f"Training with {num_train} train samples and {num_eval} eval samples.")
    print(f"Best Table-TEDS LoRA adapter will be tracked and saved to {Path(args.output_dir) / 'best'}.")
    if args.early_stopping_patience is not None and not train_only:
        print(
            "Table-TEDS early stopping enabled: "
            f"patience={args.early_stopping_patience} epochs, "
            f"threshold={args.early_stopping_threshold}, "
            f"min_steps={args.early_stopping_min_steps}."
        )
    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always save the model from the final training step.
    last_dir = output_dir / "last"
    _recreate_dir(last_dir)
    model.save_pretrained(last_dir)
    tokenizer.save_pretrained(last_dir)
    print(f"Saved last model to {last_dir}")

    best_dir = output_dir / "best"
    if table_teds_monitor.best_model_dir.exists():
        _recreate_dir(best_dir)
        _copy_saved_model_files(table_teds_monitor.best_model_dir, best_dir)
        shutil.rmtree(table_teds_monitor.best_model_dir)
        print(
            f"Saved best Table-TEDS model "
            f"(epoch {table_teds_monitor.best_epoch}, "
            f"step {table_teds_monitor.best_step}, "
            f"table_teds {table_teds_monitor.best_score:.6f}) to {best_dir}"
        )
    else:
        _recreate_dir(best_dir)
        _copy_saved_model_files(last_dir, best_dir)
        print("WARNING: No valid validation table_teds was computed; copied last model to best as a fallback.")

    extra_metadata = {
        "table_teds_monitor": {
            **table_teds_monitor.summary(),
            "best_model_dir": str(best_dir),
        }
    }
    _write_run_metadata(output_dir, args, trainer_stats.metrics, extra_metadata=extra_metadata)
    print(f"Saved LoRA adapter and tokenizer under {output_dir / 'best'} and {output_dir / 'last'}")


def _validate_table_teds_best_model_args(args: argparse.Namespace) -> None:
    if args.early_stopping_patience is not None and args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience/--patience must be >= 1.")
    if args.early_stopping_threshold < 0:
        raise ValueError("--early-stopping-threshold must be >= 0.")
    if args.early_stopping_min_steps < 0:
        raise ValueError("--early-stopping-min-steps must be >= 0.")
    if args.load_best_model_at_end:
        if args.eval_strategy == "no":
            raise ValueError("--load-best-model-at-end requires evaluation; set --eval-strategy epoch/steps or disable it.")
        if args.save_strategy == "no":
            raise ValueError("--load-best-model-at-end requires checkpoint saving; set --save-strategy epoch/steps or disable it.")
        if args.save_strategy != args.eval_strategy:
            raise ValueError(
                "--load-best-model-at-end requires matching --eval-strategy and --save-strategy. "
                "Use the default epoch/epoch settings or disable load_best_model_at_end."
            )
    if args.metric_for_best_model not in {"table_teds", "eval_table_teds"}:
        print(
            "WARNING: metric_for_best_model is not table_teds/eval_table_teds. "
            "The custom callback still saves the best adapter by validation table_teds."
        )
    if not args.greater_is_better:
        print("WARNING: greater_is_better=False, but Table-TEDS is a score. Consider using --greater-is-better.")


def _load_eval_samples_for_table_teds(path: str | Path, *, max_samples: int | None) -> list[ChandraSample]:
    """Load only the eval subset used for generated Table-TEDS scoring.

    For Arrow datasets this applies max_samples before normalizing, so validation
    generation can be capped without eagerly decoding a large full eval set.
    """

    dataset_path = Path(path)
    if dataset_path.is_dir():
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise RuntimeError("Loading Arrow eval datasets requires the 'datasets' package.") from exc
        hf_dataset = load_from_disk(str(dataset_path))
        if max_samples is not None:
            hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))
        return [normalize_sample(dict(record)) for record in hf_dataset]

    samples = load_chandra_dataset(dataset_path)
    if max_samples is not None:
        samples = samples[: max_samples]
    return samples


def _parse_gradient_checkpointing(value: str) -> str | bool | None:
    normalized = value.lower()
    if normalized == "none":
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return value


def _build_table_teds_monitor_callback(
    trainer_callback_cls: type,
    *,
    tokenizer,
    eval_samples: Iterable[ChandraSample],
    best_model_dir: Path,
    patience: int | None,
    threshold: float,
    min_steps: int,
    generation_settings: GenerationSettings,
    exclude_first_table: bool,
    initial_best_score: float | None = None,
    initial_best_epoch: float | None = None,
    initial_best_step: int | None = None,
):
    eval_samples = list(eval_samples)

    class TableTEDSMonitorCallback(trainer_callback_cls):
        """Generate validation predictions, track avg table_teds, and save the best LoRA adapter."""

        def __init__(self) -> None:
            self.best_score: float | None = initial_best_score
            self.best_epoch: float | None = initial_best_epoch
            self.best_step: int | None = initial_best_step
            self.last_score: float | None = None
            self.last_epoch: float | None = None
            self.last_step: int | None = None
            self.last_num_scored = 0
            self.last_num_skipped = 0
            self.bad_epochs = 0
            self.stopped_early = False
            self.stopped_step: int | None = None
            self.best_model_dir = best_model_dir

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            model = kwargs.get("model")
            score, num_scored, num_skipped = self._evaluate_table_teds(model)
            epoch = float(state.epoch) if state.epoch is not None else None
            self.last_score = score
            self.last_epoch = epoch
            self.last_step = state.global_step
            self.last_num_scored = num_scored
            self.last_num_skipped = num_skipped

            # Expose the generated metric to HF Trainer.  If no score can be
            # computed, use the previous best (or 0.0) so load_best_model_at_end
            # does not crash because eval_table_teds is missing.
            metric_value_for_trainer = score
            if metric_value_for_trainer is None:
                metric_value_for_trainer = self.best_score if self.best_score is not None else 0.0
            if metrics is not None:
                metrics["eval_table_teds"] = float(metric_value_for_trainer)

            improved = score is not None and (
                self.best_score is None or score > (self.best_score + threshold)
            )
            saved = False
            if improved:
                self.best_score = score
                self.best_epoch = epoch
                self.best_step = state.global_step
                self.bad_epochs = 0
                self._save_best_model(model, state)
                saved = True
            elif score is not None and patience is not None and state.global_step >= min_steps:
                self.bad_epochs += 1
                if self.bad_epochs >= patience:
                    self.stopped_early = True
                    self.stopped_step = state.global_step
                    control.should_training_stop = True

            if state.is_local_process_zero:
                score_text = "None" if score is None else f"{score:.6f}"
                best_text = "None" if self.best_score is None else f"{self.best_score:.6f}"
                print(
                    "[table_teds] "
                    f"epoch={epoch} "
                    f"global_step={state.global_step} "
                    f"validation_table_teds={score_text} "
                    f"best_validation_table_teds={best_text} "
                    f"scored={num_scored} skipped={num_skipped} "
                    f"saved_new_best={saved}"
                )
                if self.stopped_early:
                    print(
                        "Early stopping triggered on validation table_teds: "
                        f"best={best_text} at epoch {self.best_epoch}, "
                        f"current={score_text} at epoch {epoch}."
                    )
            return control

        def summary(self) -> dict:
            return {
                "enabled": bool(eval_samples),
                "check_interval": "evaluation",
                "metric": "table_teds",
                "greater_is_better": True,
                "patience": patience,
                "threshold": threshold,
                "min_steps": min_steps,
                "best_score": self.best_score,
                "best_epoch": self.best_epoch,
                "best_step": self.best_step,
                "last_score": self.last_score,
                "last_epoch": self.last_epoch,
                "last_step": self.last_step,
                "last_num_scored": self.last_num_scored,
                "last_num_skipped": self.last_num_skipped,
                "best_model_dir": str(self.best_model_dir),
                "bad_epochs": self.bad_epochs,
                "stopped_early": self.stopped_early,
                "stopped_step": self.stopped_step,
            }

        def _evaluate_table_teds(self, model) -> tuple[float | None, int, int]:
            if model is None or not eval_samples:
                return None, 0, len(eval_samples)

            was_training = bool(getattr(model, "training", False))
            try:
                model.eval()
            except Exception:
                pass

            scores: list[float] = []
            skipped = 0
            for sample in eval_samples:
                reference = sample.reference
                if reference is None or not str(reference).strip():
                    skipped += 1
                    continue
                try:
                    prediction = generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        image=sample.image,
                        prompt=sample.prompt,
                        settings=generation_settings,
                    )
                    score = table_teds_score(
                        prediction,
                        reference,
                        exclude_first_table=exclude_first_table,
                    )
                except Exception as exc:
                    skipped += 1
                    print(f"WARNING: Skipping eval sample after Table-TEDS error: {exc}")
                    continue
                if score is None or not math.isfinite(float(score)):
                    skipped += 1
                    continue
                scores.append(float(score))

            if was_training:
                try:
                    model.train()
                except Exception:
                    pass

            return (mean(scores) if scores else None), len(scores), skipped

        def _save_best_model(self, model, state) -> None:
            if model is None or not state.is_world_process_zero:
                return
            _recreate_dir(self.best_model_dir)
            model.save_pretrained(self.best_model_dir)
            tokenizer.save_pretrained(self.best_model_dir)

    return TableTEDSMonitorCallback()


def _recreate_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_saved_model_files(source_dir: Path, target_dir: Path) -> None:
    for source in source_dir.iterdir():
        target = target_dir / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)


def _write_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    metrics: dict,
    *,
    extra_metadata: dict | None = None,
) -> None:
    payload = {
        "args": vars(args),
        "metrics": metrics,
    }
    if extra_metadata:
        payload.update(extra_metadata)
    with (output_dir / "training_run.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None