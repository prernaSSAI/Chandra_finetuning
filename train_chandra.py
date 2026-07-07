#!/usr/bin/env python
from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass
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
from chandra_finetune.modeling import LoraSettings, load_training_model, set_inference_mode, set_training_mode


DEFAULT_BEST_METRIC = "table_teds"
DEFAULT_GREATER_IS_BETTER = True


@dataclass
class TrainConfig:
   
    dataset: str = "/mnt/disk/ml_data/prerna/split_140dpi/train_140_dpi"
    # Optional held-out dataset for Table-TEDS validation. None = train-loss only.
    eval_dataset: str | None = None
    max_samples: int | None = None        # cap training samples (debugging)
    max_eval_samples: int | None = None   # cap eval samples for Table-TEDS

    # ── Model / output ───────────────────────────────────────────────────
    model_name: str = DEFAULT_MODEL_NAME  # base model checkpoint
    output_dir: str = "/mnt/disk/ml_data/prerna/iter_5"
    seed: int = 3407

    # ── LoRA ─────────────────────────────────────────────────────────────
    load_in_4bit: bool = False
    gradient_checkpointing: str = "unsloth"  # "unsloth" | "true" | "false" | "none"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    use_rslora: bool = False
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = False

    # ── Optimization ─────────────────────────────────────────────────────
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 50
    max_steps: int = -1                   # -1 = train for num_train_epochs
    num_train_epochs: float = 15.0
    learning_rate: float = 3e-5
    optim: str = "adamw_8bit"
    weight_decay: float = 0.05
    lr_scheduler_type: str = "cosine"
    # Image patches count toward max_length. At full 140dpi (~1190x1540) the
    # image alone is ~1.8k tokens, so 4096 would truncate long HTML references.
    max_length: int = 10240 #10240 
    logging_steps: int = 1
    logging_strategy: str = "epoch"
    disable_tqdm: bool = False
    report_to: str = "none"

    # ── Checkpointing / best-model selection ─────────────────────────────
    eval_strategy: str = "epoch"          # "no" | "steps" | "epoch"
    save_strategy: str = "epoch"          # "no" | "steps" | "epoch"
    load_best_model_at_end: bool = True   # auto-disabled when eval_dataset is None
    metric_for_best_model: str = DEFAULT_BEST_METRIC
    greater_is_better: bool = DEFAULT_GREATER_IS_BETTER

    # ── Resume (set when continuing an interrupted run) ──────────────────
    resume_from_checkpoint: str | None = None
    resume_best_table_teds: float | None = None
    resume_best_epoch: float | None = None
    resume_best_step: int | None = None

    # ── Early stopping (only active when Table-TEDS eval is enabled) ─────
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0
    early_stopping_min_steps: int = 0

    # ── Table-TEDS validation genegenemax_lenrationmax_lenration (only used when eval is enabled) ─
    eval_generation_max_new_tokens: int = 12384
    eval_generation_temperature: float = 0.0
    eval_generation_top_p: float = 1.0
    eval_generation_top_k: int = 0
    eval_generation_repetition_penalty: float = 1.0
    table_teds_exclude_first_table: bool = True


def main() -> None:
    args = TrainConfig()

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
        # No eval dataset → train on train-loss only. No per-epoch generation or
        # Table-TEDS. Best model becomes a copy of the final (lowest-loss) one.
        args.eval_strategy = "no"
        args.load_best_model_at_end = False
        print("No eval dataset was loaded; Table-TEDS best-model selection and early stopping are disabled.")
    else:
        # Eval dataset present → turn Table-TEDS validation back on automatically.
        if args.eval_strategy == "no":
            args.eval_strategy = "epoch"
        if args.save_strategy == "no":
            args.save_strategy = "epoch"
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
        "logging_strategy": args.logging_strategy,
        "disable_tqdm": args.disable_tqdm,
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

    # Train-loss best-model tracker. Active only in train-only mode (no eval
    # dataset): saves the LoRA adapter at the epoch with the lowest mean training
    # loss so the final <output_dir>/best is the least-train-loss epoch rather
    # than the last step. No effect when Table-TEDS validation is enabled.
    best_train_loss_dir = Path(args.output_dir) / ".best_train_loss_tmp"
    if best_train_loss_dir.exists() and args.resume_from_checkpoint is None:
        shutil.rmtree(best_train_loss_dir)
    train_loss_monitor = _build_train_loss_monitor_callback(
        TrainerCallback,
        tokenizer=tokenizer,
        best_model_dir=best_train_loss_dir,
        enabled=train_only,
    )

    callbacks = [table_teds_monitor]
    if train_only:
        callbacks.append(train_loss_monitor)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        # resize="max" = no downscaling. The default ("min") falls back to a
        # 512px width cap because qwen3_vl has no vision_config.image_size,
        # which silently threw away the dataset's DPI.
        data_collator=UnslothVisionDataCollator(model, tokenizer, resize="max"),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**trainer_kwargs),
        callbacks=callbacks,
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
    elif train_only and train_loss_monitor.best_model_dir.exists():
        _recreate_dir(best_dir)
        _copy_saved_model_files(train_loss_monitor.best_model_dir, best_dir)
        shutil.rmtree(train_loss_monitor.best_model_dir)
        print(
            f"Saved best train-loss model "
            f"(epoch {train_loss_monitor.best_epoch}, "
            f"step {train_loss_monitor.best_step}, "
            f"train_loss {train_loss_monitor.best_loss:.6f}) to {best_dir}"
        )
        # "load best model at end" for train-only mode: reload the lowest
        # train-loss adapter into the live model so the returned model matches
        # best/ (last/ above keeps the final-step weights). HF's native
        # load_best_model_at_end can't do this without an eval metric.
    else:
        _recreate_dir(best_dir)
        _copy_saved_model_files(last_dir, best_dir)
        print("WARNING: No best checkpoint was tracked; copied last model to best as a fallback.")

    extra_metadata = {
        "table_teds_monitor": {
            **table_teds_monitor.summary(),
            "best_model_dir": str(best_dir),
        },
        "train_loss_monitor": train_loss_monitor.summary(),
    }
    _write_run_metadata(output_dir, args, trainer_stats.metrics, extra_metadata=extra_metadata)
    print(f"Saved LoRA adapter and tokenizer under {output_dir / 'best'} and {output_dir / 'last'}")


def _validate_table_teds_best_model_args(args: TrainConfig) -> None:
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
            eval_start = time.perf_counter()
            score, num_scored, num_skipped = self._evaluate_table_teds(model)
            eval_seconds = time.perf_counter() - eval_start
            seconds_per_page = (eval_seconds / num_scored) if num_scored else float("nan")
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
                    f"eval_seconds={eval_seconds:.1f} "
                    f"seconds_per_page={seconds_per_page:.2f} "
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

            # Switch to Unsloth inference mode so gradient checkpointing is
            # disabled and the KV cache (use_cache) is re-enabled during
            # generation.  Without this, grad checkpointing stays active and HF
            # forces use_cache=False, making every decode step recompute the
            # full sequence (O(n^2)).  Always restore training mode afterward.
            try:
                set_inference_mode(model)
            except Exception:
                pass

            scores: list[float] = []
            skipped = 0
            try:
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
            finally:
                # Restore training mode (re-enables gradient checkpointing) so
                # the next training epoch is unaffected.
                try:
                    set_training_mode(model)
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


def _build_train_loss_monitor_callback(
    trainer_callback_cls: type,
    *,
    tokenizer,
    best_model_dir: Path,
    enabled: bool,
):
    """Track the lowest *per-epoch* mean training loss and save the adapter at it.

    Used in train-only mode (no eval dataset) so the final best checkpoint is
    the epoch with the least average train loss rather than the last training
    step (or a single noisy low-loss step). on_log accumulates the smoothed
    training ``loss`` logged every logging_steps; on_epoch_end averages those
    losses for the just-finished epoch and saves the adapter only when that
    epoch's mean is a new minimum.
    """

    class TrainLossMonitorCallback(trainer_callback_cls):
        def __init__(self) -> None:
            self.best_loss: float | None = None
            self.best_epoch: float | None = None
            self.best_step: int | None = None
            self.last_loss: float | None = None
            self.best_model_dir = best_model_dir
            # Accumulators for the in-progress epoch.
            self._epoch_loss_sum: float = 0.0
            self._epoch_loss_count: int = 0
            # Guard so each epoch is scored/saved exactly once, whether its loss
            # is logged DURING the epoch (logging_strategy="steps") or AFTER
            # on_epoch_end (logging_strategy="epoch").
            self._finalized_epoch_key: int | None = None

        def on_epoch_begin(self, args, state, control, **kwargs):
            self._epoch_loss_sum = 0.0
            self._epoch_loss_count = 0
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not enabled or not logs:
                return control
            loss = logs.get("loss")
            if loss is None:
                return control
            try:
                loss = float(loss)
            except (TypeError, ValueError):
                return control
            if not math.isfinite(loss):
                return control
            self.last_loss = loss
            self._epoch_loss_sum += loss
            self._epoch_loss_count += 1
            # In logging_strategy="epoch" the epoch's loss is logged AFTER
            # on_epoch_end, so finalize here too. No optimizer step happens
            # between on_epoch_end and this log, so the model is still at its
            # exact end-of-epoch state — safe to save.
            self._maybe_finalize(state, kwargs.get("model"))
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            # In logging_strategy="steps" the per-step losses are already
            # logged during the epoch, so the mean is ready here.
            self._maybe_finalize(state, kwargs.get("model"))
            return control

        def _maybe_finalize(self, state, model) -> None:
            if not enabled or self._epoch_loss_count == 0:
                return
            epoch_key = (
                int(round(state.epoch)) if state.epoch is not None else state.global_step
            )
            if self._finalized_epoch_key == epoch_key:
                return
            self._finalized_epoch_key = epoch_key
            epoch_loss = self._epoch_loss_sum / self._epoch_loss_count
            epoch = float(state.epoch) if state.epoch is not None else None
            if self.best_loss is None or epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.best_epoch = epoch
                self.best_step = state.global_step
                self._save_best_model(model, state)
                if state.is_local_process_zero:
                    print(
                        "[train_loss] new best epoch "
                        f"mean_loss={epoch_loss:.6f} epoch={epoch} "
                        f"global_step={state.global_step} (saved adapter)"
                    )
            elif state.is_local_process_zero:
                print(
                    "[train_loss] epoch "
                    f"mean_loss={epoch_loss:.6f} epoch={epoch} "
                    f"(best is {self.best_loss:.6f} @ epoch {self.best_epoch}; not saved)"
                )

        def summary(self) -> dict:
            return {
                "enabled": enabled,
                "metric": "train_loss_epoch_mean",
                "greater_is_better": False,
                "best_loss": self.best_loss,
                "best_epoch": self.best_epoch,
                "best_step": self.best_step,
                "last_loss": self.last_loss,
                "best_model_dir": str(self.best_model_dir),
            }

        def _save_best_model(self, model, state) -> None:
            if model is None or not state.is_world_process_zero:
                return
            _recreate_dir(self.best_model_dir)
            model.save_pretrained(self.best_model_dir)
            tokenizer.save_pretrained(self.best_model_dir)

    return TrainLossMonitorCallback()


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
    args: TrainConfig,
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