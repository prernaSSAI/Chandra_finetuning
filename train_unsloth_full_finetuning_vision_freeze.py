from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")   # for oom

_HF_CACHE = "/mnt/disk/ml_data/prerna"
for _var in (
    "HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE",
    "HUGGINGFACE_HUB_CACHE", "HUGGINGFACE_ASSETS_CACHE",
):
    os.environ[_var] = _HF_CACHE
os.environ.pop("HF_CACHE_HOME", None)

import base64
import csv
import io
import json
import logging
import pickle
import random
import shutil
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image


try:
    import transformers
    from transformers import TrainerCallback, TrainerState, TrainerControl
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from unsloth import FastVisionModel, is_bf16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    FastVisionModel = None
    UnslothVisionDataCollator = None
    def is_bf16_supported() -> bool:          #  fallback
        return torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

try:
    from trl import SFTTrainer, SFTConfig
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    SFTTrainer = None
    SFTConfig  = None

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

# SECTION 1 — CONFIG

@dataclass
class MemoryConfig:
    device_map: str = "cuda:0"
    gradient_checkpointing: bool = True
    grad_accum_steps: int = 4
    torch_dtype: str = "bfloat16"
    min_response_tokens: int = 32
    cache_clear_every_n_steps: int = 10


@dataclass
class TokenConfig:
    max_length: int = 1536
    auto_profile: bool = True
    profile_n_samples: int = 64


@dataclass
class DataConfig:
    pkl_dir:        str   = "/mnt/disk/ml_data/prerna/data_pkl"
    train_ratio:    float = 0.90
    valid_ratio:    float = 0.05
    test_ratio:     float = 0.05
    seed:           int   = 42
    num_workers:    int   = 0
    batch_size:     int   = 1
    max_image_size: int   = 768


@dataclass
class ModelConfig:
    model_path:           str  = "datalab-to/chandra"
    unsloth_load_in_4bit: bool = False


@dataclass
class TrainConfig:
    epochs:           int   = 10
    lr:               float = 5e-5
    output_dir:       str   = "/mnt/disk/ml_data/prerna/fullfinetune_vision_fix"
    save_every_epoch: bool  = False
    log_file: Optional[str] = "/mnt/disk/ml_data/prerna/fullfinetune_vision_fix/train.log"


@dataclass
class TrainingConfig:
    data:   DataConfig   = field(default_factory=DataConfig)
    model:  ModelConfig  = field(default_factory=ModelConfig)
    train:  TrainConfig  = field(default_factory=TrainConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tokens: TokenConfig  = field(default_factory=TokenConfig)

    def to_dict(self) -> Dict:
        return {
            "data":   asdict(self.data),
            "model":  asdict(self.model),
            "train":  asdict(self.train),
            "memory": asdict(self.memory),
            "tokens": asdict(self.tokens),
        }

# SECTION 2 — LOGGING

def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("chandra")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


log = logging.getLogger("chandra")

# SECTION 3 — REPRODUCIBILITY & DEVICE

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.debug(f"Seed set to {seed}")


def resolve_dtype(dtype_str: str) -> torch.dtype:
    _map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    if dtype_str not in _map:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from {list(_map)}")
    return _map[dtype_str]


def resolve_primary_device(device_map: str) -> torch.device:
    if device_map == "auto":
        dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        dev = torch.device(device_map)
    log.info(f"Primary tensor device: {dev}")
    return dev


def log_gpu_memory(label: str = "") -> None:
    if not torch.cuda.is_available():
        return
    parts = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        parts.append(f"GPU{i}: {alloc:.1f}/{total:.1f} GiB")
    log.info(f"  [MEM{' ' + label if label else ''}] {' | '.join(parts)}")

# SECTION 4 — DATA LOADING

def _normalize_content(content: Any) -> List[Dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        return [content]
    if isinstance(content, list):
        return content
    return []


def _validate_page_sample(page_sample: Any, fname: str, page_idx: int) -> bool:
    if not isinstance(page_sample, dict):
        log.warning(f"  [SKIP] {fname}[{page_idx}]: not a dict")
        return False
    messages = page_sample.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        log.warning(f"  [SKIP] {fname}[{page_idx}]: 'messages' missing or < 2 elements")
        return False
    user_msg, asst_msg = messages[0], messages[1]
    if user_msg.get("role") != "user":
        log.warning(f"  [SKIP] {fname}[{page_idx}]: role[0]!=user")
        return False
    if asst_msg.get("role") != "assistant":
        log.warning(f"  [SKIP] {fname}[{page_idx}]: role[1]!=assistant")
        return False
    user_content = _normalize_content(user_msg.get("content", []))
    has_image = any(
        isinstance(b, dict)
        and b.get("type") == "image"
        and isinstance(b.get("image"), (Image.Image, bytes, bytearray, str))
        for b in user_content
    )
    if not has_image:
        log.warning(f"  [SKIP] {fname}[{page_idx}]: no valid image in user message")
        return False
    asst_content = _normalize_content(asst_msg.get("content", []))
    has_text = any(
        isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
        for b in asst_content
    )
    if not has_text:
        log.warning(f"  [SKIP] {fname}[{page_idx}]: no GT text in assistant message")
        return False
    return True


def _ensure_images_rgb(messages: List[Dict], max_image_size: int = 768) -> List[Dict]:
    user_content = messages[0].get("content", [])
    if not isinstance(user_content, list):
        return messages
    for block in user_content:
        if not isinstance(block, dict) or block.get("type") != "image":
            continue
        img_val = block.get("image")
        if img_val is None:
            continue
        try:
            if isinstance(img_val, Image.Image):
                img = img_val.convert("RGB")
            elif isinstance(img_val, (bytes, bytearray)):
                img = Image.open(io.BytesIO(img_val)).convert("RGB")
            elif isinstance(img_val, str):
                img = Image.open(io.BytesIO(base64.b64decode(img_val))).convert("RGB")
            else:
                continue
            w, h = img.size
            if max(w, h) > max_image_size:
                scale = max_image_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                log.debug(f"  Image resized: ({w}x{h}) → ({new_w}x{new_h})")
            block["image"] = img
        except Exception as e:
            log.debug(f"  Image conversion failed: {e}")
    return messages


def load_pkl_files(pkl_dir: str, max_image_size: int = 768) -> Tuple[List[str], Dict[str, List[Dict]]]:
    pkl_dir_path = Path(pkl_dir)
    if not pkl_dir_path.is_dir():
        raise FileNotFoundError(f"pkl_dir not found: {pkl_dir}")
    pkl_files = sorted(pkl_dir_path.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files in: {pkl_dir}")
    log.info(f"Found {len(pkl_files)} .pkl file(s) in '{pkl_dir}'")
    doc_ids:   List[str]             = []
    doc_pages: Dict[str, List[Dict]] = {}
    for fpath in pkl_files:
        fname  = fpath.name
        doc_id = fpath.stem
        try:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            log.warning(f"  [SKIP] Cannot unpickle {fname}: {e}")
            continue
        if not isinstance(data, list) or len(data) == 0:
            log.warning(f"  [SKIP] {fname}: expected non-empty list")
            continue
        pages: List[Dict] = []
        for page_idx, page_sample in enumerate(data):
            if not _validate_page_sample(page_sample, fname, page_idx):
                continue
            preserved = _ensure_images_rgb(page_sample["messages"], max_image_size=max_image_size)
            pages.append({
                "messages": preserved,
                "file":     fname,
                "doc_id":   doc_id,
                "page_idx": page_idx,
            })
        if not pages:
            log.warning(f"  [SKIP] {fname}: 0 valid pages")
            continue
        doc_ids.append(doc_id)
        doc_pages[doc_id] = pages
        log.info(f"  Loaded {fname}: {len(pages)} page(s)")
    total = sum(len(p) for p in doc_pages.values())
    log.info(f"Load summary: {len(doc_ids)} docs | {total} pages | {len(pkl_files) - len(doc_ids)} skipped")
    return doc_ids, doc_pages


# SECTION 5 — TRAIN / VALID / TEST SPLIT (page-level)

def split_dataset(
    doc_ids: List[str], doc_pages: Dict[str, List[Dict]],
    train_ratio: float, valid_ratio: float, test_ratio: float,
    seed: int, output_dir: str,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("Ratios must sum to 1.0")
    all_pages = [page for doc_id in doc_ids for page in doc_pages[doc_id]]
    n = len(all_pages)
    if n < 3:
        raise ValueError(f"Need >= 3 pages total, got {n}.")
    rng = random.Random(seed)
    shuffled = all_pages.copy()
    rng.shuffle(shuffled)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test  = n - n_train - n_valid
    if n_train < 1 or n_valid < 1 or n_test < 1:
        raise ValueError(f"Empty partition: train={n_train}, valid={n_valid}, test={n_test}")
    train_samples = shuffled[:n_train]
    valid_samples = shuffled[n_train:n_train + n_valid]
    test_samples  = shuffled[n_train + n_valid:]
    log.info(f"Page-level split: {n_train} train | {n_valid} valid | {n_test} test")

    def page_refs(samples: List[Dict]) -> List[Dict[str, Any]]:
        """Store exact page identity for reproducible train/valid/test evaluation."""
        return [
            {
                "doc_id":   str(s["doc_id"]),
                "page_idx": int(s["page_idx"]),
                "file":     str(s.get("file", "")),
            }
            for s in samples
        ]

    manifest = {
        "split_mode": "page_level",
        "seed": seed,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": test_ratio,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "train_pages": page_refs(train_samples),
        "valid_pages": page_refs(valid_samples),
        "test_pages":  page_refs(test_samples),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(Path(output_dir) / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return train_samples, valid_samples, test_samples


# SECTION 6 — TOKEN BUDGET PROFILER

def profile_token_budget(
    samples: List[Dict], processor: Any,
    n_samples: int = 64,
    percentiles: Tuple[int, ...] = (50, 90, 95, 99),
) -> Dict[str, Any]:
    rng = random.Random(0)
    subset = rng.sample(samples, min(n_samples, len(samples)))
    full_lengths: List[int] = []
    prompt_lengths: List[int] = []
    response_lengths: List[int] = []
    log.info(f"[TOKEN PROFILER] Measuring {len(subset)} samples (no truncation)...")
    for s in subset:
        messages = s["messages"]
        try:
            full_out = processor.apply_chat_template(
                [messages], tokenize=True, add_generation_prompt=False,
                return_dict=True, return_tensors="pt", truncation=False,
            )
            prompt_out = processor.apply_chat_template(
                [[messages[0]]], tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", truncation=False,
            )
            full_len   = full_out["input_ids"].shape[-1]
            prompt_len = prompt_out["input_ids"].shape[-1]
            full_lengths.append(full_len)
            prompt_lengths.append(prompt_len)
            response_lengths.append(max(0, full_len - prompt_len))
        except Exception as e:
            log.debug(f"  [PROFILER] Sample failed: {e}")
    if not full_lengths:
        log.warning("[TOKEN PROFILER] No samples profiled.")
        return {}
    def pct(lst: List[int], p: int) -> int:
        idx = max(0, int(len(lst) * p / 100) - 1)
        return sorted(lst)[idx]
    stats: Dict[str, Any] = {
        "n_profiled": len(full_lengths),
        "full":     {f"p{p}": pct(full_lengths, p)     for p in percentiles},
        "prompt":   {f"p{p}": pct(prompt_lengths, p)   for p in percentiles},
        "response": {f"p{p}": pct(response_lengths, p) for p in percentiles},
        "recommended_max_length": pct(full_lengths, 95),
    }
    log.info("[TOKEN PROFILER] Results:")
    log.info(f"  {'Metric':<12} " + "  ".join(f"p{p:>3}" for p in percentiles))
    for metric in ("full", "prompt", "response"):
        row = "  ".join(f"{stats[metric][f'p{p}']:>5}" for p in percentiles)
        log.info(f"  {metric:<12} {row}")
    log.info(f"  → Recommended max_length (p95): {stats['recommended_max_length']}")
    return stats


# SECTION 7 — DATASET

class ChandraOCRDataset(Dataset):
    """
    Thin wrapper around page-level samples.

    Each __getitem__ returns a dict with at minimum:
        {"messages": <conversation list>}
    which is the only key that UnslothVisionDataCollator requires.
    Extra metadata keys (file, doc_id, page_idx) are kept for debugging;
    SFTTrainer ignores them because remove_unused_columns=False.
    """

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "messages": s["messages"],   # ← the only key consumed by the collator
            "file":     s.get("file", ""),
            "doc_id":   s.get("doc_id", ""),
            "page_idx": s.get("page_idx", -1),
        }

# SECTION 8 — MODEL LOADING

def load_model_and_processor(
    model_cfg: ModelConfig, memory_cfg: MemoryConfig,
) -> Tuple[nn.Module, Any, Any]:
    if not UNSLOTH_AVAILABLE:
        raise ImportError("unsloth not installed. Run: pip install unsloth")
    if not HF_AVAILABLE:
        raise ImportError("transformers not installed.")
    dtype = resolve_dtype(memory_cfg.torch_dtype)
    log.info(
        f"[Unsloth] Loading: {model_cfg.model_path}  "
        f"dtype={memory_cfg.torch_dtype}  "
        f"load_in_4bit={model_cfg.unsloth_load_in_4bit}  full_finetuning=True"
    )
    model, processor = FastVisionModel.from_pretrained(
        model_name      = model_cfg.model_path,
        torch_dtype     = dtype,
        load_in_4bit    = model_cfg.unsloth_load_in_4bit,
        full_finetuning = True,    # no LoRA / PEFT
    )
    processor.tokenizer.padding_side = "right"
    if memory_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        log.info("Gradient checkpointing: ENABLED")
    freeze_for_language_only_training(model)
    log_gpu_memory("after model load")
    return model, processor, processor.tokenizer


def freeze_for_language_only_training(model: nn.Module) -> None:
    """Freeze vision encoder; train language backbone + lm_head only."""
    n_frozen = n_trainable = 0
    for name, param in model.named_parameters():
        if name.startswith("model.visual"):
            param.requires_grad_(False)
            n_frozen += param.numel()
        elif name.startswith("model.language_model") or name.startswith("lm_head"):
            param.requires_grad_(True)
            n_trainable += param.numel()
        else:
            param.requires_grad_(False)
            n_frozen += param.numel()

    total = n_frozen + n_trainable
    pct   = 100.0 * n_trainable / total if total else 0.0
    log.info("─" * 65)
    log.info("  [FREEZE] Language-only training:")
    log.info(f"  [FREEZE]   Frozen params     : {n_frozen:>14,}")
    log.info(f"  [FREEZE]   Trainable params  : {n_trainable:>14,}  ({pct:.1f}%)")
    log.info(f"  [FREEZE]   Total params      : {total:>14,}")
    log.info("─" * 65)

    if n_trainable == 0:
        raise RuntimeError("[FREEZE] No trainable parameters found — check param names.")
    if n_frozen == 0:
        log.warning("[FREEZE] No parameters were frozen — unexpected for this config.")

    # diagnostic: show first 30 trainable names
    shown = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            log.debug(f"  TRAIN: {name}")
            shown += 1
            if shown >= 30:
                break

# SECTION 9 — HISTORY CALLBACK

class TrainingHistoryCallback(TrainerCallback):
    """
    Records per-epoch train / eval loss and writes training_history.{json,csv}
    after every epoch — mirrors the original hand-rolled history logic.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.history: List[Dict] = []
        self._best_val: float = float("inf")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None or state.epoch is None:
            return
        # SFTTrainer logs train loss at each step and eval loss after evaluation.
        # We only write to history at epoch boundaries (integer epoch values).
        epoch = round(state.epoch)
        if state.epoch != epoch:
            return  # not yet at an epoch boundary

        # find most-recent entries for each metric
        train_loss = logs.get("loss") or logs.get("train_loss")
        eval_loss  = logs.get("eval_loss")

        if train_loss is None and eval_loss is None:
            return

        # update or append record for this epoch
        existing = next((h for h in self.history if h["epoch"] == epoch), None)
        if existing is None:
            existing = {"epoch": epoch, "train_loss": None, "valid_loss": None, "is_best": False}
            self.history.append(existing)
        if train_loss is not None:
            existing["train_loss"] = round(float(train_loss), 6)
        if eval_loss is not None:
            existing["valid_loss"] = round(float(eval_loss), 6)
            if existing["valid_loss"] < self._best_val:
                self._best_val = existing["valid_loss"]
                for h in self.history:
                    h["is_best"] = False
                existing["is_best"] = True

        save_training_history(self.history, self.output_dir)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        log.info(f"\n{'=' * 65}")
        log.info("  TRAINING COMPLETE")
        log.info(f"{'=' * 65}")
        log.info(f"  {'Epoch':<8} {'Train Loss':<16} {'Valid Loss':<16} Best")
        log.info(f"  {'─' * 55}")
        for h in self.history:
            star = " ← *" if h.get("is_best") else ""
            tl   = f"{h['train_loss']:<16}" if h["train_loss"] is not None else f"{'N/A':<16}"
            vl   = f"{h['valid_loss']:<16}" if h["valid_loss"] is not None else f"{'N/A':<16}"
            log.info(f"  {h['epoch']:<8} {tl} {vl}{star}")
        log.info(f"  Best Val Loss  : {self._best_val:.4f}")
        log.info(f"  Best model     : {Path(self.output_dir) / 'best_model'}")
        log.info("=" * 65)

# SECTION 10 — CHECKPOINT & HISTORY HELPERS

def save_training_history(history: List[Dict], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    if history:
        with open(out / "training_history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=history[0].keys())
            w.writeheader()
            w.writerows(history)
    log.info(f"History -> {out / 'training_history.csv'}")


# SECTION 11 — MAIN

def main(cfg: TrainingConfig) -> None:
    global log
    log = setup_logging(cfg.train.log_file)

    log.info("=" * 65)
    log.info("  CHANDRA OCR — UNSLOTH FULL-FINETUNE (language-only, SFTTrainer)")
    log.info("=" * 65)
    for section, vals in cfg.to_dict().items():
        for k, v in vals.items():
            log.info(f"  [{section}] {k:<28}: {v}")
    log.info("=" * 65)

    if not UNSLOTH_AVAILABLE:
        log.error("unsloth not installed — run: pip install unsloth")
        sys.exit(1)
    if not TRL_AVAILABLE:
        log.error("trl not installed — run: pip install trl")
        sys.exit(1)

    set_seed(cfg.data.seed)
    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.train.output_dir) / "run_config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    #  data 
    doc_ids, doc_pages = load_pkl_files(cfg.data.pkl_dir, cfg.data.max_image_size)
    if not doc_ids:
        log.error("No valid documents loaded — exiting.")
        sys.exit(1)

    train_samples, valid_samples, _ = split_dataset(
        doc_ids, doc_pages,
        cfg.data.train_ratio, cfg.data.valid_ratio, cfg.data.test_ratio,
        cfg.data.seed, cfg.train.output_dir,
    )
    if not train_samples or not valid_samples:
        log.error("Train or valid set is empty — exiting.")
        sys.exit(1)

    # model 
    model, processor, tokenizer = load_model_and_processor(cfg.model, cfg.memory)

    # MUST be called after from_pretrained, before training, to activate
    # Unsloth's fast training kernels (SDPA, custom RoPE, etc.)
    FastVisionModel.for_training(model)
    log.info("[Unsloth] FastVisionModel.for_training() — kernels active")

    # Re-apply freeze: for_training() may touch param settings
    freeze_for_language_only_training(model)

    # token profiler 
    if cfg.tokens.auto_profile:
        profile_stats = profile_token_budget(
            train_samples, processor, n_samples=cfg.tokens.profile_n_samples,
        )
        recommended = profile_stats.get("recommended_max_length")
        if recommended and recommended > cfg.tokens.max_length:
            log.warning(
                f"  [TOKEN PROFILER] max_length={cfg.tokens.max_length} < p95={recommended}. "
                "Many samples will be truncated — consider raising max_length."
            )
        elif recommended and recommended < cfg.tokens.max_length * 0.5:
            log.info(
                f"  [TOKEN PROFILER] max_length={cfg.tokens.max_length} >> p95={recommended}. "
                f"Consider lowering to {recommended} to save memory."
            )

    # datasets 
    train_ds = ChandraOCRDataset(train_samples)
    valid_ds = ChandraOCRDataset(valid_samples)
    log.info(f"Train: {len(train_ds)} samples | Valid: {len(valid_ds)} samples")

    # ── collator — Unsloth's vision-aware collator 
    # UnslothVisionDataCollator(model, processor):
    #   • calls processor.apply_chat_template on each "messages" field
    #   • handles image token expansion, padding, attention masks
    #   • masks prompt tokens in labels (loss only on assistant tokens)
    # This replaces the previous custom SafeCollateFn entirely.
    collator = UnslothVisionDataCollator(model, processor)

    # optimizer string for SFTConfig 
    # PagedAdamW8bit keeps optimizer states in int8 (~4× less GPU memory).
    # Fall back to standard AdamW if bitsandbytes is unavailable.
    if BNB_AVAILABLE:
        optim_str = "paged_adamw_8bit"
        log.info("[OPT] Using paged_adamw_8bit (bitsandbytes)")
    else:
        optim_str = "adamw_torch"
        log.warning("[OPT] bitsandbytes not found — using fp32 AdamW (higher memory usage)")

    #  SFTConfig 
   
    use_bf16 = is_bf16_supported()
    sft_args = SFTConfig(
        output_dir                 = cfg.train.output_dir,
        num_train_epochs           = cfg.train.epochs,
        per_device_train_batch_size= cfg.data.batch_size,
        per_device_eval_batch_size = cfg.data.batch_size,
        gradient_accumulation_steps= cfg.memory.grad_accum_steps,
        learning_rate              = cfg.train.lr,
        bf16                       = use_bf16,
        fp16                       = not use_bf16,
        logging_steps              = 1,
        eval_strategy              = "epoch",   # run validation after every epoch
        save_strategy              = "epoch",
        save_total_limit           = None if cfg.train.save_every_epoch else 1,
        load_best_model_at_end     = True,
        metric_for_best_model      = "eval_loss",
        greater_is_better          = False,
        gradient_checkpointing     = cfg.memory.gradient_checkpointing,
        optim                      = optim_str,
        seed                       = cfg.data.seed,
        dataloader_num_workers     = cfg.data.num_workers,
        report_to                  = "none",    # disable wandb / tensorboard by default
        # ── vision-specific: skip SFT's own tokenisation pipeline ──
        dataset_text_field         = "",
        dataset_kwargs             = {"skip_prepare_dataset": True},
        remove_unused_columns      = False,
        max_seq_length             = cfg.tokens.max_length,
    )

    # SFTTrainer 
    
    trainer = SFTTrainer(
        model          = model,
        tokenizer      = processor,      # pass processor (not just tokenizer) for vision
        data_collator  = collator,        # UnslothVisionDataCollator
        train_dataset  = train_ds,
        eval_dataset   = valid_ds,
        args           = sft_args,
        callbacks      = [TrainingHistoryCallback(cfg.train.output_dir)],
    )

    log.info(
        f"SFTTrainer ready | "
        f"eff. batch = {cfg.data.batch_size} × {cfg.memory.grad_accum_steps} = "
        f"{cfg.data.batch_size * cfg.memory.grad_accum_steps} | "
        f"epochs = {cfg.train.epochs} | lr = {cfg.train.lr}"
    )
    log_gpu_memory("before training")

    trainer.train()

    # save final best model 
    best_dir = Path(cfg.train.output_dir) / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    log.info(f"Best model saved → {best_dir}")
    log_gpu_memory("after training")


# SECTION 12 — ENTRY POINT

if __name__ == "__main__":
    out_dir = "/mnt/disk/ml_data/prerna/fullfinetune_vision_fix"

    cfg = TrainingConfig(
        data=DataConfig(
            pkl_dir        = "/mnt/disk/ml_data/prerna/data_pkl",
            train_ratio    = 0.90,
            valid_ratio    = 0.05,
            test_ratio     = 0.05,
            batch_size     = 1,
            max_image_size = 512,
        ),
        model=ModelConfig(
            model_path           = "datalab-to/chandra",
            unsloth_load_in_4bit = False,  # no QLoRA / 4-bit loading
        ),
        train=TrainConfig(
            epochs           = 10,
            lr               = 5e-5,
            output_dir       = out_dir,
            save_every_epoch = False,
            log_file         = f"{out_dir}/train.log",
        ),
        memory=MemoryConfig(
            device_map                = "cuda:0",
            gradient_checkpointing    = True,
            grad_accum_steps          = 4,
            torch_dtype               = "bfloat16",
            min_response_tokens       = 32,
            cache_clear_every_n_steps = 10,
        ),
        tokens=TokenConfig(
            max_length        = 1536,
            auto_profile      = True,
            profile_n_samples = 64,
        ),
    )
    main(cfg)