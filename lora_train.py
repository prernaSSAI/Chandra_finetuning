from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")  #for oom 

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
from PIL import Image

try:
    from unsloth import FastVisionModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    FastVisionModel = None

try:
    import transformers
    from transformers import TrainingArguments
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from unsloth.trainer import UnslothVisionDataCollator
    UNSLOTH_COLLATOR_AVAILABLE = True
except ImportError:
    UNSLOTH_COLLATOR_AVAILABLE = False

try:
    from trl import SFTTrainer, SFTConfig
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False

try:
    from datasets import Dataset as HFDataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    random_state: int = 3407
    use_rslora: bool = False
    target_modules: str = "all-linear"
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True


@dataclass
class DataConfig:
    pkl_dir: str = "./data_pkl"
    train_ratio: float = 0.90
    valid_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42
    num_workers: int = 0


@dataclass
class ModelConfig:
    model_path: str = "datalab-to/chandra"
    load_in_4bit: bool = False
    torch_dtype: str = "bfloat16"


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 2e-4
    output_dir: str = "/mnt/disk/ml_data/prerna/chandra_output"
    log_file: Optional[str] = "/mnt/disk/ml_data/prerna/chandra_output/train.log"

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    max_length: int = 2048
    logging_steps: int = 1
    optim: str = "adamw_8bit"


@dataclass
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def to_dict(self) -> Dict:
        return {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "train": asdict(self.train),
            "lora": asdict(self.lora),
        }


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.debug(f"Seed set to {seed}")


def resolve_dtype(dtype_str: str) -> torch.dtype:
    _map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in _map:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from {list(_map)}")
    return _map[dtype_str]


def log_gpu_memory(label: str = "") -> None:
    if not torch.cuda.is_available():
        return

    parts = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        parts.append(f"GPU{i}: {alloc:.1f}/{total:.1f} GiB")

    log.info(f"  [MEM{' ' + label if label else ''}] {' | '.join(parts)}")


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
        log.warning(f"  [SKIP] {fname}[{page_idx}]: role[0] != 'user'")
        return False

    if asst_msg.get("role") != "assistant":
        log.warning(f"  [SKIP] {fname}[{page_idx}]: role[1] != 'assistant'")
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
        isinstance(b, dict)
        and b.get("type") == "text"
        and b.get("text", "").strip()
        for b in asst_content
    )

    if not has_text:
        log.warning(f"  [SKIP] {fname}[{page_idx}]: no GT text in assistant message")
        return False

    return True


def _ensure_images_rgb(messages: List[Dict]) -> List[Dict]:
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

            img.info["dpi"] = (600, 600)
            block["image"] = img

        except Exception as e:
            log.debug(f"  Image conversion failed: {e}")

    return messages


def load_pkl_files(pkl_dir: str) -> Tuple[List[str], Dict[str, List[Dict]]]:
    pkl_dir_path = Path(pkl_dir)

    if not pkl_dir_path.is_dir():
        raise FileNotFoundError(f"pkl_dir not found: {pkl_dir}")

    pkl_files = sorted(pkl_dir_path.glob("*.pkl"))

    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files in: {pkl_dir}")

    log.info(f"Found {len(pkl_files)} .pkl file(s) in '{pkl_dir}'")

    doc_ids: List[str] = []
    doc_pages: Dict[str, List[Dict]] = {}

    for fpath in pkl_files:
        fname = fpath.name
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

            preserved = _ensure_images_rgb(page_sample["messages"])

            pages.append({
                "messages": preserved,
                "file": fname,
                "doc_id": doc_id,
                "page_idx": page_idx,
            })

        if not pages:
            log.warning(f"  [SKIP] {fname}: 0 valid pages")
            continue

        doc_ids.append(doc_id)
        doc_pages[doc_id] = pages
        log.info(f"  Loaded {fname}: {len(pages)} page(s)")

    total = sum(len(p) for p in doc_pages.values())

    log.info(
        f"Load summary: {len(doc_ids)} docs | {total} pages | "
        f"{len(pkl_files) - len(doc_ids)} skipped"
    )

    return doc_ids, doc_pages


def split_dataset(
    doc_ids: List[str],
    doc_pages: Dict[str, List[Dict]],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
    output_dir: str,
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
    n_test = n - n_train - n_valid

    if n_train < 1 or n_valid < 1 or n_test < 1:
        raise ValueError(f"Empty partition: train={n_train}, valid={n_valid}, test={n_test}")

    train_samples = shuffled[:n_train]
    valid_samples = shuffled[n_train:n_train + n_valid]
    test_samples = shuffled[n_train + n_valid:]

    log.info(f"Page-level split: {n_train} train | {n_valid} valid | {n_test} test")

    manifest = {
        "split_mode": "page_level",
        "seed": seed,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": test_ratio,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "train_pages": [
            {
                "file": sample["file"],
                "doc_id": sample["doc_id"],
                "page_idx": int(sample["page_idx"]),
            }
            for sample in train_samples
        ],
        "valid_pages": [
            {
                "file": sample["file"],
                "doc_id": sample["doc_id"],
                "page_idx": int(sample["page_idx"]),
            }
            for sample in valid_samples
        ],
        "test_pages": [
            {
                "file": sample["file"],
                "doc_id": sample["doc_id"],
                "page_idx": int(sample["page_idx"]),
            }
            for sample in test_samples
        ],
    }

    manifest_path = Path(output_dir) / "split_manifest.json"
    os.makedirs(output_dir, exist_ok=True)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Split manifest -> {manifest_path}")

    return train_samples, valid_samples, test_samples


def samples_to_hf_dataset(samples: List[Dict]) -> "HFDataset":
    rows = [{"messages": s["messages"]} for s in samples]
    return HFDataset.from_list(rows)


def load_model_and_processor(
    model_cfg: ModelConfig,
    lora_cfg: LoRAConfig,
) -> Tuple[nn.Module, Any, Any]:

    if not UNSLOTH_AVAILABLE:
        raise ImportError("unsloth not installed. Run: pip install unsloth")

    if not HF_AVAILABLE:
        raise ImportError("transformers not installed.")

    dtype = resolve_dtype(model_cfg.torch_dtype)

    log.info(
        f"[Unsloth] Loading model: {model_cfg.model_path}  "
        f"dtype={model_cfg.torch_dtype}  "
        f"load_in_4bit={model_cfg.load_in_4bit}  "
        f"full_finetuning=False  (LoRA will be applied)"
    )

    model, processor = FastVisionModel.from_pretrained(
        model_name=model_cfg.model_path,
        torch_dtype=dtype,
        load_in_4bit=model_cfg.load_in_4bit,
        full_finetuning=False,
        use_gradient_checkpointing="unsloth",
    )

    log.info(
        f"[LoRA] Attaching adapters: r={lora_cfg.r}  alpha={lora_cfg.lora_alpha}  "
        f"vision={lora_cfg.finetune_vision_layers}  "
        f"language={lora_cfg.finetune_language_layers}  "
        f"attention={lora_cfg.finetune_attention_modules}  "
        f"mlp={lora_cfg.finetune_mlp_modules}"
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=lora_cfg.finetune_vision_layers,
        finetune_language_layers=lora_cfg.finetune_language_layers,
        finetune_attention_modules=lora_cfg.finetune_attention_modules,
        finetune_mlp_modules=lora_cfg.finetune_mlp_modules,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        random_state=lora_cfg.random_state,
        use_rslora=lora_cfg.use_rslora,
        loftq_config=None,
        target_modules=lora_cfg.target_modules,
    )

    tokenizer = processor.tokenizer
    processor.tokenizer.padding_side = "right"

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * n_trainable / n_total if n_total else 0.0

    log.info(f"[LoRA] Trainable params: {n_trainable:,} / {n_total:,}  ({pct:.2f}%)")

    log_gpu_memory("after model + LoRA load")

    return model, processor, tokenizer


def main(cfg: TrainingConfig) -> None:
    global log
    log = setup_logging(cfg.train.log_file)

    log.info("=" * 65)
    log.info("  CHANDRA OCR — UNSLOTH LoRA TRAINING  (SFTTrainer)")
    log.info("=" * 65)

    for section, vals in cfg.to_dict().items():
        for k, v in vals.items():
            log.info(f"  [{section}] {k:<32}: {v}")

    log.info("=" * 65)

    if not UNSLOTH_AVAILABLE:
        log.error("Unsloth not installed — run: pip install unsloth")
        sys.exit(1)

    if not TRL_AVAILABLE:
        log.error("TRL not installed — run: pip install trl")
        sys.exit(1)

    if not HF_DATASETS_AVAILABLE:
        log.error("datasets not installed — run: pip install datasets")
        sys.exit(1)

    if not UNSLOTH_COLLATOR_AVAILABLE:
        log.error("UnslothVisionDataCollator not found. Update unsloth.")
        sys.exit(1)

    set_seed(cfg.data.seed)
    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(cfg.train.output_dir) / "run_config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    doc_ids, doc_pages = load_pkl_files(cfg.data.pkl_dir)

    if not doc_ids:
        log.error("No valid documents loaded — exiting.")
        sys.exit(1)

    train_samples, valid_samples, test_samples = split_dataset(
        doc_ids,
        doc_pages,
        cfg.data.train_ratio,
        cfg.data.valid_ratio,
        cfg.data.test_ratio,
        cfg.data.seed,
        cfg.train.output_dir,
    )

    if not train_samples or not valid_samples:
        log.error("Train or valid set is empty — exiting.")
        sys.exit(1)

    log.info(f"Train: {len(train_samples)} samples | Valid: {len(valid_samples)} samples")

    # 🔍 DEBUG START
    log.info(">>> Converting TRAIN samples to HF dataset...")
    train_hf = samples_to_hf_dataset(train_samples)
    log.info(">>> TRAIN dataset created")

    log.info(">>> Converting VALID samples to HF dataset...")
    valid_hf = samples_to_hf_dataset(valid_samples)
    log.info(">>> VALID dataset created")

    log.info(">>> Loading model now...")
    model, processor, tokenizer = load_model_and_processor(cfg.model, cfg.lora)
    log.info(">>> Model loaded successfully")
    # 🔍 DEBUG END

    FastVisionModel.for_training(model)
    log.info("[Unsloth] FastVisionModel.for_training() — training kernels active")

    sft_cfg = SFTConfig(
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        warmup_steps=cfg.train.warmup_steps,
        num_train_epochs=cfg.train.epochs,
        learning_rate=cfg.train.lr,
        optim=cfg.train.optim,
        weight_decay=cfg.train.weight_decay,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        logging_steps=cfg.train.logging_steps,
        seed=cfg.train.seed,
        output_dir=cfg.train.output_dir,
        report_to="none",
        eval_strategy="epoch",
        save_strategy="best",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=cfg.train.max_length,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=train_hf,
        eval_dataset=valid_hf,
        args=sft_cfg,
    )

    log.info(
        f"Effective batch size: "
        f"{cfg.train.per_device_train_batch_size * cfg.train.gradient_accumulation_steps}"
    )

    log_gpu_memory("before training")

    log.info("\n" + "=" * 65)
    log.info("  STARTING TRAINING")
    log.info("=" * 65)

    trainer_result = trainer.train()

    log.info("\n" + "=" * 65)
    log.info("  TRAINING COMPLETE")
    log.info("=" * 65)
    log.info(f"  Train runtime : {trainer_result.metrics.get('train_runtime', 0):.1f}s")
    log.info(f"  Train loss    : {trainer_result.metrics.get('train_loss', 'n/a')}")

    best_dir = Path(cfg.train.output_dir) / "best_model"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    log.info(f"  Best model saved → {best_dir}")

    log_gpu_memory("after training")


if __name__ == "__main__":
    cfg = TrainingConfig(
        data=DataConfig(
            pkl_dir="./data_pkl",
            train_ratio=0.90,
            valid_ratio=0.05,
            test_ratio=0.05,
            seed=42,
        ),
        model=ModelConfig(
            model_path="datalab-to/chandra",
            load_in_4bit=False,
            torch_dtype="bfloat16",
        ),
        train=TrainConfig(
            epochs=10,
            lr=2e-4,
            output_dir="/mnt/disk/ml_data/prerna/chandra_output",
            log_file="/mnt/disk/ml_data/prerna/chandra_output/train.log",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            max_length=2048,
            logging_steps=1,
            optim="adamw_8bit",
        ),
        lora=LoRAConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            random_state=3407,
            use_rslora=False,
            target_modules="all-linear",
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
        ),
    )

    main(cfg)