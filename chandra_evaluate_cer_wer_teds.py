from __future__ import annotations

import argparse
import base64
import csv
import difflib
import io
import json
import logging
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
PageKey = Tuple[str, int]

import re

try:
    from apted import APTED, Config
    from apted.helpers import Tree
    APTED_AVAILABLE = True
except ImportError:
    APTED_AVAILABLE = False
    APTED = None
    Config = object
    Tree = object

try:
    from unsloth import FastVisionModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    FastVisionModel = None

try:
    from transformers import AutoProcessor
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


@dataclass
class EvalConfig:
    pkl_dir:        str = "/mnt/disk/ml_data/prerna/data_pkl"
    model_path:     str = "/mnt/disk/ml_data/prerna/chandra_output/best_model"
    split_manifest: str = "/mnt/disk/ml_data/prerna/chandra_output/split_manifest.json"
    output_dir:     str = "/mnt/disk/ml_data/prerna/chandra_eval_test"
    log_file:       Optional[str] = "/mnt/disk/ml_data/prerna/chandra_eval_test/eval.log"
    eval_split:     str = "all"

    batch_size:     int = 1
    num_workers:    int = 0

    max_generation_length: int = 12384
    num_beams:             int = 1
    do_sample:             bool = False
    temperature:           float = 1.0
    top_p:                 float = 1.0

    compute_loss: bool = False
    seed:         int = 42
    device:       str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(
        description="Chandra OCR — Evaluation (CER / WER / TEDS)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--pkl_dir", default="/mnt/disk/ml_data/prerna/data_pkl")
    parser.add_argument("--model_path", default="/mnt/disk/ml_data/prerna/chandra_output/best_model")
    parser.add_argument("--split_manifest", default="/mnt/disk/ml_data/prerna/chandra_output/split_manifest.json")
    parser.add_argument("--output_dir", default="/mnt/disk/ml_data/prerna/chandra_eval_test")
    parser.add_argument("--log_file", default="/mnt/disk/ml_data/prerna/chandra_eval_test/eval.log")
    parser.add_argument("--eval_split", default="all", choices=["train", "valid", "test", "all"])

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_generation_length", type=int, default=12384)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--compute_loss", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()
    return EvalConfig(**vars(args))


def setup_logging(log_file: Optional[str]) -> logging.Logger:
    logger = logging.getLogger("chandra_eval")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


log = logging.getLogger("chandra_eval")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)



def load_split_page_keys(manifest_path: str, split: str) -> FrozenSet[PageKey]:
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"split_manifest not found: {manifest_path}")

    with manifest_file.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    split_mode = manifest.get("split_mode", "unknown")
    if split_mode != "page_level":
        raise ValueError(
            f"Expected split_mode='page_level' in manifest, got '{split_mode}'."
        )

    key = f"{split}_pages"
    pages = manifest.get(key)
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"'{key}' is missing or empty in {manifest_path}")

    page_keys: FrozenSet[PageKey] = frozenset(
        (entry["doc_id"], int(entry["page_idx"]))
        for entry in pages
        if "doc_id" in entry and "page_idx" in entry
    )

    log.info(
        "Manifest loaded: split_mode=%s | requested=%s | %d pages",
        split_mode,
        split,
        len(page_keys),
    )
    log.info(
        "  n_train=%d | n_valid=%d | n_test=%d",
        manifest.get("n_train", "?"),
        manifest.get("n_valid", "?"),
        manifest.get("n_test", "?"),
    )

    return page_keys


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
        log.warning("  [SKIP] %s[%d]: not a dict", fname, page_idx)
        return False

    messages = page_sample.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        log.warning("  [SKIP] %s[%d]: 'messages' missing or < 2 elements", fname, page_idx)
        return False

    user_msg, asst_msg = messages[0], messages[1]

    if user_msg.get("role") != "user":
        log.warning("  [SKIP] %s[%d]: first role is not user", fname, page_idx)
        return False

    if asst_msg.get("role") != "assistant":
        log.warning("  [SKIP] %s[%d]: second role is not assistant", fname, page_idx)
        return False

    user_content = _normalize_content(user_msg.get("content", []))

    has_image = any(
        isinstance(block, dict)
        and block.get("type") == "image"
        and isinstance(block.get("image"), (Image.Image, bytes, bytearray, str))
        for block in user_content
    )

    if not has_image:
        log.warning("  [SKIP] %s[%d]: no valid image in user message", fname, page_idx)
        return False

    asst_content = _normalize_content(asst_msg.get("content", []))

    has_text = any(
        isinstance(block, dict)
        and block.get("type") == "text"
        and block.get("text", "").strip()
        for block in asst_content
    )

    if not has_text:
        log.warning("  [SKIP] %s[%d]: no GT text in assistant message", fname, page_idx)
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

        except Exception as exc:
            log.debug("  Image conversion failed: %s", exc)

    return messages


def _extract_gt_text(messages: List[Dict]) -> str:
    assistant_content = _normalize_content(messages[1].get("content", []))

    parts = [
        block["text"]
        for block in assistant_content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and block.get("text", "").strip()
    ]

    return "\n".join(parts)


def load_eval_samples(
    pkl_dir: str,
    page_keys: FrozenSet[PageKey],
) -> List[Dict[str, Any]]:
    pkl_dir_path = Path(pkl_dir)

    if not pkl_dir_path.is_dir():
        raise FileNotFoundError(f"pkl_dir not found: {pkl_dir}")

    samples: List[Dict[str, Any]] = []
    pages_skipped: int = 0
    files_scanned: int = 0

    log.info("Scanning '%s' for %d held-out page(s)...", pkl_dir, len(page_keys))

    for pkl_path in sorted(pkl_dir_path.glob("*.pkl")):
        doc_id = pkl_path.stem
        fname = pkl_path.name

        if not any(dk == doc_id for dk, _ in page_keys):
            continue

        files_scanned += 1

        try:
            with pkl_path.open("rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            log.warning("  [SKIP FILE] %s: cannot unpickle — %s", fname, exc)
            continue

        if not isinstance(data, list) or not data:
            log.warning("  [SKIP FILE] %s: expected non-empty list", fname)
            continue

        loaded_from_file = 0

        for page_idx, page_sample in enumerate(data):
            if (doc_id, page_idx) not in page_keys:
                continue

            if not _validate_page_sample(page_sample, fname, page_idx):
                pages_skipped += 1
                continue

            messages = _ensure_images_rgb(page_sample["messages"])
            gt_text = _extract_gt_text(messages)

            samples.append({
                "messages": messages,
                "gt_text": gt_text,
                "file": fname,
                "doc_id": doc_id,
                "page_idx": page_idx,
            })

            loaded_from_file += 1

        if loaded_from_file:
            log.info("  %s: %d page(s) loaded", fname, loaded_from_file)

    found_keys: Set[PageKey] = {(s["doc_id"], s["page_idx"]) for s in samples}
    missing = page_keys - found_keys

    if missing:
        log.warning(
            "  [MISSING] %d page(s) from manifest not found on disk: %s",
            len(missing),
            sorted(missing)[:10],
        )

    log.info(
        "Load summary: %d file(s) scanned | %d page(s) skipped | %d sample(s) ready",
        files_scanned,
        pages_skipped,
        len(samples),
    )

    return samples


class ChandraEvalDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            s = self.samples[idx]
            return {
                "messages": s["messages"],
                "gt_text": s["gt_text"],
                "file": s["file"],
                "doc_id": s["doc_id"],
                "page_idx": s["page_idx"],
            }
        except Exception as exc:
            log.warning("  [DATASET idx=%d] %s", idx, exc)
            return None


class EvalCollateFn:
    def __init__(self, processor: Any, compute_loss: bool = False):
        self.processor = processor
        self.compute_loss = compute_loss

    def __call__(self, batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        valid = [item for item in batch if item is not None]

        if not valid:
            return None

        prompt_convs = [[item["messages"][0]] for item in valid]

        try:
            prompt_inputs = self.processor.apply_chat_template(
                prompt_convs,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
        except Exception as exc:
            log.warning("  [COLLATE] prompt apply_chat_template failed: %s", exc)
            return None

        result: Dict[str, Any] = {
            "input_ids": prompt_inputs["input_ids"],
            "attention_mask": prompt_inputs["attention_mask"],
            "gt_texts": [item["gt_text"] for item in valid],
            "files": [item["file"] for item in valid],
            "doc_ids": [item["doc_id"] for item in valid],
            "page_idxs": [item["page_idx"] for item in valid],
            "prompt_input_ids": prompt_inputs["input_ids"].clone(),
        }

        for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            if key in prompt_inputs:
                result[key] = prompt_inputs[key]

        if self.compute_loss:
            result.update(self._build_loss_inputs(valid, prompt_inputs))

        return result

    def _build_loss_inputs(
        self,
        valid: List[Dict[str, Any]],
        prompt_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        full_convs = [item["messages"] for item in valid]

        try:
            full_inputs = self.processor.apply_chat_template(
                full_convs,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
        except Exception as exc:
            log.warning("  [COLLATE] full-conversation apply_chat_template failed: %s", exc)
            return {}

        labels = full_inputs["input_ids"].clone()

        prompt_len = int((prompt_inputs["attention_mask"][0] == 1).sum().item())
        labels[:, :prompt_len] = -100

        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        extra: Dict[str, Any] = {
            "loss_input_ids": full_inputs["input_ids"],
            "loss_attention_mask": full_inputs["attention_mask"],
            "labels": labels,
        }

        for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            if key in full_inputs:
                extra[f"loss_{key}"] = full_inputs[key]

        return extra


def load_model_and_processor(
    model_path: str,
    device: torch.device,
) -> Tuple[nn.Module, Any]:
    if not HF_AVAILABLE:
        raise ImportError("transformers not installed.")

    if not UNSLOTH_AVAILABLE:
        raise ImportError("unsloth not installed. Run: pip install unsloth")

    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32

    try:
        log.info("Loading processor from checkpoint: %s", model_path)
        processor = AutoProcessor.from_pretrained(model_path)
    except Exception as exc:
        log.warning("Processor load from checkpoint failed: %s", exc)
        log.info("Loading processor from base model: datalab-to/chandra")
        processor = AutoProcessor.from_pretrained("datalab-to/chandra")

    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "right"

    log.info("Loading Unsloth model from: %s", model_path)

    model, _ = FastVisionModel.from_pretrained(
        model_name=model_path,
        torch_dtype=dtype,
        load_in_4bit=False,
        full_finetuning=False,
    )

    FastVisionModel.for_inference(model)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    log.info(
        "Model ready — %s params | dtype=%s | device=%s",
        f"{n_params:,}",
        next(model.parameters()).dtype,
        device,
    )

    return model, processor


_GEN_KEYS = frozenset([
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "mm_token_type_ids",
])

_LOSS_KEYS = frozenset([
    "loss_input_ids",
    "loss_attention_mask",
    "labels",
    "loss_pixel_values",
    "loss_image_grid_thw",
    "loss_mm_token_type_ids",
])


def _batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
    keys: FrozenSet,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key in keys and isinstance(value, torch.Tensor)
    }


def _lxml_available() -> bool:
    try:
        import lxml  # noqa: F401
        return True
    except ImportError:
        return False


def html_to_plain_text(html: str) -> str:
    if BS4_AVAILABLE:
        parser = "lxml" if _lxml_available() else "html.parser"
        return BeautifulSoup(html, parser).get_text(separator=" ").strip()

    return re.sub(r"<[^>]+>", " ", html).strip()


class TableTree(Tree):
    def __init__(
        self,
        tag: str,
        colspan: Optional[str] = None,
        rowspan: Optional[str] = None,
        content: str = "",
        *children: "TableTree",
    ):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)


class TEDSConfig(Config):
    def rename(self, node1: TableTree, node2: TableTree) -> float:
        if node1.tag != node2.tag:
            return 1.0

        if node1.tag == "td":
            if node1.colspan != node2.colspan or node1.rowspan != node2.rowspan:
                return 1.0

            return 1.0 - difflib.SequenceMatcher(
                None,
                node1.content or "",
                node2.content or "",
            ).ratio()

        return 0.0

    def children(self, node: TableTree) -> List[TableTree]:
        return node.children


def _normalize_html_table(html: str) -> Optional[Any]:
    if not BS4_AVAILABLE:
        raise ImportError(
            "beautifulsoup4 is required for official TEDS. Run: pip install beautifulsoup4 lxml apted"
        )

    parser = "lxml" if _lxml_available() else "html.parser"
    soup = BeautifulSoup(html or "", parser)
    table = soup.find("table")

    if table is None:
        return None

    return table


def _cell_text(cell: Any) -> str:
    return re.sub(r"\s+", " ", cell.get_text(separator=" ")).strip()


def html_to_tree(html: str) -> Optional[TableTree]:
    table = _normalize_html_table(html)

    if table is None:
        return None

    def convert(node: Any) -> Optional[TableTree]:
        if getattr(node, "name", None) is None:
            return None

        children: List[TableTree] = []

        for child_node in node.children:
            child = convert(child_node)
            if child is not None:
                children.append(child)

        tag = node.name.lower()

        if tag in ("td", "th"):
            return TableTree(
                "td",
                str(node.get("colspan", "1")),
                str(node.get("rowspan", "1")),
                _cell_text(node),
                *children,
            )

        return TableTree(tag, None, None, "", *children)

    return convert(table)


def count_nodes(node: TableTree) -> int:
    return 1 + sum(count_nodes(child) for child in node.children)


def compute_teds(pred: str, gt: str) -> float:
    if not APTED_AVAILABLE:
        raise ImportError("apted is required for official TEDS. Run: pip install apted")

    if "<table" not in (pred or "").lower() or "<table" not in (gt or "").lower():
        return 0.0

    pred_tree = html_to_tree(pred)
    gt_tree = html_to_tree(gt)

    if pred_tree is None or gt_tree is None:
        return 0.0

    max_nodes = max(count_nodes(pred_tree), count_nodes(gt_tree))

    if max_nodes == 0:
        return 0.0

    distance = APTED(pred_tree, gt_tree, TEDSConfig()).compute_edit_distance()
    score = 1.0 - (float(distance) / float(max_nodes))

    return round(max(0.0, min(1.0, score)), 6)


def _edit_distance_rate(ref: List[Any], hyp: List[Any]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0

    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))

    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i

        for j in range(1, m + 1):
            tmp = dp[j]

            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])

            prev = tmp

    return dp[m] / n


def compute_cer(pred: str, gt: str) -> float:
    if JIWER_AVAILABLE:
        try:
            transform = jiwer.Compose([jiwer.ReduceToListOfListOfChars()])
            return jiwer.wer(
                [gt],
                [pred],
                truth_transform=transform,
                hypothesis_transform=transform,
            )
        except Exception:
            pass

    return _edit_distance_rate(list(gt), list(pred))


def compute_wer(pred: str, gt: str) -> float:
    if JIWER_AVAILABLE:
        try:
            return jiwer.wer([gt], [pred])
        except Exception:
            pass

    return _edit_distance_rate(gt.split(), pred.split())


def compute_metrics(pred: str, gt: str) -> Dict[str, float]:
    pred_plain = html_to_plain_text(pred)
    gt_plain = html_to_plain_text(gt)

    return {
        "cer": round(min(compute_cer(pred_plain, gt_plain), 999.0), 6),
        "wer": round(min(compute_wer(pred_plain, gt_plain), 999.0), 6),
        "teds": round(compute_teds(pred, gt), 6),
    }


def compute_batch_loss(
    model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
) -> Optional[float]:
    if "labels" not in batch:
        return None

    try:
        loss_inputs = _batch_to_device(batch, device, _LOSS_KEYS)

        forward_kwargs: Dict[str, torch.Tensor] = {
            "input_ids": loss_inputs["loss_input_ids"],
            "attention_mask": loss_inputs["loss_attention_mask"],
            "labels": loss_inputs["labels"],
        }

        for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            src = f"loss_{key}"
            if src in loss_inputs:
                forward_kwargs[key] = loss_inputs[src]

        outputs = model(**forward_kwargs)

        if outputs.loss is not None and not torch.isnan(outputs.loss):
            return float(outputs.loss.item())

    except Exception as exc:
        log.warning("  [LOSS] Forward pass failed: %s", exc)

    return None


def decode_generated_text(
    generated_ids: torch.Tensor,
    prompt_input_ids: torch.Tensor,
    tokenizer: Any,
) -> List[str]:
    decoded: List[str] = []

    for row_idx in range(generated_ids.shape[0]):
        full_ids = generated_ids[row_idx]
        prompt_ids = prompt_input_ids[row_idx]
        prompt_len = prompt_ids.shape[0]

        use_trimmed = (
            full_ids.shape[0] >= prompt_len
            and torch.equal(full_ids[:prompt_len].cpu(), prompt_ids.cpu())
        )

        ids_to_decode = full_ids[prompt_len:] if use_trimmed else full_ids

        try:
            text = tokenizer.decode(ids_to_decode, skip_special_tokens=True).strip()
        except Exception as exc:
            log.warning("  [DECODE sample=%d] %s", row_idx, exc)
            text = ""

        decoded.append(text)

    return decoded


def run_evaluation(
    model: nn.Module,
    processor: Any,
    loader: DataLoader,
    device: torch.device,
    cfg: EvalConfig,
) -> List[Dict[str, Any]]:
    model.eval()

    results: List[Dict[str, Any]] = []
    batch_errors = 0

    gen_params: Dict[str, Any] = {
        "max_new_tokens": cfg.max_generation_length,
        "num_beams": cfg.num_beams,
        "do_sample": cfg.do_sample,
    }

    if cfg.do_sample:
        gen_params["temperature"] = cfg.temperature
        gen_params["top_p"] = cfg.top_p

    iterator = tqdm(loader, desc=f"Evaluating [{cfg.eval_split}]") if TQDM_AVAILABLE else loader

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator, start=1):
            if batch is None:
                batch_errors += 1
                continue

            batch_size = len(batch["gt_texts"])
            gen_inputs = _batch_to_device(batch, device, _GEN_KEYS)

            t0 = time.perf_counter()

            try:
                generated_ids = model.generate(**gen_inputs, **gen_params)
            except Exception as exc:
                log.warning("  [Batch %d] Generation failed: %s", batch_idx, exc)
                batch_errors += batch_size
                continue

            latency_s = time.perf_counter() - t0

            predictions = decode_generated_text(
                generated_ids=generated_ids,
                prompt_input_ids=batch["prompt_input_ids"],
                tokenizer=processor.tokenizer,
            )

            batch_loss: Optional[float] = None

            if cfg.compute_loss:
                batch_loss = compute_batch_loss(model, batch, device)

            per_sample_latency = latency_s / max(batch_size, 1)

            for i in range(batch_size):
                pred = predictions[i]
                gt = batch["gt_texts"][i]
                metrics = compute_metrics(pred, gt)

                row: Dict[str, Any] = {
                    "file": batch["files"][i],
                    "doc_id": batch["doc_ids"][i],
                    "page_idx": batch["page_idxs"][i],
                    "prediction": pred,
                    "ground_truth": gt,
                    "cer": metrics["cer"],
                    "wer": metrics["wer"],
                    "teds": metrics["teds"],
                    "latency_s": round(per_sample_latency, 4),
                }

                if batch_loss is not None:
                    row["loss"] = round(batch_loss, 6)

                results.append(row)

            if TQDM_AVAILABLE and results:
                n = len(results)

                iterator.set_postfix(
                    cer=f"{sum(r['cer'] for r in results) / n:.3f}",
                    wer=f"{sum(r['wer'] for r in results) / n:.3f}",
                    teds=f"{sum(r['teds'] for r in results) / n:.3f}",
                )

    log.info(
        "Evaluation complete: %d samples evaluated | %d batch/sample errors skipped",
        len(results),
        batch_errors,
    )

    return results


def compute_aggregate(results: List[Dict[str, Any]], n_requested: int) -> Dict[str, Any]:
    if not results:
        return {
            "sample_count": 0,
            "skipped_count": n_requested,
            "avg_cer": None,
            "avg_wer": None,
            "avg_teds": None,
            "avg_latency_s": None,
            "avg_loss": None,
        }

    n = len(results)
    losses = [row["loss"] for row in results if "loss" in row]

    return {
        "sample_count": n,
        "skipped_count": n_requested - n,
        "avg_cer": round(sum(r["cer"] for r in results) / n, 6),
        "avg_wer": round(sum(r["wer"] for r in results) / n, 6),
        "avg_teds": round(sum(r["teds"] for r in results) / n, 6),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / n, 4),
        "avg_loss": round(sum(losses) / len(losses), 6) if losses else None,
    }


def save_results(
    results: List[Dict[str, Any]],
    aggregate: Dict[str, Any],
    cfg: EvalConfig,
) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "eval_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)

    with (output_dir / "evaluation_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    if results:
        with (output_dir / "evaluation_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(results[0].keys()),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(results)

    log.info("Saved outputs to: %s", output_dir.resolve())


def main(cfg: EvalConfig) -> None:
    global log
    log = setup_logging(cfg.log_file)

    log.info("=" * 70)
    log.info("CHANDRA OCR — EVALUATION  (CER / WER / TEDS)")
    log.info("=" * 70)

    for key, value in cfg.to_dict().items():
        log.info("  %-24s: %s", key, value)

    log.info("=" * 70)

    if not JIWER_AVAILABLE:
        log.warning("jiwer not installed — using Python edit-distance fallback for CER/WER")

    if not BS4_AVAILABLE:
        log.warning(
            "beautifulsoup4 not installed — official TEDS will fail. "
            "Run: pip install beautifulsoup4 lxml apted"
        )

    if not APTED_AVAILABLE:
        log.warning("apted not installed — official TEDS will fail. Run: pip install apted")

    set_seed(cfg.seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.device)
    log.info("Using device: %s", device)

    model, processor = load_model_and_processor(cfg.model_path, device)

    splits_to_run = ["train", "valid", "test"] if cfg.eval_split == "all" else [cfg.eval_split]
    all_aggregates: Dict[str, Any] = {}

    for split in splits_to_run:
        log.info("")
        log.info("=" * 70)
        log.info("EVALUATING SPLIT: %s", split.upper())
        log.info("=" * 70)

        try:
            page_keys = load_split_page_keys(cfg.split_manifest, split=split)
        except Exception as exc:
            log.error("  Failed to load manifest for split '%s': %s — skipping.", split, exc)
            continue

        if not page_keys:
            log.warning("  No pages found for split '%s' — skipping.", split)
            continue

        samples = load_eval_samples(cfg.pkl_dir, page_keys)

        if not samples:
            log.warning("  No valid samples loaded for split '%s' — skipping.", split)
            continue

        log.info("  %d / %d pages loaded for split '%s'", len(samples), len(page_keys), split)

        split_output_dir = str(Path(cfg.output_dir) / split)
        split_cfg = EvalConfig(**{**cfg.to_dict(), "eval_split": split, "output_dir": split_output_dir})

        dataset = ChandraEvalDataset(samples)

        collate_fn = EvalCollateFn(
            processor=processor,
            compute_loss=cfg.compute_loss,
        )

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )

        results = run_evaluation(model, processor, loader, device, split_cfg)
        aggregate = compute_aggregate(results, n_requested=len(samples))
        save_results(results, aggregate, split_cfg)

        all_aggregates[split] = aggregate

        log.info(
            "  [%s] CER=%.4f | WER=%.4f | TEDS=%.4f | n=%d",
            split,
            aggregate["avg_cer"] or 0,
            aggregate["avg_wer"] or 0,
            aggregate["avg_teds"] or 0,
            aggregate["sample_count"],
        )

    combined_path = Path(cfg.output_dir) / "all_splits_summary.json"

    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(all_aggregates, f, indent=2)

    log.info("")
    log.info("=" * 70)
    log.info("ALL SPLITS COMPLETE")
    log.info("=" * 70)

    for split, agg in all_aggregates.items():
        if agg["sample_count"]:
            log.info(
                "  [%-5s] CER=%.4f | WER=%.4f | TEDS=%.4f | n=%d",
                split,
                agg["avg_cer"],
                agg["avg_wer"],
                agg["avg_teds"],
                agg["sample_count"],
            )

    log.info("  Summary → %s", combined_path.resolve())
    log.info("=" * 70)


if __name__ == "__main__":
    main(parse_args())