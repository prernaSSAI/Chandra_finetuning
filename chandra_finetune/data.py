from __future__ import annotations

import json
import pickle
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from prompts import OCR_PROMPT


@dataclass
class ChandraSample:
    """Normalized sample used by training and inference."""

    messages: list[dict[str, Any]]
    image: Image.Image
    prompt: str
    reference: str | None = None
    metadata: dict[str, Any] | None = None

    def training_record(self) -> dict[str, Any]:
        return {"messages": self.messages}


def load_chandra_dataset(path: str | Path, *, prompt_fallback: str = OCR_PROMPT) -> list[ChandraSample]:
    """Load Chandra samples from pickle, JSON/JSONL, or an Arrow dataset directory."""

    dataset_path = Path(path)
    raw_records = _load_records(dataset_path)
    if isinstance(raw_records, dict):
        for key in ("data", "samples", "dataset", "records"):
            if key in raw_records and isinstance(raw_records[key], list):
                raw_records = raw_records[key]
                break

    if not isinstance(raw_records, list):
        raise ValueError(f"Expected {dataset_path} to contain a list of samples.")

    samples = [
        normalize_sample(record, base_dir=dataset_path.parent, prompt_fallback=prompt_fallback)
        for record in raw_records
    ]
    if not samples:
        raise ValueError(f"No samples were loaded from {dataset_path}.")
    return samples


def normalize_sample(
    sample: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
    prompt_fallback: str = OCR_PROMPT,
) -> ChandraSample:
    """Normalize one sample into the Unsloth vision conversation format."""

    if not isinstance(sample, dict):
        raise ValueError(f"Expected sample to be a dict, got {type(sample).__name__}.")

    metadata = _parse_metadata(sample.get("metadata"))
    messages = sample.get("messages")

    if isinstance(messages, list):
        image = _find_image(sample, messages, base_dir=base_dir)
        prompt = _extract_user_prompt(messages) or prompt_fallback
        reference = _extract_assistant_text(messages)
        normalized_messages = _normalize_messages(messages, image=image, prompt_fallback=prompt)
    elif "image" in sample and "prompt" in sample:
        image = _coerce_image(sample["image"], base_dir=base_dir)
        prompt = str(sample.get("prompt") or prompt_fallback)
        reference = sample.get("reference")
        reference = str(reference) if reference not in (None, "") else None
        normalized_messages = make_messages(image=image, prompt=prompt, reference=reference)
    else:
        raise ValueError("Sample must contain either a 'messages' list or normalized image/prompt fields.")

    return ChandraSample(
        messages=normalized_messages,
        image=image,
        prompt=prompt,
        reference=reference,
        metadata=metadata,
    )


def split_samples(
    samples: list[ChandraSample],
    *,
    eval_ratio: float = 0.1,
    seed: int = 3407,
) -> tuple[list[ChandraSample], list[ChandraSample]]:
    """Create a deterministic train/held-out split."""

    if not 0 <= eval_ratio < 1:
        raise ValueError("Split ratio must be >= 0 and < 1.")

    if eval_ratio == 0 or len(samples) < 2:
        return list(samples), []

    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    eval_count = max(1, int(round(len(samples) * eval_ratio)))
    eval_count = min(eval_count, len(samples) - 1)
    eval_indices = set(indices[:eval_count])

    train = [sample for index, sample in enumerate(samples) if index not in eval_indices]
    eval_set = [sample for index, sample in enumerate(samples) if index in eval_indices]
    return train, eval_set


def save_samples_to_pickle(samples: Iterable[ChandraSample], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump([sample_to_record(sample) for sample in samples], handle)


def save_samples_to_arrow(samples: Iterable[ChandraSample], path: str | Path) -> None:
    try:
        from datasets import Dataset, Features, Image as DatasetImage, Value
    except ImportError as exc:
        raise RuntimeError(
            "Saving Arrow datasets requires the 'datasets' package. "
            "Install dependencies from README_CHANDRA_FINETUNE.md."
        ) from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features = Features(
        {
            "image": DatasetImage(),
            "prompt": Value("string"),
            "reference": Value("string"),
            "metadata": Value("string"),
        }
    )
    dataset = Dataset.from_list(samples_to_arrow_rows(samples), features=features)
    dataset.save_to_disk(str(output_path))


def sample_to_record(sample: ChandraSample) -> dict[str, Any]:
    record = sample.training_record()
    record["metadata"] = sample.metadata or {}
    return record


def samples_to_arrow_rows(samples: Iterable[ChandraSample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        rows.append(
            {
                "image": {"bytes": _image_to_png_bytes(sample.image), "path": None},
                "prompt": sample.prompt,
                "reference": sample.reference or "",
                "metadata": json.dumps(sample.metadata or {}, ensure_ascii=False),
            }
        )
    return rows


def samples_to_training_records(samples: Iterable[ChandraSample]) -> list[dict[str, Any]]:
    return [sample.training_record() for sample in samples]


class LazyArrowTrainingDataset:
    """Wraps a HuggingFace Dataset (Arrow) so images are decoded lazily,
    one at a time, instead of loading all into RAM at once.

    The SFTTrainer only needs ``__len__`` and ``__getitem__`` — this class
    provides both while keeping the Arrow files memory-mapped (near-zero RAM
    overhead during dataset loading).
    """

    def __init__(self, hf_dataset: Any, *, prompt_fallback: str = OCR_PROMPT):
        self._ds = hf_dataset
        self._prompt_fallback = prompt_fallback

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self._ds[idx]  # decodes only THIS one image from Arrow
        sample = normalize_sample(record, prompt_fallback=self._prompt_fallback)
        return sample.training_record()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def load_lazy_training_dataset(
    path: str | Path,
    *,
    prompt_fallback: str = OCR_PROMPT,
    max_samples: int | None = None,
) -> LazyArrowTrainingDataset | None:
    """Load an Arrow dataset directory as a lazy training dataset.

    Returns ``None`` if *path* is not a directory (i.e. it is pkl/json),
    so callers can fall back to the eager ``load_chandra_dataset`` path.
    """
    dataset_path = Path(path)
    if not dataset_path.is_dir():
        return None

    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "Loading Arrow datasets requires the 'datasets' package."
        ) from exc

    hf_dataset = load_from_disk(str(dataset_path))
    if max_samples is not None:
        hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))
    return LazyArrowTrainingDataset(hf_dataset, prompt_fallback=prompt_fallback)


def build_image_samples(
    image_paths: Iterable[str | Path],
    *,
    prompt: str = OCR_PROMPT,
) -> list[ChandraSample]:
    samples: list[ChandraSample] = []
    for image_path in image_paths:
        path = Path(image_path)
        image = load_image(path)
        messages = make_messages(image=image, prompt=prompt, reference=None)
        samples.append(
            ChandraSample(
                messages=messages,
                image=image,
                prompt=prompt,
                reference=None,
                metadata={"image_path": str(path)},
            )
        )
    return samples


def build_pdf_samples(
    pdf_path: str | Path,
    *,
    prompt: str = OCR_PROMPT,
    dpi: int = 300,
    page_range: str | None = None,
    references: dict[int, str] | None = None,
) -> list[ChandraSample]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for --pdf inputs. Install it with: pip install pymupdf") from exc

    path = Path(pdf_path)
    doc = fitz.open(path)
    try:
        pages = parse_page_range(page_range, page_count=doc.page_count)
        samples: list[ChandraSample] = []
        for page_number in pages:
            page_index = page_number - 1
            page = doc[page_index]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            reference = references.get(page_number) if references else None
            messages = make_messages(image=image, prompt=prompt, reference=reference)
            samples.append(
                ChandraSample(
                    messages=messages,
                    image=image,
                    prompt=prompt,
                    reference=reference,
                    metadata={"pdf_path": str(path), "page_number": page_number, "dpi": dpi},
                )
            )
        return samples
    finally:
        doc.close()


def load_reference_map(path: str | Path, *, text_field: str = "markdown") -> dict[int, str]:
    """Load page-numbered references from the existing annotation JSON shape."""

    records = _load_records(Path(path))
    if isinstance(records, dict):
        for key in ("data", "pages", "records"):
            if key in records and isinstance(records[key], list):
                records = records[key]
                break

    if not isinstance(records, list):
        raise ValueError("Reference JSON must contain a list of page records.")

    result: dict[int, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        page = record.get("page") or record.get("page_number")
        text = record.get(text_field)
        if page is not None and text is not None:
            result[int(page)] = str(text)
    return result


def parse_page_range(page_range: str | None, *, page_count: int) -> list[int]:
    if not page_range:
        return list(range(1, page_count + 1))

    pages: set[int] = set()
    for part in page_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid page range: {part}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))

    invalid = [page for page in pages if page < 1 or page > page_count]
    if invalid:
        raise ValueError(f"Pages out of range 1-{page_count}: {invalid}")
    return sorted(pages)


def make_messages(
    *,
    image: Image.Image,
    prompt: str,
    reference: str | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        }
    ]
    if reference is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": reference}],
            }
        )
    return messages


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_records(path: Path) -> Any:
    if path.is_dir():
        return _load_arrow_records(path)

    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    raise ValueError(
        f"Unsupported dataset path for {path}. Use .pkl, .pickle, .json, .jsonl, "
        "or a Hugging Face Dataset directory saved with save_to_disk()."
    )


def _load_arrow_records(path: Path) -> list[dict[str, Any]]:
    # WARNING: This eagerly decodes ALL images into RAM.  For large Arrow
    # datasets (hundreds of 600-DPI images) this can consume 100+ GB and
    # trigger the Linux OOM killer.  Prefer ``load_lazy_training_dataset``
    # for training to avoid this.
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "Loading Arrow datasets requires the 'datasets' package. "
            "Install dependencies from README_CHANDRA_FINETUNE.md."
        ) from exc

    dataset = load_from_disk(str(path))
    warnings.warn(
        f"Eagerly loading {len(dataset)} Arrow records into RAM.  "
        "Use load_lazy_training_dataset() for large datasets to avoid OOM.",
        stacklevel=2,
    )
    return [dict(record) for record in dataset]


def _normalize_messages(
    messages: list[dict[str, Any]],
    *,
    image: Image.Image,
    prompt_fallback: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    saw_user_image = False
    saw_user_text = False

    for message in messages:
        role = message.get("role")
        content = message.get("content", [])
        if not isinstance(content, list):
            content = [{"type": "text", "text": str(content)}]

        normalized_content: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image":
                if role == "user":
                    normalized_content.append({"type": "image", "image": image})
                    saw_user_image = True
            elif item_type == "text":
                text = str(item.get("text", ""))
                normalized_content.append({"type": "text", "text": text})
                if role == "user" and text:
                    saw_user_text = True
            elif "text" in item:
                text = str(item.get("text", ""))
                normalized_content.append({"type": "text", "text": text})
                if role == "user" and text:
                    saw_user_text = True

        if role == "user":
            if not saw_user_text:
                normalized_content.insert(0, {"type": "text", "text": prompt_fallback})
                saw_user_text = True
            if not saw_user_image:
                normalized_content.append({"type": "image", "image": image})
                saw_user_image = True

        normalized.append({"role": role, "content": normalized_content})

    if not any(message.get("role") == "user" for message in normalized):
        normalized.insert(
            0,
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_fallback},
                    {"type": "image", "image": image},
                ],
            },
        )
    return normalized


def _find_image(
    sample: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    base_dir: str | Path | None,
) -> Image.Image:
    for image in _iter_image_candidates(sample, messages):
        return _coerce_image(image, base_dir=base_dir)
    raise ValueError("Sample does not contain an image or image_path.")


def _coerce_image(image: Any, *, base_dir: str | Path | None) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return load_image(_resolve_path(image, base_dir=base_dir))
    if isinstance(image, dict):
        if image.get("path"):
            return load_image(_resolve_path(image["path"], base_dir=base_dir))
        if image.get("bytes"):
            import io

            with Image.open(io.BytesIO(image["bytes"])) as pil_image:
                return pil_image.convert("RGB")
    raise ValueError(f"Unsupported image payload type: {type(image).__name__}")


def _iter_image_candidates(sample: dict[str, Any], messages: list[dict[str, Any]]) -> Iterable[Any]:
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image":
                if "image" in item:
                    yield item["image"]
                if "image_path" in item:
                    yield item["image_path"]

    metadata = sample.get("metadata")
    if isinstance(metadata, dict) and metadata.get("image_path"):
        yield metadata["image_path"]
    if sample.get("image_path"):
        yield sample["image_path"]
    if sample.get("image"):
        yield sample["image"]


def _resolve_path(path: str | Path, *, base_dir: str | Path | None) -> Path:
    resolved = Path(path)
    if resolved.is_absolute() or base_dir is None:
        return resolved
    candidate = Path(base_dir) / resolved
    if candidate.exists():
        return candidate
    return resolved


def _extract_user_prompt(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        if message.get("role") != "user":
            continue
        for item in message.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                return str(item["text"])
    return None


def _extract_assistant_text(messages: list[dict[str, Any]]) -> str | None:
    texts: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            return str(content)
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
    return "\n".join(texts) if texts else None


def _parse_metadata(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)
    if isinstance(metadata, str):
        if not metadata:
            return {}
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return {"raw_metadata": metadata}
        return dict(parsed) if isinstance(parsed, dict) else {"metadata": parsed}
    return {"metadata": metadata}


def _image_to_png_bytes(image: Image.Image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()
