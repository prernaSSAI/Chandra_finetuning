from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class GenerationSettings:
    max_new_tokens: int = 12384
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float | None = None
    repetition_penalty: float = 1.0
    presence_penalty: float | None = None
    use_cache: bool = True

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0


def generate_text(
    *,
    model: Any,
    tokenizer: Any,
    image: Image.Image,
    prompt: str,
    settings: GenerationSettings | None = None,
    device: str = "auto",
) -> str:
    """Generate OCR HTML/markdown for one image."""

    settings = settings or GenerationSettings()
    _disable_tokenizer_truncation(tokenizer)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        truncation=False,
        return_tensors="pt",
    )

    target_device = _resolve_device(model, device)
    if target_device is not None:
        inputs = inputs.to(target_device)

    generation_kwargs = {
        "max_new_tokens": settings.max_new_tokens,
        "use_cache": settings.use_cache,
        "repetition_penalty": settings.repetition_penalty,
        "do_sample": settings.do_sample,
    }
    if settings.do_sample:
        generation_kwargs["temperature"] = settings.temperature
        generation_kwargs["top_p"] = settings.top_p
        generation_kwargs["top_k"] = settings.top_k
        if settings.min_p is not None:
            generation_kwargs["min_p"] = settings.min_p
    if settings.presence_penalty is not None:
        generation_kwargs["presence_penalty"] = settings.presence_penalty

    try:
        import torch
    except ImportError:
        torch = None

    if torch is None:
        generated_ids = model.generate(**inputs, **generation_kwargs)
    else:
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **generation_kwargs)

    input_length = inputs["input_ids"].shape[-1]
    generated_ids = generated_ids[:, input_length:]
    decoded = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip() if decoded else ""


def _resolve_device(model: Any, device: str) -> str | None:
    if device == "auto":
        try:
            return next(model.parameters()).device
        except Exception:
            return None
    if device == "none":
        return None
    return device


def _disable_tokenizer_truncation(tokenizer: Any) -> None:
    """Prevent saved training-time 2048-token truncation from cutting image tokens."""

    for candidate in (tokenizer, getattr(tokenizer, "tokenizer", None)):
        if candidate is None:
            continue
        try:
            candidate.model_max_length = max(int(getattr(candidate, "model_max_length", 0)), 262144)
        except Exception:
            pass

        backend = getattr(candidate, "_tokenizer", None)
        if backend is None:
            continue
        try:
            backend.no_truncation()
        except Exception:
            pass
