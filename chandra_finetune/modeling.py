from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import DEFAULT_MODEL_NAME


@dataclass
class LoraSettings:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    random_state: int = 3407
    use_rslora: bool = False
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True


def load_model_and_tokenizer(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    load_in_4bit: bool = True,
    use_gradient_checkpointing: str | bool | None = "unsloth",
) -> tuple[Any, Any]:
    FastVisionModel = _fast_vision_model()
    kwargs: dict[str, Any] = {
        "model_name": model_name,
        "load_in_4bit": load_in_4bit,
    }
    if use_gradient_checkpointing is not None:
        kwargs["use_gradient_checkpointing"] = use_gradient_checkpointing
    return FastVisionModel.from_pretrained(**kwargs)


def load_training_model(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    load_in_4bit: bool = True,
    use_gradient_checkpointing: str | bool | None = "unsloth",
    lora: LoraSettings | None = None,
) -> tuple[Any, Any]:
    FastVisionModel = _fast_vision_model()
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    lora = lora or LoraSettings()
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=lora.finetune_vision_layers,
        finetune_language_layers=lora.finetune_language_layers,
        finetune_attention_modules=lora.finetune_attention_modules,
        finetune_mlp_modules=lora.finetune_mlp_modules,
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        bias=lora.bias,
        random_state=lora.random_state,
        use_rslora=lora.use_rslora,
        loftq_config=None,
    )
    return model, tokenizer


def load_inference_model(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    adapter: str | None = None,
    load_in_4bit: bool = True,
) -> tuple[Any, Any]:
    load_name = adapter or model_name
    model, tokenizer = load_model_and_tokenizer(
        model_name=load_name,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing=None,
    )
    set_inference_mode(model)
    return model, tokenizer


def set_training_mode(model: Any) -> None:
    FastVisionModel = _fast_vision_model()
    FastVisionModel.for_training(model)


def set_inference_mode(model: Any) -> None:
    FastVisionModel = _fast_vision_model()
    FastVisionModel.for_inference(model)


def _fast_vision_model() -> Any:
    try:
        from unsloth import FastVisionModel
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth is required for Chandra fine-tuning/inference. "
            "Use a Python 3.10-3.12 CUDA environment and install the packages "
            "listed in README_CHANDRA_FINETUNE.md."
        ) from exc
    return FastVisionModel

