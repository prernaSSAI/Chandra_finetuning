#!/usr/bin/env python

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from chandra_finetune import DEFAULT_MODEL_NAME
from chandra_finetune.data import (
    ChandraSample,
    build_image_samples,
    build_pdf_samples,
    load_chandra_dataset,
    load_reference_map,
)
from chandra_finetune.metrics import aggregate_metrics, clean_html, compute_metrics, parse_metric_names
from prompts import PROMPT_MAPPING


# ---------------------------------------------------------------------------
# vLLM client helpers
# ---------------------------------------------------------------------------

def _image_to_data_url(image: Any) -> str:
    """Encode a PIL Image as a base64 PNG data-URL."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _build_request_payload(
    *,
    model: str,
    prompt: str,
    image: Any,
    max_tokens: int,
) -> dict[str, Any]:
    """Build the /v1/chat/completions JSON body.

    Temperature is fixed at 0.0. No top-p/top-k/min-p or penalties are sent.
    """
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_url(image)},
                    },
                ],
            }
        ],
    }


def generate_text_vllm(
    *,
    vllm_url: str,
    model: str,
    image: Any,
    prompt: str,
    max_new_tokens: int = 12384,
    request_timeout: int = 300,
    retry_attempts: int = 3,
    retry_delay: float = 5.0,
) -> str:
    """Send one image+prompt to the vLLM server and return the generated text."""
    endpoint = vllm_url.rstrip("/") + "/v1/chat/completions"
    payload = _build_request_payload(
        model=model,
        prompt=prompt,
        image=image,
        max_tokens=max_new_tokens,
    )
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    last_error: Exception | None = None
    for attempt in range(1, retry_attempts + 1):
        try:
            req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
            choices = response_data.get("choices", [])
            if not choices:
                raise ValueError(f"vLLM returned no choices: {response_data}")
            return choices[0]["message"]["content"].strip()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < retry_attempts:
                print(f"  [vLLM] Attempt {attempt}/{retry_attempts} failed ({exc}); "
                      f"retrying in {retry_delay}s …")
                time.sleep(retry_delay)
        except Exception as exc:
            # Non-retryable (e.g. JSON decode error, bad response shape).
            raise RuntimeError(f"vLLM request failed: {exc}") from exc

    raise RuntimeError(
        f"vLLM request failed after {retry_attempts} attempts. "
        f"Last error: {last_error}"
    )


def wait_for_vllm(vllm_url: str, model: str, timeout: int = 120) -> None:
    """Block until the vLLM /v1/models endpoint lists the expected model,
    or raise if it doesn't become ready within *timeout* seconds."""
    endpoint = vllm_url.rstrip("/") + "/v1/models"
    deadline = time.monotonic() + timeout
    print(f"Waiting for vLLM server at {vllm_url} (model={model}) …", flush=True)
    while True:
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            ids = [m.get("id") for m in data.get("data", [])]
            if model in ids:
                print(f"  vLLM server ready. Available models: {ids}")
                return
            # Server is up but LoRA model not yet listed — keep polling.
            print(f"  Server up but model '{model}' not ready yet. "
                  f"Available: {ids}", flush=True)
        except Exception:
            pass  # Server not up yet.

        if time.monotonic() > deadline:
            raise RuntimeError(
                f"vLLM server did not become ready within {timeout}s. "
                "Is it running? Check --vllm-url."
            )
        time.sleep(5)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Chandra inference via vLLM and compute OCR metrics."
    )
    # --- Input sources (unchanged from original) ---
    parser.add_argument("--dataset", help="Path to dataset artifact: Arrow dir, .pkl, .json, or .jsonl.")
    parser.add_argument("--image", action="append", default=[], help="Image path. Can be repeated.")
    parser.add_argument("--pdf", help="PDF path to render and process.")
    parser.add_argument("--references-json", help="Optional page reference JSON for --pdf inputs.")
    parser.add_argument("--reference-field", default="markdown",
                        help="Reference text field in --references-json.")
    parser.add_argument("--page-range", help='PDF pages, e.g. "1-5,7,9".')
    parser.add_argument("--dpi", type=int, default=600, help="PDF render DPI.")
    parser.add_argument("--prompt-type", default="ocr", choices=sorted(PROMPT_MAPPING),
                        help="Prompt for image/PDF inputs.")
    parser.add_argument("--override-prompt", help="Force this prompt for every sample.")

    parser.add_argument("--pkl-dir", help="Directory with source .pkl files.")
    parser.add_argument("--manifest", help="Split manifest JSON.")
    parser.add_argument("--split", choices=["train", "valid", "test"],
                        help="Split to use with --pkl-dir.")

    # --- vLLM client ---
    parser.add_argument("--vllm-url", default="http://localhost:8000",
                        help="Base URL of the vLLM OpenAI-compatible server "
                             "(default: http://localhost:8000).")
    parser.add_argument("--vllm-model", default="chandra_lora",
                        help="Model name as registered in vLLM --lora-modules "
                             "(default: chandra_lora).")
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip the startup health-check and go straight to inference.")
    parser.add_argument("--wait-timeout", type=int, default=120,
                        help="Seconds to wait for the vLLM server to become ready (default: 120).")
    parser.add_argument("--request-timeout", type=int, default=300,
                        help="Per-request HTTP timeout in seconds (default: 300).")
    parser.add_argument("--retry-attempts", type=int, default=3,
                        help="Retry count per sample on transient network errors (default: 3).")
    parser.add_argument("--retry-delay", type=float, default=5.0,
                        help="Seconds between retries (default: 5).")

    # --- Legacy flags (accepted but ignored so existing scripts don't break) ---
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help="[IGNORED] Legacy: base model checkpoint.")
    parser.add_argument("--adapter",
                        help="[IGNORED] Legacy: LoRA adapter path (now set server-side).")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False,
                        help="[IGNORED] Legacy: 4-bit quantisation flag.")
    parser.add_argument("--device", default="auto",
                        help="[IGNORED] Legacy: device selection.")

    # --- Generation settings ---
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default="predictions.json",
                        help="JSON output path. Must end with .json.")
    parser.add_argument("--metrics", default="cer,wer,teds,table_teds")
    parser.add_argument("--max-new-tokens", type=int, default=12384)
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    # Warn about ignored legacy flags.
    if args.adapter:
        print(
            f"[INFO] --adapter={args.adapter!r} is ignored when using vLLM. "
            "The LoRA adapter is loaded by the server via --lora-modules."
        )
    if args.load_in_4bit:
        print("[INFO] --load-in-4bit is ignored when using vLLM.")

    metric_names = parse_metric_names(args.metrics)
    samples = _load_samples(args)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No inference samples were selected.")

    # Health-check: make sure the server is up and the model is registered.
    if not args.no_wait:
        wait_for_vllm(args.vllm_url, args.vllm_model, timeout=args.wait_timeout)

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        prompt = args.override_prompt or sample.prompt

        # ------------------------------------------------------------------ #
        # REPLACEMENT: vLLM API call instead of generate_text(model, …)       #
        # ------------------------------------------------------------------ #
        prediction_raw = generate_text_vllm(
            vllm_url=args.vllm_url,
            model=args.vllm_model,
            image=sample.image,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            request_timeout=args.request_timeout,
            retry_attempts=args.retry_attempts,
            retry_delay=args.retry_delay,
        )
        # ------------------------------------------------------------------ #

        # Everything below is identical to the original infer_chandra.py.
        prediction = clean_html(prediction_raw)
        reference_raw = sample.reference
        reference = clean_html(reference_raw) if reference_raw else reference_raw
        metrics = compute_metrics(prediction, reference, metric_names=metric_names)
        row = {
            "index": index,
            "metadata": sample.metadata or {},
            "prompt": prompt,
            "reference": reference,
            "prediction": prediction,
            "metrics": metrics,
        }
        rows.append(row)
        print(_format_progress(row, total=len(samples)))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(output_path, rows)

    aggregate = aggregate_metrics(rows)
    print(f"Wrote {len(rows)} predictions to {output_path}")
    print("Aggregate metrics:")
    for name in metric_names:
        value = aggregate.get(name)
        printable = "n/a" if value is None else f"{value:.6f}"
        print(f"  {name}: {printable}")


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.suffix.lower() != ".json":
        raise ValueError("Output must end with .json because this script writes JSON only.")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Sample loading (unchanged from original infer_chandra.py)
# ---------------------------------------------------------------------------

def _load_samples(args: argparse.Namespace) -> list[ChandraSample]:
    sources = (
        int(bool(args.dataset))
        + int(bool(args.image))
        + int(bool(args.pdf))
        + int(bool(args.pkl_dir))
    )
    if sources != 1:
        raise ValueError(
            "Provide exactly one input source: --dataset, --image, --pdf, or --pkl-dir."
        )

    if args.pkl_dir:
        if not args.manifest or not args.split:
            raise ValueError("--pkl-dir requires both --manifest and --split.")
        from split import build_split

        with Path(args.manifest).open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        split_key = f"{args.split}_pages"
        if split_key not in manifest:
            raise ValueError(f"Manifest does not contain {split_key!r}.")
        return build_split(manifest[split_key], Path(args.pkl_dir))

    if args.dataset:
        return load_chandra_dataset(args.dataset)

    prompt = PROMPT_MAPPING[args.prompt_type]
    if args.image:
        return build_image_samples(args.image, prompt=prompt)

    references = None
    if args.references_json:
        references = load_reference_map(args.references_json, text_field=args.reference_field)
    return build_pdf_samples(
        args.pdf,
        prompt=prompt,
        dpi=args.dpi,
        page_range=args.page_range,
        references=references,
    )


# ---------------------------------------------------------------------------
# Formatting helper (unchanged from original infer_chandra.py)
# ---------------------------------------------------------------------------

def _format_progress(row: dict[str, Any], *, total: int) -> str:
    metrics = row.get("metrics") or {}
    parts = []
    for name, value in metrics.items():
        formatted = "n/a" if value is None else f"{value:.4f}"
        parts.append(f"{name}={formatted}")
    metric_text = ", ".join(parts)
    return f"[{row['index']}/{total}] {metric_text or 'metrics=n/a'}"


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
