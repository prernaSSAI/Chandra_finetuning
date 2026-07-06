#!/usr/bin/env python

from __future__ import annotations

import base64
import io
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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

from augmentation import apply_noise, NOISE_PARAMS, NOISE_NAMES


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
    max_new_tokens: int = 4096,
    request_timeout: int = 900,
    retry_attempts: int = 5,
    retry_delay: float = 5.0,
) -> str:
    """Send one image+prompt to the vLLM server and 2048return the generated text."""
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
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """All inference settings — edit these values directly in code.

    These were previously command-line flags. They now live here so a run is
    fully reproducible from this file and there are no CLI args to remember.
    To start a run, edit the values below and run:  python inf_vllm.py

    Provide exactly ONE input source: ``dataset``, ``image``, ``pdf``, or
    ``pkl_dir`` (the latter also needs ``manifest`` + ``split``).
    """

    # ── Input sources (provide exactly one) ─────────────────────────────────
    # Path to dataset artifact: Arrow dir, .pkl, .json, or .jsonl.
    dataset: str | None = None
    # Image path(s). Add one or more entries, e.g. ["a.png", "b.png"].
    image: list[str] = field(default_factory=list)
    pdf: str | None = "/mnt/disk/ml_data/prerna/data/AH250020.pdf"  # PDF path to render and process
    references_json: str | None = None           # optional page reference JSON for pdf inputs
    reference_field: str = "markdown"            # reference text field in references_json
    page_range: str | None = None                # PDF pages, e.g. "1-5,7,9"
    dpi: int = 600                               # PDF render DPI
    prompt_type: str = "ocr"                     # prompt for image/PDF inputs (key in PROMPT_MAPPING)
    override_prompt: str | None = None           # force this prompt for every sample

    pkl_dir: str | None = None                   # directory with source .pkl files
    manifest: str | None = None                  # split manifest JSON
    split: str | None = None                     # "train" | "valid" | "test" (with pkl_dir)

    # ── vLLM client ─────────────────────────────────────────────────────────
    # Base URL of the vLLM OpenAI-compatible server.
    vllm_url: str = "http://localhost:8000"
    # Model name as registered in vLLM --lora-modules.
    vllm_model: str = "chandra_lora"
    no_wait: bool = False                        # skip startup health-check
    wait_timeout: int = 900                     # seconds to wait for the server to become ready
    request_timeout: int = 900                   # per-request HTTP timeout in seconds
    retry_attempts: int = 5                      # retries per sample on transient network errors
    retry_delay: float = 5.0                     # seconds between retries
    # Pages to process in parallel (1 = sequential). Set 4-8 for concurrent
    # inference with vLLM continuous batching.
    concurrency: int = 25

    # ── Model (base checkpoint, for reference/metadata only) ────────────────
    model_name: str = DEFAULT_MODEL_NAME

    # ── Generation / output ─────────────────────────────────────────────────
    max_samples: int | None = None               # cap number of samples (debugging)
    output: str = "/mnt/disk/ml_data/prerna/iter3_recover_epoch8/complex/AH250020.json"  # JSON output path; must end with .json
    metrics: str = "cer,wer,teds,table_teds"     # comma-separated metric names
    max_new_tokens: int = 4096


# ---------------------------------------------------------------------------
# Single-page inference worker (used by both sequential and concurrent modes)
# ---------------------------------------------------------------------------

def _process_one_page(
    *,
    index: int,
    sample: ChandraSample,
    args: InferenceConfig,
    metric_names: list[str],
) -> dict[str, Any]:
    """Run inference + metrics for a single page. Returns the result row dict.

    On failure, returns a row with an 'error' key instead of raising.
    """
    prompt = args.override_prompt or sample.prompt
    reference_raw = sample.reference
    reference = clean_html(reference_raw) if reference_raw else reference_raw

    # Clean/enhance the page image (CLAHE + gamma + brightness + contrast +
    # sharpen) before OCR. .copy() so the original sample is left untouched.
    cleaned_img = apply_noise(
        sample.image.copy(),
        NOISE_NAMES,
        noise_params=NOISE_PARAMS,
    )

    _t0 = time.perf_counter()
    try:
        prediction_raw = generate_text_vllm(
            vllm_url=args.vllm_url,
            model=args.vllm_model,
            image=cleaned_img,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            request_timeout=args.request_timeout,
            retry_attempts=args.retry_attempts,
            retry_delay=args.retry_delay,
        )
    except Exception as exc:
        gen_seconds = time.perf_counter() - _t0
        return {
            "index": index,
            "metadata": sample.metadata or {},
            "prompt": prompt,
            "reference": reference,
            "reference_raw": reference_raw,
            "prediction": "",
            "prediction_raw": "",
            "metrics": {},
            "gen_seconds": round(gen_seconds, 3),
            "error": str(exc),
        }

    gen_seconds = time.perf_counter() - _t0
    prediction = clean_html(prediction_raw)
    metrics = compute_metrics(prediction, reference, metric_names=metric_names)
    # NOTE: metrics are computed on the CLEANED text only (fair comparison).
    # The *_raw fields hold the unprocessed model output / ground truth, saved
    # purely for debugging + visualization (e.g. to tell whether a dropped
    # element was removed by clean_html or never produced by the model).
    return {
        "index": index,
        "metadata": sample.metadata or {},
        "prompt": prompt,
        "reference": reference,
        "reference_raw": reference_raw,
        "prediction": prediction,
        "prediction_raw": prediction_raw,
        "metrics": metrics,
        "gen_seconds": round(gen_seconds, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = InferenceConfig()

    metric_names = parse_metric_names(args.metrics)
    samples = _load_samples(args)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No inference samples were selected.")

    # Health-check: make sure the server is up and the model is registered.
    if not args.no_wait:
        wait_for_vllm(args.vllm_url, args.vllm_model, timeout=args.wait_timeout)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: reload any previously-saved predictions so a re-run skips pages
    # that already succeeded (failed pages, marked with "error", are retried).
    rows: list[dict[str, Any]] = []
    done_indices: set[int] = set()
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            for r in existing:
                if isinstance(r, dict) and r.get("index") is not None and not r.get("error"):
                    rows.append(r)
                    done_indices.add(int(r["index"]))
            if done_indices:
                print(f"[resume] {len(done_indices)} completed pages found in {output_path}; skipping them.")
        except Exception as exc:
            print(f"[resume] Could not read existing {output_path} ({exc}); starting fresh.")
            rows, done_indices = [], set()

    # Build the list of (index, sample) pairs that still need processing.
    pending = [
        (index, sample)
        for index, sample in enumerate(samples, start=1)
        if index not in done_indices
    ]

    wall_start = time.perf_counter()
    if not pending:
        print("All pages already completed. Nothing to do.")
    else:
        concurrency = max(1, args.concurrency)
        print(f"Processing {len(pending)} pages with concurrency={concurrency} …")

        if concurrency == 1:
            # --- Sequential mode (original behaviour) ---
            _run_sequential(pending, args, metric_names, rows, output_path, len(samples))
        else:
            # --- Concurrent mode ---
            _run_concurrent(pending, args, metric_names, rows, output_path, len(samples), concurrency)
    wall_elapsed = time.perf_counter() - wall_start

    aggregate = aggregate_metrics(rows)
    failed = sum(1 for r in rows if r.get("error"))
    generated = len(rows) - failed
    total_gen_seconds = sum(r.get("gen_seconds", 0) for r in rows if not r.get("error"))

    print(f"\nWrote {len(rows)} predictions to {output_path}")
    if failed:
        print(f"  WARNING: {failed} page(s) failed (saved with empty prediction + 'error'). Re-run to retry just those.")
    print("Aggregate metrics:")
    for name in metric_names:
        value = aggregate.get(name)
        printable = "n/a" if value is None else f"{value:.6f}"
        print(f"  {name}: {printable}")

    print("Timing:")
    print(f"  pages generated: {generated}")
    print(f"  wall-clock time: {wall_elapsed:.1f}s ({wall_elapsed / 60:.2f} min)")
    print(f"  sum of gen_seconds: {total_gen_seconds:.1f}s ({total_gen_seconds / 60:.2f} min)")
    print(f"  avg per page (wall): {wall_elapsed / generated:.2f}s" if generated else "  avg per page: n/a")


def _run_sequential(
    pending: list[tuple[int, ChandraSample]],
    args: InferenceConfig,
    metric_names: list[str],
    rows: list[dict[str, Any]],
    output_path: Path,
    total_samples: int,
) -> None:
    """Process pages one at a time (original behaviour)."""
    for index, sample in pending:
        row = _process_one_page(
            index=index, sample=sample, args=args, metric_names=metric_names,
        )
        rows.append(row)
        rows.sort(key=lambda r: r.get("index", 0))
        write_predictions(output_path, rows)

        if row.get("error"):
            print(f"[{index}/{total_samples}] FAILED: {row['error']} ({row['gen_seconds']:.1f}s) — continuing")
        else:
            print(f"{_format_progress(row, total=total_samples)} ({row['gen_seconds']:.2f}s)")


def _run_concurrent(
    pending: list[tuple[int, ChandraSample]],
    args: InferenceConfig,
    metric_names: list[str],
    rows: list[dict[str, Any]],
    output_path: Path,
    total_samples: int,
    concurrency: int,
) -> None:
    """Process pages in parallel using a thread pool.

    vLLM handles concurrent requests via continuous batching on the GPU side.
    We use threads (not processes) because the work is I/O-bound (HTTP calls).
    """
    save_lock = threading.Lock()
    completed = 0

    def _on_result(row: dict[str, Any]) -> None:
        nonlocal completed
        with save_lock:
            rows.append(row)
            rows.sort(key=lambda r: r.get("index", 0))
            write_predictions(output_path, rows)
            completed += 1
            if row.get("error"):
                print(f"[{row['index']}/{total_samples}] (done {completed}/{len(pending)}) "
                      f"FAILED: {row['error']} ({row['gen_seconds']:.1f}s)")
            else:
                print(f"[{row['index']}/{total_samples}] (done {completed}/{len(pending)}) "
                      f"{_format_metrics(row)} ({row['gen_seconds']:.2f}s)")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _process_one_page,
                index=index,
                sample=sample,
                args=args,
                metric_names=metric_names,
            ): index
            for index, sample in pending
        }
        for future in as_completed(futures):
            row = future.result()
            _on_result(row)


def _format_metrics(row: dict[str, Any]) -> str:
    """Format metrics from a row dict into a compact string."""
    metrics = row.get("metrics") or {}
    parts = []
    for name, value in metrics.items():
        formatted = "n/a" if value is None else f"{value:.4f}"
        parts.append(f"{name}={formatted}")
    return ", ".join(parts) or "metrics=n/a"


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

def _load_samples(args: InferenceConfig) -> list[ChandraSample]:
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
