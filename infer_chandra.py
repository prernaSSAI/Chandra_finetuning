#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
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
from chandra_finetune.generation import GenerationSettings, generate_text
from chandra_finetune.metrics import aggregate_metrics, clean_html, compute_metrics, parse_metric_names
from chandra_finetune.modeling import load_inference_model
from prompts import PROMPT_MAPPING


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Chandra inference and optional OCR metrics.")
    parser.add_argument("--dataset", help="Path to dataset artifact: Arrow dir, .pkl, .json, or .jsonl.")
    parser.add_argument("--image", action="append", default=[], help="Image path. Can be repeated.")
    parser.add_argument("--pdf", help="PDF path to render and process.")
    parser.add_argument("--references-json", help="Optional page reference JSON for --pdf inputs.")
    parser.add_argument("--reference-field", default="markdown", help="Reference text field in --references-json.")
    parser.add_argument("--page-range", help='PDF pages, for example "1-5,7,9".')
    parser.add_argument("--dpi", type=int, default=600, help="PDF render DPI.")
    parser.add_argument("--prompt-type", default="ocr", choices=sorted(PROMPT_MAPPING), help="Prompt for image/PDF inputs.")
    parser.add_argument("--override-prompt", help="Force this prompt for every sample.")

    parser.add_argument("--pkl-dir", help="Directory with .pkl files (baseline style).")
    parser.add_argument("--manifest", help="Split manifest JSON.")
    parser.add_argument("--split", choices=["train", "valid", "test"], help="Split to use.")

    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base model checkpoint.")
    parser.add_argument("--adapter", help="LoRA adapter directory or HF id saved by train_chandra.py.")
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load model in 4-bit QLoRA mode. Default is off.",
    )
    parser.add_argument("--device", default="auto", help='Device for tokenizer tensors: auto, cuda, cpu, or "none".')

    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--output", default="predictions.json", help="Output path. Use .csv for CSV or .jsonl for JSON lines.")
    parser.add_argument("--metrics", default="cer,wer,teds,table_teds")
    parser.add_argument("--max-new-tokens", type=int, default=12384)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metric_names = parse_metric_names(args.metrics)
    samples = _load_samples(args)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No inference samples were selected.")

    model, tokenizer = load_inference_model(
        model_name=args.model_name,
        adapter=args.adapter,
        load_in_4bit=args.load_in_4bit,
    )
    settings = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
    )

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        prompt = args.override_prompt or sample.prompt
        prediction_raw = generate_text(
            model=model,
            tokenizer=tokenizer,
            image=sample.image,
            prompt=prompt,
            settings=settings,
            device=args.device,
        )
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
    write_predictions(output_path, rows, metric_names)

    aggregate = aggregate_metrics(rows)
    print(f"Wrote {len(rows)} predictions to {output_path}")
    print("Aggregate metrics:")
    for name in metric_names:
        value = aggregate.get(name)
        printable = "n/a" if value is None else f"{value:.6f}"
        print(f"  {name}: {printable}")


def write_predictions(path: Path, rows: list[dict[str, Any]], metric_names: list[str]) -> None:
    if path.suffix.lower() == ".csv":
        fieldnames = ["index", "metadata", "prompt", "reference", "prediction", *metric_names]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                metrics = row.get("metrics") or {}
                writer.writerow(
                    {
                        "index": row.get("index"),
                        "metadata": json.dumps(row.get("metadata") or {}, ensure_ascii=False),
                        "prompt": row.get("prompt"),
                        "reference": row.get("reference"),
                        "prediction": row.get("prediction"),
                        **{name: metrics.get(name) for name in metric_names},
                    }
                )
        return

    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
        return

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_samples(args: argparse.Namespace) -> list[ChandraSample]:
    sources = int(bool(args.dataset)) + int(bool(args.image)) + int(bool(args.pdf)) + int(bool(args.pkl_dir))
    if sources != 1:
        raise ValueError("Provide exactly one input source: --dataset, --image, --pdf, or --pkl-dir.")

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
