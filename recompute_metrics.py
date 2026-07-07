"""Recompute metrics on an existing prediction JSON after correcting references.

Use case: you already have a final predictions JSON (index/reference/prediction/
metrics/...) but some ground-truth references were misaligned. Fix the `reference`
fields in the JSON, then run this to recompute per-row metrics + the aggregate —
WITHOUT re-running inference. Same metric functions as the inference pipeline are
reused, so scores stay consistent with the original run.

By default the corrected reference is passed through clean_html() (matching the
inference pipeline, since you said GT is still raw/uncleaned). The prediction is
left as-is because it was already cleaned when the JSON was produced.

Example:
    python recompute_metrics.py /mnt/disk/ml_data/prerna/iter_5/pred_checkpoint-1323.json
    # writes  pred_checkpoint-1323.recomputed.json  next to it

    python recompute_metrics.py <input.json> --in-place   # overwrite original
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chandra_finetune.metrics import (
    aggregate_metrics,
    clean_html,
    compute_metrics,
    parse_metric_names,
)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to the predictions JSON (list of rows).")
    parser.add_argument(
        "-o", "--output",
        help="Output path (default: <input>.recomputed.json). Ignored with --in-place.",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite the input file instead of writing a new one.",
    )
    parser.add_argument(
        "--metrics", default="cer,wer,teds,table_teds",
        help="Comma-separated metric names to recompute.",
    )
    parser.add_argument(
        "--no-clean-reference", action="store_true",
        help="Do NOT run clean_html() on references (use them exactly as stored).",
    )
    parser.add_argument(
        "--clean-prediction", action="store_true",
        help="Also run clean_html() on predictions (off by default; they are already cleaned).",
    )
    args = parser.parse_args()

    metric_names = parse_metric_names(args.metrics)
    in_path = Path(args.input)
    rows = json.loads(in_path.read_text())
    if not isinstance(rows, list):
        raise SystemExit(f"Expected a JSON list of rows, got {type(rows).__name__}")

    old_aggregate = aggregate_metrics(rows)

    updated = 0
    for row in rows:
        reference_raw = row.get("reference")
        prediction_raw = row.get("prediction")
        if reference_raw is None or not prediction_raw:
            # No reference or empty/failed prediction -> leave metrics untouched.
            continue

        reference = reference_raw if args.no_clean_reference else clean_html(reference_raw)
        prediction = clean_html(prediction_raw) if args.clean_prediction else prediction_raw

        row["reference"] = reference
        if args.clean_prediction:
            row["prediction"] = prediction
        row["metrics"] = compute_metrics(prediction, reference, metric_names=metric_names)
        updated += 1

    new_aggregate = aggregate_metrics(rows)

    out_path = in_path if args.in_place else Path(args.output or in_path.with_suffix(".recomputed.json"))
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    print(f"Recomputed metrics on {updated}/{len(rows)} rows.")
    print(f"Wrote: {out_path}")
    print(f"\n{'metric':<12}{'old':>10}{'new':>10}")
    print("-" * 32)
    for name in metric_names:
        print(f"{name:<12}{_fmt(old_aggregate.get(name)):>10}{_fmt(new_aggregate.get(name)):>10}")


if __name__ == "__main__":
    main()
