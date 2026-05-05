#!/usr/bin/env python
"""Recompute CER/WER/TEDS/table_TEDS on existing predictions after cleaning HTML.

Usage:
    python recompute_metrics.py \
        --input /mnt/disk/ml_data/prerna/temp_files/infer_predictions.json \
        --output /mnt/disk/ml_data/prerna/temp_files/infer_predictions_recomputed.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import os
from pathlib import Path
from statistics import mean

# ---------------------------------------------------------------------------
# clean_html (same logic as chandra_finetune.metrics.clean_html)
# ---------------------------------------------------------------------------

def clean_html(html: str) -> str:
    """Clean raw HTML: collapse whitespace, strip inter-tag spaces, trim cell padding."""
    html = html.replace("\n", "")
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r">\s+<", "><", html)
    html = re.sub(r"(<(?:td|th|p|div|li|caption)[^>]*>)\s+", r"\1", html)
    html = re.sub(r"\s+(</(?:td|th|p|div|li|caption)>)", r"\1", html)
    return html.strip()


# ---------------------------------------------------------------------------
# Import metric functions from chandra_finetune.metrics
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from chandra_finetune.metrics import (
    character_error_rate,
    word_error_rate,
    teds_score,
    table_teds_score,
)


def recompute(entry: dict) -> dict:
    """Clean prediction/reference and recompute all four metrics."""
    pred = clean_html(entry.get("prediction", ""))
    ref = clean_html(entry.get("reference", ""))

    cer = character_error_rate(pred, ref)
    wer = word_error_rate(pred, ref)
    teds = teds_score(pred, ref)
    t_teds = table_teds_score(pred, ref)

    return {
        **entry,
        "prediction": pred,
        "reference": ref,
        "metrics": {
            "cer": round(cer, 6),
            "wer": round(wer, 6),
            "teds": round(teds, 6),
            "table_teds": round(t_teds, 6) if t_teds is not None else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Recompute metrics on existing predictions after HTML cleaning.")
    parser.add_argument("--input", required=True, help="Path to predictions JSON")
    parser.add_argument("--output", default=None, help="Output path (default: <input>_recomputed.json)")
    args = parser.parse_args()

    output = args.output
    if output is None:
        p = Path(args.input)
        output = str(p.parent / f"{p.stem}_recomputed{p.suffix}")

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]

    print(f"Processing {len(data)} entries...")
    results = []
    for i, entry in enumerate(data, 1):
        r = recompute(entry)
        results.append(r)
        m = r["metrics"]
        print(f"  [{i}/{len(data)}] CER={m['cer']:.4f}  WER={m['wer']:.4f}  TEDS={m['teds']:.4f}  table_TEDS={m['table_teds']:.4f}" if m['table_teds'] is not None else f"  [{i}/{len(data)}] CER={m['cer']:.4f}  WER={m['wer']:.4f}  TEDS={m['teds']:.4f}  table_TEDS=n/a")

    # Aggregate
    n = len(results)
    avg_cer = mean(r["metrics"]["cer"] for r in results)
    avg_wer = mean(r["metrics"]["wer"] for r in results)
    avg_teds = mean(r["metrics"]["teds"] for r in results)
    t_teds_vals = [r["metrics"]["table_teds"] for r in results if r["metrics"]["table_teds"] is not None]
    avg_table_teds = mean(t_teds_vals) if t_teds_vals else None

    print(f"\n{'='*50}")
    print(f"Average metrics on {n} samples (cleaned text):")
    print(f"  CER:        {avg_cer:.6f}")
    print(f"  WER:        {avg_wer:.6f}")
    print(f"  TEDS:       {avg_teds:.6f}")
    print(f"  Table TEDS: {avg_table_teds:.6f}" if avg_table_teds is not None else "  Table TEDS: n/a")
    print(f"{'='*50}")

    Path(output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
