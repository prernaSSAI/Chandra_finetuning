"""
Post-process Chandra evaluation results — clean prediction HTML.

Usage:
    python postprocess_predictions.py --input evaluation_results.json --output cleaned_results.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def clean_prediction(html: str) -> str:
    """Clean a raw prediction HTML string."""

    # 1. Remove literal \n newlines between/around HTML tags
    html = html.replace("\n", "")

    # 2. Collapse multiple whitespace into single space (but not inside <pre>)
    html = re.sub(r"[ \t]+", " ", html)

    # 3. Remove spaces between consecutive tags: > <  →  ><
    html = re.sub(r">\s+<", "><", html)

    # 4. Remove leading/trailing whitespace inside tags:
    #    <td> text </td>  →  <td>text</td>
    html = re.sub(r"(<(?:td|th|p|div|li|caption)[^>]*>)\s+", r"\1", html)
    html = re.sub(r"\s+(</(?:td|th|p|div|li|caption)>)", r"\1", html)

    # 5. Remove empty lines / excessive whitespace that might remain
    html = html.strip()

    return html


def process_file(input_path: str, output_path: str) -> None:
    """Read evaluation JSON, clean prediction/reference HTML, write output."""

    data = json.loads(Path(input_path).read_text(encoding="utf-8"))

    if isinstance(data, dict):
        data = [data]

    html_fields = ("prediction", "reference")
    cleaned_count = 0

    for entry in data:
        for field in html_fields:
            raw = entry.get(field, "")

            if not isinstance(raw, str) or not raw:
                continue

            cleaned = clean_prediction(raw)

            if cleaned != raw:
                cleaned_count += 1

            entry[field] = cleaned

    Path(output_path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Processed {len(data)} entries.")
    print(f"Cleaned {cleaned_count} HTML fields: prediction/reference.")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean prediction HTML in Chandra evaluation results."
    )
    parser.add_argument(
        "--input", required=True, help="Path to evaluation_results.json"
    )
    parser.add_argument(
        "--output", default=None, help="Output path (default: <input>_cleaned.json)"
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        p = Path(args.input)
        output = str(p.parent / f"{p.stem}_cleaned{p.suffix}")

    process_file(args.input, output)


if __name__ == "__main__":
    main()
