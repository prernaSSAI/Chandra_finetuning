#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from chandra_finetune.data import (
    load_chandra_dataset,
    save_samples_to_arrow,
    save_samples_to_pickle,
    split_samples,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a Chandra pickle/json dataset into explicit train/test artifacts."
    )
    parser.add_argument("--input", required=True, help="Source dataset path: .pkl, .pickle, .json, .jsonl, or Arrow dir.")
    parser.add_argument("--output-dir", required=True, help="Directory where split artifacts are written.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Held-out test split ratio.")
    parser.add_argument("--seed", type=int, default=3407, help="Deterministic shuffle seed.")
    parser.add_argument("--format", choices=("arrow", "pickle", "both"), default="arrow")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    _prepare_output_dir(output_dir, overwrite=args.overwrite)

    samples = load_chandra_dataset(args.input)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    train_samples, test_samples = split_samples(samples, eval_ratio=args.test_ratio, seed=args.seed)
    if not train_samples or not test_samples:
        raise ValueError("Both train and test splits must contain at least one sample.")

    if args.format in {"arrow", "both"}:
        save_samples_to_arrow(train_samples, output_dir / "train")
        save_samples_to_arrow(test_samples, output_dir / "test")

    if args.format in {"pickle", "both"}:
        save_samples_to_pickle(train_samples, output_dir / "train.pkl")
        save_samples_to_pickle(test_samples, output_dir / "test.pkl")

    metadata = {
        "input": str(args.input),
        "format": args.format,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "train_path": "train" if args.format in {"arrow", "both"} else "train.pkl",
        "test_path": "test" if args.format in {"arrow", "both"} else "test.pkl",
    }
    with (output_dir / "split_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"Loaded {len(samples)} samples.")
    print(f"Train: {len(train_samples)}")
    print(f"Test : {len(test_samples)}")
    print(f"Saved split artifacts to {output_dir}")


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
