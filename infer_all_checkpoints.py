#!/usr/bin/env python
"""Run inference over EVERY LoRA checkpoint in a folder, one by one.

This is a thin driver around the building blocks in ``inf_vllm.py`` — it does
NOT modify that file. The vLLM server is started once with every checkpoint
registered as a separate ``--lora-modules`` entry (model name == folder name);
this script then loops over those names, runs the full dataset through each,
writes ``pred_<checkpoint>.json`` per checkpoint, and accumulates all aggregate
metrics into ``metrics_summary.json``.

Usage
-----
1. Print the Terminal-1 server command (copy/paste it, leave it running):

       venv/bin/python infer_all_checkpoints.py --print-server-cmd

2. Once the server is up, run the loop (Terminal 2 / tmux):

       venv/bin/python infer_all_checkpoints.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from inf_vllm import (
    InferenceConfig,
    _load_samples,
    _run_concurrent,
    wait_for_vllm,
    write_predictions,
)
from chandra_finetune.metrics import aggregate_metrics, parse_metric_names

# ── Defaults (edit here or override via CLI) ────────────────────────────────
ITER_DIR = Path("/mnt/disk/ml_data/prerna/iter_5")
DATASET = "/mnt/disk/ml_data/prerna/final_arrow/test_36"
METRICS = "cer,wer,teds,table_teds"
MAX_NEW_TOKENS = 4096
CONCURRENCY = 25
VLLM_URL = "http://localhost:8000"
MAX_LORA_RANK = 16
MAX_MODEL_LEN = 24576
GPU = "1"
PORT = 8000
BASE_MODEL = "datalab-to/chandra"


def find_checkpoints(folder: Path) -> list[Path]:
    """Return checkpoint dirs (those with adapter_config.json), numbered ones
    first in step order, then any named dirs (e.g. best/last)."""
    cps = [
        p for p in folder.iterdir()
        if p.is_dir() and (p / "adapter_config.json").exists()
    ]

    def sort_key(p: Path):
        name = p.name
        if name.startswith("checkpoint-"):
            try:
                return (0, int(name.split("-", 1)[1]), name)
            except ValueError:
                return (1, 0, name)
        return (2, 0, name)  # best / last at the end

    return sorted(cps, key=sort_key)


def build_server_cmd(checkpoints: list[Path]) -> str:
    lora_modules = " \\\n      ".join(
        f"{cp.name}={cp}" for cp in checkpoints
    )
    return (
        f"CUDA_VISIBLE_DEVICES={GPU} venv/bin/vllm serve {BASE_MODEL} \\\n"
        f"  --enable-lora \\\n"
        f"  --max-loras 1 --max-lora-rank {MAX_LORA_RANK} "
        f"--max-model-len {MAX_MODEL_LEN} \\\n"
        f"  --trust-remote-code --port {PORT} \\\n"
        f"  --lora-modules \\\n      {lora_modules}"
    )


def _load_existing(out_path: Path) -> tuple[list[dict[str, Any]], set[int]]:
    """Resume support: reload previously-saved, error-free rows."""
    rows: list[dict[str, Any]] = []
    done: set[int] = set()
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as fh:
                for r in json.load(fh):
                    if isinstance(r, dict) and r.get("index") is not None and not r.get("error"):
                        rows.append(r)
                        done.add(int(r["index"]))
        except Exception as exc:
            print(f"  [resume] could not read {out_path} ({exc}); starting fresh.")
            return [], set()
    return rows, done


def run(args) -> None:
    iter_dir = Path(args.iter_dir)
    checkpoints = find_checkpoints(iter_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints with adapter_config.json found in {iter_dir}")

    if args.print_server_cmd:
        print("# Terminal 1 — start the server ONCE with all checkpoints loaded:\n")
        print(f"cd {Path.cwd()}")
        print(build_server_cmd(checkpoints))
        return

    metric_names = parse_metric_names(args.metrics)

    # Load the dataset ONCE; identical across every checkpoint.
    # Explicitly clear image/pdf/pkl_dir so we don't inherit whatever
    # single-input defaults inf_vllm.py currently has.
    probe_cfg = InferenceConfig(
        dataset=args.dataset, image=[], pdf=None, pkl_dir=None,
        metrics=args.metrics,
        max_new_tokens=args.max_new_tokens, concurrency=args.concurrency,
        vllm_url=args.vllm_url,
    )
    samples = _load_samples(probe_cfg)
    print(f"Loaded {len(samples)} samples from {args.dataset}")
    print(f"Will evaluate {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  - {cp.name}")

    summary_path = iter_dir / "metrics_summary.json"
    summary: list[dict[str, Any]] = []

    for n, cp in enumerate(checkpoints, start=1):
        model_name = cp.name
        out_path = iter_dir / f"pred_{model_name}.json"
        cfg = InferenceConfig(
            dataset=args.dataset, image=[], pdf=None, pkl_dir=None,
            vllm_model=model_name, output=str(out_path),
            metrics=args.metrics, max_new_tokens=args.max_new_tokens,
            concurrency=args.concurrency, vllm_url=args.vllm_url,
        )

        print(f"\n{'='*70}\n[{n}/{len(checkpoints)}] {model_name}\n{'='*70}")
        # Make sure THIS adapter name is registered before we hit it.
        wait_for_vllm(cfg.vllm_url, model_name, timeout=args.wait_timeout)

        rows, done = _load_existing(out_path)
        if done:
            print(f"  [resume] {len(done)} pages already done in {out_path.name}")
        pending = [
            (idx, s) for idx, s in enumerate(samples, start=1) if idx not in done
        ]

        t0 = time.perf_counter()
        if pending:
            _run_concurrent(
                pending, cfg, metric_names, rows, out_path,
                len(samples), max(1, args.concurrency),
            )
        else:
            print("  All pages already done; nothing to run.")
        elapsed = time.perf_counter() - t0

        agg = aggregate_metrics(rows)
        failed = sum(1 for r in rows if r.get("error"))
        entry = {
            "checkpoint": model_name,
            "path": str(cp),
            "predictions_file": str(out_path),
            "num_pages": len(samples),
            "failed_pages": failed,
            "wall_seconds": round(elapsed, 1),
            "metrics": agg,
        }
        summary.append(entry)

        # Persist the summary after every checkpoint so progress survives a crash.
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

        metric_str = ", ".join(
            f"{k}={'n/a' if v is None else f'{v:.4f}'}" for k, v in agg.items()
        )
        print(f"  -> {metric_str}  ({elapsed/60:.1f} min, {failed} failed)")

    print(f"\nDone. Per-checkpoint metrics written to {summary_path}")
    header = "checkpoint".ljust(22) + "  ".join(m.rjust(10) for m in metric_names)
    print(header)
    for e in summary:
        cells = []
        for m in metric_names:
            v = e["metrics"].get(m)
            cells.append(("n/a" if v is None else f"{v:.4f}").rjust(10))
        print(e["checkpoint"].ljust(22) + "  ".join(cells))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iter-dir", default=str(ITER_DIR))
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--metrics", default=METRICS)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--concurrency", type=int, default=CONCURRENCY)
    p.add_argument("--vllm-url", default=VLLM_URL)
    p.add_argument("--wait-timeout", type=int, default=900)
    p.add_argument("--print-server-cmd", action="store_true",
                   help="Print the Terminal-1 vllm serve command and exit.")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
