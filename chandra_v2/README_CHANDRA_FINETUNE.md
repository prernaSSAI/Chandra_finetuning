# Chandra Unsloth Fine-Tuning

This directory contains reusable scripts for LoRA fine-tuning and evaluation of
`datalab-to/chandra` using the dataset shape produced by `custom_dataset.py`.

## Environment

The current base interpreter in this workspace is Python 3.13, and it does not
include Unsloth, Transformers, TRL, or Datasets. Use a dedicated CUDA Python
3.10-3.12 environment for actual training/inference.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-unsloth.txt
```

For GPU training, install the PyTorch build that matches your CUDA driver before
installing Unsloth if your platform requires a specific wheel.

## Training

First split the original pickle into explicit train/test artifacts. Arrow is the
default because it stores a typed image column and reloads more reliably than a
large pickle of PIL objects.

```bash
python prepare_chandra_dataset.py \
  --input /path/to/vlm_dataset.pkl \
  --output-dir data/chandra_splits \
  --test-ratio 0.1 \
  --seed 3407 \
  --format arrow
```

This creates:

```text
data/chandra_splits/
  train/
  test/
  split_metadata.json
```

Use `--format pickle` or `--format both` if you also need `train.pkl` and
`test.pkl` for compatibility.

```bash
python train_chandra.py \
  --dataset data/chandra_splits/train \
  --eval-dataset data/chandra_splits/test \
  --model-name datalab-to/chandra \
  --output-dir outputs/chandra_lora \
  --max-steps 30
```

The training script loads Arrow directories, `.pkl`, `.json`, or `.jsonl`
datasets, normalizes records into PIL-backed Unsloth messages, applies LoRA
with the notebook defaults, and saves the adapter plus tokenizer.

Use `--max-steps -1 --num-train-epochs 1` for epoch-based training.

## Inference And Evaluation

```bash
python infer_chandra.py \
  --dataset data/chandra_splits/test \
  --model-name datalab-to/chandra \
  --adapter outputs/chandra_lora \
  --output predictions.jsonl \
  --metrics cer,wer,teds,table_teds
```

Dataset inference runs over every sample in the artifact passed to `--dataset`.
Pass `data/chandra_splits/test` for held-out evaluation.

Image and PDF inputs are also supported:

```bash
python infer_chandra.py --image page.png --output page_prediction.jsonl
python infer_chandra.py --pdf input.pdf --page-range 1-3 --output pdf_predictions.csv
```

If you have page references in the same annotation JSON shape used by
`custom_dataset.py`, pass them for PDF metrics:

```bash
python infer_chandra.py \
  --pdf input.pdf \
  --references-json annotated.json \
  --reference-field markdown
```

Metrics:

- `cer`: normalized character edit distance.
- `wer`: normalized whitespace-token word edit distance.
- `teds`: ordered tree similarity over the full parsed HTML.
- `table_teds`: ordered tree similarity over extracted `<table>` elements only;
  reported as `n/a` when the reference has no tables.
