# Chandra Unsloth Fine-Tuning

This directory contains reusable scripts for LoRA fine-tuning and evaluation of
`datalab-to/chandra` using the dataset shape produced by `custom_dataset.py`.

The current workflow uses Arrow / Hugging Face Dataset artifacts as the primary
dataset format. The older pickle flow is only kept for compatibility. The data
pipeline now also uses lazy image wrappers so that page images are loaded from
disk only when a training or inference sample is consumed, instead of keeping
large PIL objects inside serialized dataset records.

Evaluation also treats invalid or failed generations more strictly. CER, WER,
TEDS, and table-level TEDS should reflect real model failures instead of silently
skipping empty predictions, malformed outputs, or generations that cannot be
parsed.

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

## Dataset Preparation

First create or prepare the Chandra dataset artifacts. Arrow is the default
format because it stores the dataset in a typed Hugging Face Dataset layout and
reloads more reliably than a large pickle of PIL objects.

The pipeline should now be treated as Arrow-first:

```bash
python prepare_chandra_dataset.py \
  --input /path/to/vlm_dataset \
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

The `train/` and `test/` directories are Arrow dataset directories. These are the
main artifacts used by training and evaluation.

Use `--format pickle` or `--format both` only if you still need `train.pkl` and
`test.pkl` for an older compatibility path. New training and inference runs
should prefer Arrow directories.

## Arrow Dataset Format

Arrow is now the expected dataset format for this workflow.

The dataset preparation step writes Hugging Face Dataset directories instead of
using a single `.pkl` file as the primary source of truth. This is useful because
Arrow datasets are easier to reload, inspect, split, and reuse across training,
normal inference, and vLLM inference.

A typical split layout is:

```text
data/chandra_splits/
  train/
    data-00000-of-00001.arrow
    dataset_info.json
    state.json
  test/
    data-00000-of-00001.arrow
    dataset_info.json
    state.json
  split_metadata.json
```

Each dataset record contains the fields needed to reconstruct an Unsloth
vision-language message at runtime. Image data is not expected to be stored as a
large pickled PIL object. Instead, image references are resolved when the sample
is loaded.

Keep any generated page-image directory or image paths available after dataset
creation. The lazy image wrapper depends on those paths being valid during
training and inference.

## Lazy Image Wrapper

The dataset loading path now supports lazy image wrappers.

Previously, the workflow depended more heavily on pickle-style records that could
contain in-memory PIL images. That approach made artifacts large, fragile, and
harder to move between machines. The updated flow keeps image references in the
dataset and loads the actual image only when it is needed.

The lazy image wrapper is responsible for:

- Keeping Arrow dataset rows lightweight.
- Opening page images from disk only at sample access time.
- Converting images into the PIL/RGB shape expected by Chandra and Unsloth.
- Avoiding loading the full document image set into memory at startup.
- Making train/test Arrow splits easier to reuse for both inference paths.

This means the dataset artifact and the image files should be treated as a pair.
Do not delete or move the referenced images after preparing the dataset unless
you also regenerate or update the dataset references.

## Training

Run LoRA training on the Arrow train split:

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

For new runs, prefer Arrow directories:

```text
data/chandra_splits/train
data/chandra_splits/test
```

Use `--max-steps -1 --num-train-epochs 1` for epoch-based training.

The training flow should:

- Load the Arrow dataset split.
- Resolve image references lazily.
- Convert each sample into the Chandra/Unsloth message format.
- Attach LoRA adapters to `datalab-to/chandra`.
- Train only the adapter weights.
- Save the adapter and tokenizer under the configured output directory.

Expected output:

```text
outputs/chandra_lora/
  adapter_config.json
  adapter_model.safetensors
  tokenizer files
  training artifacts/logs
```

## Inference And Evaluation

The repository supports two inference paths:

1. **Normal inference** using `infer_chandra.py`
2. **vLLM inference** using `inf_vllm.py`

Use normal inference when you want the existing Hugging Face/Unsloth generation
flow. Use vLLM inference when you want to run generation through the vLLM-based
path.

Both inference paths should use the same Arrow test split when comparing results.

### Normal Inference

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

### vLLM Inference

Use `inf_vllm.py` for the vLLM-based inference flow. This path is useful when
you want faster generation or want to compare the normal inference output with
the vLLM output on the same test split.

```bash
python inf_vllm.py \
  --dataset data/chandra_splits/test \
  --model-name datalab-to/chandra \
  --adapter outputs/chandra_lora \
  --output vllm_predictions.csv \
  --metrics cer,wer,teds,table_teds
```

When comparing normal inference and vLLM inference, keep the dataset, model,
adapter, output path, and metrics arguments consistent between both runs.

Recommended comparison flow:

```bash
# Normal inference
python infer_chandra.py \
  --dataset data/chandra_splits/test \
  --model-name datalab-to/chandra \
  --adapter outputs/chandra_lora \
  --output normal_predictions.jsonl \
  --metrics cer,wer,teds,table_teds

# vLLM inference
python inf_vllm.py \
  --dataset data/chandra_splits/test \
  --model-name datalab-to/chandra \
  --adapter outputs/chandra_lora \
  --output vllm_predictions.csv \
  --metrics cer,wer,teds,table_teds
```

### Switching Between Normal And vLLM Inference

To switch between inference modes, use the corresponding script:

```text
infer_chandra.py  -> normal inference
inf_vllm.py       -> vLLM inference
```

Both inference paths should follow the same dataset and output conventions so
that results can be compared directly.

## Metrics

The evaluation scripts support the following metrics:

- `cer`: normalized character edit distance.
- `wer`: normalized whitespace-token word edit distance.
- `teds`: ordered tree similarity over the full parsed HTML.
- `table_teds`: ordered tree similarity over extracted `<table>` elements only;
  reported as `n/a` when the reference has no tables.

### Metric Penalization

Metric handling has been updated so that bad generations are counted as failures
instead of being ignored.

The evaluation path should penalize cases such as:

- Empty model predictions.
- Failed generation calls.
- Outputs with no usable OCR text.
- Malformed HTML when HTML is required for TEDS.
- Invalid or missing `<table>` structures when table-level scoring is expected.
- Samples that cannot be parsed or aligned with the expected reference.
- Runtime failures for individual samples.

This is important because skipping those samples can make the model look better
than it is. The summary metrics should reflect both output quality and generation
reliability.

Expected behavior:

- Empty or failed predictions should receive worst-case CER/WER treatment.
- Invalid HTML should receive poor TEDS treatment instead of being dropped.
- Missing table output should be penalized when the reference contains tables.
- Per-sample result files should still record the error or failure reason.
- Aggregate metrics should include penalized samples.

## Recommended Workflow

Use this order for a clean training and evaluation run:

```bash
# 1. Prepare Arrow train/test splits
python prepare_chandra_dataset.py \
  --input /path/to/vlm_dataset \
  --output-dir data/chandra_splits \
  --test-ratio 0.1 \
  --seed 3407 \
  --format arrow

# 2. Train LoRA adapter
python train_chandra.py \
  --dataset data/chandra_splits/train \
  --eval-dataset data/chandra_splits/test \
  --model-name datalab-to/chandra \
  --output-dir outputs/chandra_lora \
  --max-steps 30

# 3. Run normal inference
python infer_chandra.py \
  --dataset data/chandra_splits/test \
  --model-name datalab-to/chandra \
  --adapter outputs/chandra_lora \
  --output normal_predictions.jsonl \
  --metrics cer,wer,teds,table_teds

# 4. Optionally run vLLM inference
python inf_vllm.py \
  --dataset data/chandra_splits/test \
  --model-name datalab-to/chandra \
  --adapter outputs/chandra_lora \
  --output vllm_predictions.csv \
  --metrics cer,wer,teds,table_teds
```

## Notes

- Use a CUDA-compatible Python 3.10-3.12 environment for training and inference.
- Treat Arrow directories as the main dataset format.
- Treat pickle files as compatibility artifacts only.
- Keep referenced page images available for lazy loading.
- Use the same test split when comparing normal inference and vLLM inference.
- Keep metric penalization enabled so failed samples are reflected in the final
  evaluation numbers.
- Review per-sample outputs in addition to aggregate metrics when debugging model
  behavior.
