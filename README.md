# Chandra Fine-Tuning

## Repository Layout

```text
chandra_finetuning/
├── README.md                      # This file
├── requirements-unsloth.txt       # Python dependencies for the CUDA training env
│
├── prompts.py                     # OCR / OCR-layout prompt templates + allowed HTML tags
├── custom_dataset.py              # Build an Arrow dataset from a single PDF + OCR JSON
├── prepare_chandra_dataset.py     # Split a dataset into train/test artifacts
├── train_chandra.py               # LoRA fine-tuning entry point
├── infer_chandra.py               # Transformers / Unsloth inference + evaluation
├── inf_vllm.py                    # vLLM server-based inference + evaluation
├── postprocess.py                 # Clean prediction/reference HTML in result files
│
└── chandra_finetune/              # Reusable library package
    ├── __init__.py                # DEFAULT_MODEL_NAME = "datalab-to/chandra"
    ├── data.py                    # Dataset loading, normalization, lazy Arrow, splitting
    ├── modeling.py                # Unsloth FastVisionModel loading + LoRA configuration
    ├── generation.py              # Single-image text generation helper
    ├── metrics.py                 # Public metric API (re-exports)
    └── metrics_relaxed.py         # CER / WER / TEDS / table-TEDS implementations
```



## Requirements

Pinned dependencies (`requirements-unsloth.txt`):

```text
unsloth
transformers==4.57.1
trl==0.22.2
datasets==4.3.0
pillow
pymupdf
beautifulsoup4
lxml
```

---

## Installation
file:///home/softsensor/Downloads/chandra_full_pipeline_reference.png
```bash
# Create and activate a dedicated environment (3.10–3.12)
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-unsloth.txt
```

---

## Pipeline Flow

The project is four stages run in sequence. Raw documents become a dataset, the dataset is split, training produces a LoRA adapter, and inference scores that adapter against the held-out test set.

```text
Stage 1 ─ Dataset      Stage 2 ─ Split        Stage 3 ─ Train
PDF + OCR JSON  ──────►  train/ + test/  ──────►  best/ + last/
(custom_dataset.py)     (prepare_chandra_       (train_chandra.py)
                         dataset.py)                   │
                                                       ▼
                                              Stage 4 ─ Inference
                                              eval via vLLM server
                                              (inf_vllm.py)
```


### Stage 1 — Dataset creation (`custom_dataset.py`)

```text
read in-file constants (PDF_PATH, OCR_PATH, OUTPUT_DIR, DPI)
        │
        ├── open PDF (PyMuPDF)        ── page count, render handle
        └── load OCR JSON             ── page → markdown
        │
        ▼
for each annotated page:
    render page → PNG bytes
    attach OCR prompt + reference markdown + metadata
    (skip pages out of range or with empty markdown)
        │
        ▼
write one Arrow dataset directory
    columns: image · prompt · reference · metadata
```

This script is self-contained (it does not use the `chandra_finetune` package except the prompt text).

### Stage 2 — Split (`prepare_chandra_dataset.py`)

```text
load_chandra_dataset(input)        ── read Arrow dir / pkl / json
        │
        ▼
split_samples(test_ratio, seed)    ── deterministic shuffle + split
        │
        ├── save_samples_to_arrow()   → train/  and  test/
        └── save_samples_to_pickle()  (only if --format pickle/both)
        │
        ▼
write split_metadata.json          ── counts + settings
```

### Stage 3 — Training (`train_chandra.py`)

```text
load train split (lazy Arrow — images decode one at a time)
        │
        ▼
build LoRA config from CLI args        (LoraSettings)
        │
        ▼
load model + attach adapter            (Unsloth FastVisionModel; base frozen)
        │
        ▼
SFTTrainer loop ──────────────► trains adapter weights only
        ▲                              │ each epoch
        │                              ▼
        └──── Table-TEDS callback: generate on eval set → score →
              track best adapter → drive early stopping
        │
        ▼
save best/  (highest validation Table-TEDS)
save last/  (final training step)
```

### Stage 4 — Inference (`inf_vllm.py`, two terminals)

The model runs in a separate vLLM server you start first. The inference script loads **no** model — it is a client.

```text
TERMINAL 1 (long-running)            TERMINAL 2 (per run: inf_vllm.py)
┌───────────────────────┐            load test split
│ vLLM server           │            health-check server (poll /v1/models)
│ model + adapter on GPU │◄──────────┐       │
└───────────────────────┘  POST      │       ▼
            ▲               /v1/chat/ └── per sample: send image → get text
            └───────────────completions      │
                                              ▼
                                       clean_html()
                                              │
                                              ▼
                                       compute_metrics()   ── penalizes bad output
                                              │
                                              ▼
                                       aggregate_metrics() → write results file
```

Two values must line up between the terminals: `--vllm-url` must match the server's host/port, and `--vllm-model` must exactly match the adapter name registered in the server's `--lora-modules`. Because the model loads once in terminal 1, you can re-run terminal 2 as often as you like without paying the model-load cost again.

---

## How to Run Each File

Activate your environment first (`source .venv/bin/activate`). Run every command from the project root.

### `custom_dataset.py` — build an Arrow dataset from one PDF

Edit the constants at the top of the file, then run with no arguments:

```bash
# 1. Open custom_dataset.py and set:
#    PDF_PATH    = "/path/to/source.pdf"
#    OCR_PATH    = "/path/to/annotated.json"   # list of {"page": N, "markdown": "..."}
#    OUTPUT_DIR  = "/path/to/output_arrow_dir"
#    DPI         = 600
# 2. Run it:
python custom_dataset.py
```

Produces an Arrow dataset directory at `OUTPUT_DIR`. Run once per source document.

### `prepare_chandra_dataset.py` — split into train/test

```bash
python prepare_chandra_dataset.py \
  --input /path/to/vlm_dataset \
  --output-dir data/chandra_splits \
  --test-ratio 0.1 \
  --seed 3407 \
  --format arrow
```

Produces `data/chandra_splits/train/`, `data/chandra_splits/test/`, and `split_metadata.json`.

### `train_chandra.py` — fine-tune the LoRA adapter

```bash
python train_chandra.py \
  --dataset data/chandra_splits/train \
  --eval-dataset data/chandra_splits/test \
  --model-name datalab-to/chandra \
  --output-dir outputs/chandra_lora \
  --num-train-epochs 15 \
  --early-stopping-patience 5
```

Produces `outputs/chandra_lora/best/` and `outputs/chandra_lora/last/`. Omit `--eval-dataset` to train without best-model selection (then also omit `--eval-strategy` or set it to `no`).

### `inf_vllm.py` — inference + evaluation via vLLM (recommended; two terminals)

**Terminal 1 — start the server (leave it running):**

```bash
vllm serve datalab-to/chandra \
  --enable-lora \
  --lora-modules chandra_lora=outputs/chandra_lora/best \
  --port 8000
```

**Terminal 2 — run inference against it:**

```bash
python inf_vllm.py \
  --dataset data/chandra_splits/test \
  --vllm-url http://localhost:8000 \
  --vllm-model chandra_lora \
  --output vllm_predictions.csv \
  --metrics cer,wer,teds,table_teds
```

`--vllm-model` must match the name on the left of `=` in `--lora-modules`. The client health-checks the server first and waits up to `--wait-timeout` seconds; pass `--no-wait` only if the server is already up. Re-run terminal 2 freely without restarting terminal 1.

### `infer_chandra.py` — inference + evaluation in-process (single terminal, no server)

```bash
python infer_chandra.py \
  --dataset data/chandra_splits/test \
  --model-name datalab-to/chandra \
  --adapter outputs/chandra_lora/best \
  --output predictions.jsonl \
  --metrics cer,wer,teds,table_teds
```

Loads the model and adapter in-process (slower to start, reloads every run). Also accepts `--image` or `--pdf` instead of `--dataset`.


## Dataset Format

Arrow is the expected, primary format. Each record is a typed Hugging Face `Dataset` row with these fields:

| Field | Type | Description |
| --- | --- | --- |
| `image` | `datasets.Image` | The page image, stored as PNG bytes (decoded lazily at access time). |
| `prompt` | `string` | The instruction shown to the model (defaults to the OCR prompt). |
| `reference` | `string` | Ground-truth HTML/markdown for the page. |
| `metadata` | `string` | JSON string with provenance such as `pdf_path`, `ocr_path`, `page_number`. |

A typical split directory looks like:

```text
data/chandra_splits/
├── train/
│   ├── data-00000-of-00001.arrow
│   ├── dataset_info.json
│   └── state.json
├── test/
│   ├── data-00000-of-00001.arrow
│   ├── dataset_info.json
│   └── state.json
└── split_metadata.json
```

The loader also accepts `.pkl` / `.pickle`, `.json`, and `.jsonl` inputs for compatibility, and will pull a list out of a dict under keys like `data`, `samples`, `dataset`, or `records`. At normalization time, each sample is turned into an Unsloth vision conversation: a `user` turn carrying the prompt text and the image, optionally followed by an `assistant` turn carrying the reference.




