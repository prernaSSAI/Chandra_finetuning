# Chandra Fine-Tuning

This folder holds the full pipeline for fine-tuning the `datalab-to/chandra` vision-language OCR model with LoRA, and for evaluating it: turning a PDF + OCR annotations into a training dataset, splitting it, training a LoRA adapter with Unsloth, and then serving/evaluating that adapter through vLLM.

Two separate Python virtual environments are used on this machine — one for training, one for inference — see [Environments](#environments--requirements) below.

---

## Repository Layout

```text
Chandra_finetuning/
├── README.md                      # This file
├── requirements-unsloth.txt       # Pinned deps shared by both venvs
│
├── prompts.py                     # OCR / OCR-layout prompt templates + allowed HTML tags
├── augmentation.py                # albumentations image cleaning (inference) + augraphy augmenters (training augmentation)
├── custom_dataset.py              # Build an Arrow dataset from a single PDF + OCR JSON
├── prepare_chandra_dataset.py     # Split a dataset into train/test artifacts
├── train_chandra.py               # LoRA fine-tuning entry point (edit TrainConfig, then run) — train_venv
├── infer_chandra.py               # Transformers/Unsloth in-process inference + evaluation (CLI) — train_venv
├── inf_vllm.py                    # vLLM server-based inference + evaluation (edit InferenceConfig, then run) — venv
├── visualize_predictions.py       # Render a predictions JSON to a side-by-side reference/prediction HTML page
│
├── train_venv/                    # Training environment (Unsloth + trl, no vLLM)
├── venv/                          # Inference environment (Unsloth + vLLM)
│
└── chandra_finetune/               # Reusable library package
    ├── __init__.py                 # DEFAULT_MODEL_NAME = "datalab-to/chandra"
    ├── data.py                     # Dataset loading, normalization, lazy Arrow, splitting
    ├── modeling.py                 # Unsloth FastVisionModel loading + LoRA configuration
    ├── generation.py                # Single-image text generation helper
    ├── metrics.py                   # Public metric API (re-exports; clean_html = postprocess_html_for_metrics)
    └── metrics_relaxed.py            # CER / WER / TEDS / table-TEDS implementations
```


## Environments & Requirements

This project uses **two separate venvs** on this machine because training (Unsloth) and serving (vLLM) pin conflicting/heavy CUDA builds of `torch`. Keep them separate — do not `pip install vllm` into `train_venv` or vice versa.

| Venv | Used for | Runs which scripts | Key packages |
| --- | --- | --- | --- |
| `train_venv` | Dataset prep + LoRA training + in-process inference | `custom_dataset.py`, `prepare_chandra_dataset.py`, `train_chandra.py`, `infer_chandra.py` | `unsloth`, `transformers==4.57.1`, `trl==0.22.2`, `datasets`, `torch` (CUDA build) |
| `venv` | vLLM server + client-side evaluation | `vllm serve …`, `inf_vllm.py`, `visualize_predictions.py` | everything in `train_venv` **plus** `vllm` |

Pinned base dependencies (`requirements-unsloth.txt`, install into **both** venvs):

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

Also required but not pinned in the requirements file:

```text
albumentations   # used by augmentation.py's apply_noise(), called on every image before inference
```

`augraphy` is only needed if you call `build_augraphy_augmenters()` in `augmentation.py` for training-data augmentation; its import is lazy so it's not required otherwise (`pip install augraphy==8.2.6`).

### Setting up `train_venv`

```bash
python3.10 -m venv train_venv
source train_venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-unsloth.txt
python -m pip install albumentations

deactivate
```

### Setting up `venv` (inference)

```bash
python3.10 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-unsloth.txt
python -m pip install albumentations
python -m pip install vllm

deactivate
```

Switch between them explicitly with `source train_venv/bin/activate` / `source venv/bin/activate` (and `deactivate` when done) — never assume which one is currently active, check your shell prompt.

---

## Pipeline Flow

```text
Stage 1 ─ Dataset      Stage 2 ─ Split        Stage 3 ─ Train              Stage 4 ─ Inference
PDF + OCR JSON  ──────►  train/ + test/  ──────►  best/ + last/   ──────►  vLLM serve + inf_vllm.py
(custom_dataset.py)     (prepare_chandra_       (train_chandra.py)         (venv, two terminals)
   [train_venv]          dataset.py)              [train_venv]
                          [train_venv]
```

1. **`custom_dataset.py`** — renders a PDF's pages to images and pairs each with its OCR markdown to build an Arrow dataset.
2. **`prepare_chandra_dataset.py`** — deterministically splits that dataset into `train/` and `test/`.
3. **`train_chandra.py`** — LoRA fine-tunes `datalab-to/chandra` on `train/`, saving `best/` (highest validation Table-TEDS, or lowest train loss if no eval set) and `last/`.
4. **`inf_vllm.py`** — with the trained adapter served by vLLM, runs inference over a dataset/PDF/images and computes CER/WER/TEDS/table-TEDS metrics.
5. **`visualize_predictions.py`** — optional; turns a predictions JSON from step 4 into a side-by-side HTML report for manual QA.

---

## How to Run Each File

Run every command from the project root, with the correct venv activated first.

### 1. `custom_dataset.py` — build an Arrow dataset from one PDF (`train_venv`)

```bash
source train_venv/bin/activate

# Edit the constants at the top of custom_dataset.py first:
#   PDF_PATH    = "/path/to/source.pdf"
#   OCR_PATH    = "/path/to/annotated.json"   # list of {"page": N, "markdown": "..."}
#   OUTPUT_DIR  = "/path/to/output_arrow_dir"
#   DPI         = 600

python custom_dataset.py
```

Produces an Arrow dataset directory at `OUTPUT_DIR`. Run once per source document.

### 2. `prepare_chandra_dataset.py` — split into train/test (`train_venv`)

```bash
python prepare_chandra_dataset.py \
  --input /path/to/vlm_dataset \
  --output-dir data/chandra_splits \
  --test-ratio 0.1 \
  --seed 3407 \
  --format arrow
```

Produces `data/chandra_splits/train/`, `data/chandra_splits/test/`, and `split_metadata.json`.

### 3. `train_chandra.py` — fine-tune the LoRA adapter (`train_venv`, long-running — use tmux)

No CLI flags. Open the file, edit the `TrainConfig` dataclass fields (`dataset`, `eval_dataset`, `output_dir`, LoRA rank/alpha, learning rate, epochs, early-stopping patience, etc.), then run inside a tmux session (see [Running long jobs in tmux](#running-long-jobs-in-tmux)):

```bash
tmux new -s train
source train_venv/bin/activate
python train_chandra.py
# Ctrl+B then D to detach and let it keep running
```

Produces `<output_dir>/best/` and `<output_dir>/last/`. Leave `eval_dataset = None` to train on train-loss only (no Table-TEDS best-model selection).

### 4. Serve + `inf_vllm.py` — inference + evaluation via vLLM (`venv`, two terminals/tmux windows)

The model is served separately from the client script — start the server first, then run the client against it.

**Terminal 1 — start the vLLM server (leave it running, use tmux):**

```bash
tmux new -s vllm_server
source venv/bin/activate
vllm serve datalab-to/chandra \
  --enable-lora \
  --lora-modules chandra_lora=outputs/chandra_lora/best \
  --port 8000
# Ctrl+B then D to detach and leave the server running
```

**Terminal 2 — run inference against it:**

```bash
tmux new -s inference
source venv/bin/activate

# Edit InferenceConfig in inf_vllm.py first: exactly one input source
# (dataset / image / pdf / pkl_dir), vllm_url, vllm_model, output, metrics, concurrency.

python inf_vllm.py
# Ctrl+B then D to detach if it's a long run
```

`vllm_model` in `InferenceConfig` must exactly match the name on the left of `=` in `--lora-modules`, and `vllm_url` must match the server's host/port. The client health-checks the server first and waits up to `wait_timeout` seconds; set `no_wait = True` only if the server is already confirmed up. Because the model loads once in terminal 1, re-run terminal 2 as often as you like without paying the model-load cost again — it also resumes automatically, skipping any page already written successfully (no `"error"` key) in the output file.

### 5. `visualize_predictions.py` — side-by-side HTML viewer (either venv)

```bash
python visualize_predictions.py outputs/chandra_lora/predictions.json report.html
```

Renders the predictions JSON (from `inf_vllm.py` or `infer_chandra.py`) into one HTML page with the source page image next to the reference and prediction HTML, for visual QA. Pass `--pdf /path/to/source.pdf` to force rendering the source PDF page instead of relying on `metadata.pdf_path`; `--title` sets the page heading.

### `infer_chandra.py` — inference + evaluation in-process (`train_venv`, single terminal, no server)

Use this only when you don't want to stand up a vLLM server (e.g. a quick single-checkpoint sanity check). It reloads the model every run, so it's slower to start than the vLLM path.

```bash
python infer_chandra.py \
  --dataset data/chandra_splits/test \
  --model-name datalab-to/chandra \
  --adapter outputs/chandra_lora/best \
  --output predictions.jsonl \
  --metrics cer,wer,teds,table_teds
```

Also accepts `--image` or `--pdf` instead of `--dataset`.

---

## Running Long Jobs in tmux

Training and inference runs can take hours, so they should be run inside `tmux` rather than a bare SSH/terminal session — if your connection drops, a bare terminal job dies with it, but a tmux session keeps running on the server.

**Start a new named session:**

```bash
tmux new -s <session_name>     # e.g. tmux new -s train
```

**Detach (leave it running in the background):**

Press `Ctrl+B`, release, then press `D`. This returns you to your normal shell while the session keeps running.

**Re-attach to a session later (e.g. after reconnecting):**

```bash
tmux attach -t <session_name>
```

**List all running sessions:**

```bash
tmux ls
```

**Stop/kill a session** (only once the job inside it is actually done, or you intentionally want to abort it):

```bash
tmux kill-session -t <session_name>
```

Run each long job (`train_chandra.py`, `vllm serve`, `inf_vllm.py`) in its own named tmux session so you can check on them independently — e.g. `train`, `vllm_server`, `inference`.

---

## Other Notes for Anyone Working in This Codebase

- **GPU is shared/single**: training and vLLM serving both need the GPU. Don't start `train_chandra.py` while a vLLM server for the same GPU is running unless you've confirmed there's enough VRAM headroom.
- **In-file configs, not CLI flags**: `train_chandra.py` and `inf_vllm.py` are configured by editing the `TrainConfig` / `InferenceConfig` dataclasses at the top of each file, not by passing command-line arguments. Always double check the paths in these before kicking off a run — several fields (e.g. `dataset`, `output_dir`, `pdf`, `output`) currently point at absolute paths from a prior run and must be updated for a new one.
- **Resumable outputs**: both `train_chandra.py` (via `resume_from_checkpoint`) and `inf_vllm.py` (automatically, via the existing output JSON) support resuming — check for an existing output/checkpoint before assuming you need a full re-run.
- **`vllm_model` naming must match**: the adapter name in `--lora-modules chandra_lora=...` and `InferenceConfig.vllm_model` in `inf_vllm.py` must be identical strings, or the client's health-check will hang/fail.
- **Metrics are computed on cleaned text**: `clean_html()` is applied to both prediction and reference before CER/WER/TEDS/table-TEDS are computed, so a raw model output that "looks right" but fails `clean_html()`'s HTML rules will still score poorly — the `*_raw` fields in prediction JSON are kept for debugging this.
- **`.gitignore`** already excludes `venv/`, `train_venv/`, and `unsloth_compiled_cache/` — never commit these.
- Do not modify `unsloth_compiled_cache/` by hand; it's regenerated by Unsloth at import time.

---

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

The loader also accepts `.pkl`/`.pickle`, `.json`, and `.jsonl` inputs for compatibility, pulling a list out of a dict under keys like `data`, `samples`, `dataset`, or `records`. At normalization time, each sample is turned into an Unsloth vision conversation: a `user` turn carrying the prompt text and the image, optionally followed by an `assistant` turn carrying the reference.
