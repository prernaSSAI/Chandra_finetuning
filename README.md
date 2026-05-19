# Chandra OCR Finetuning fo Amneal

## Overview

This repository defines the OCR model adaptation workflow for pharmaceutical document processing using the **Chandra OCR vision-language model** and the **Unsloth LoRA finetuning framework**.

The workflow is designed for client-specific document understanding, where PDF pages and their corresponding OCR JSON annotations are converted into a vision-language training dataset. The generated `.pkl` dataset is then used to finetune Chandra with LoRA adapters, enabling efficient model adaptation with reduced VRAM usage and faster training.

---

## Objective

The objective is to improve OCR extraction quality on pharmaceutical document pages by:

* Converting PDF pages and OCR JSON annotations into page-level VLM training samples
* Training the Chandra OCR model using LoRA-based finetuning with Unsloth
* Evaluating OCR output quality using CER, WER, and TEDS metrics
* Maintaining a reproducible dataset creation, training, and evaluation workflow

---

## Base Model

* **Model:** `datalab-to/chandra`
* **Type:** Vision-Language OCR Model
* **Finetuning Method:** LoRA using Unsloth
* **Frameworks Used:**

  * Unsloth `FastVisionModel`
  * Hugging Face Transformers
  * TRL `SFTTrainer`
  * Hugging Face Datasets

---

## Installation

Install dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Input Data

Before creating the dataset, prepare the following inputs:

* PDF document file
* OCR JSON annotation file

The dataset creation script reads the PDF and OCR JSON, converts each PDF page into an image, maps it with the corresponding OCR markdown text, and saves the output in training-ready format.

---

## Data Format

Each generated sample follows a conversation-style vision-language format:

```text
User → OCR instruction + document page image
Assistant → Ground truth OCR text
```

Each sample contains:

* One page image converted from the PDF
* One ground truth OCR text response from the OCR JSON
* Message-style structure compatible with vision-language finetuning

---

## Execution Order

Follow the steps below in order.

### Step 1: Create the Custom Dataset

Update the PDF path, OCR JSON path, and output directory inside `custom_dataset.py`, then run:

```bash
python custom_dataset.py
```

This script will:

* Read the input PDF
* Read the OCR JSON annotation file
* Convert PDF pages into RGB images
* Pair each image with its corresponding OCR markdown text
* Save page images for inspection
* Generate dataset outputs

Expected outputs:

```text
data_pkl/page_images/
data_pkl/vlm_dataset.json
data_pkl/vlm_dataset.pkl
```

The `.pkl` file contains the final training-ready dataset and is used by the LoRA training script.

---

### Step 2: Train the Model with Unsloth LoRA

After the `.pkl` files are created, place them inside the directory configured in the training script, for example:

```text
./data_pkl/
```

Then run:

```bash
python train_lora.py
```

The training script will:

* Load all `.pkl` files from the configured dataset directory
* Validate the page-level image/text samples
* Convert samples into a Hugging Face Dataset
* Split the data into train, validation, and test sets
* Load the Chandra base model
* Attach LoRA adapters using Unsloth
* Train the model using `SFTTrainer`
* Save the best model checkpoint and split manifest

Expected training outputs include:

```text
chandra_output/
├── best_model/
├── run_config.json
├── split_manifest.json
└── train.log
```

---

### Step 3: Evaluate the Trained Model

After training is complete, update the evaluation script paths for:

* `pkl_dir`
* `model_path`
* `split_manifest`
* `output_dir`

Then run:

```bash
python chandra_evaluate_cer_wer_teds.py
```

The evaluation script will:

* Load the held-out evaluation split from the split manifest
* Load the trained model checkpoint
* Generate OCR predictions for evaluation samples
* Compare predictions with ground truth text
* Compute CER, WER, and TEDS metrics
* Save detailed and aggregate evaluation results

Expected evaluation outputs:

```text
chandra_eval_test/
├── eval_config.json
├── evaluation_results.json
├── evaluation_results.csv
├── evaluation_summary.json
└── eval.log
```

---

## Evaluation Metrics

The evaluation script reports:

* **CER (Character Error Rate):** Measures character-level OCR errors
* **WER (Word Error Rate):** Measures word-level OCR errors
* **TEDS (Tree Edit Distance Similarity):** Measures table-structure similarity for HTML/table outputs

---

## Why LoRA and Unsloth

LoRA is used to make finetuning more efficient by training a smaller number of adapter parameters instead of updating the full model. This helps reduce GPU memory usage and improves experimentation speed.

Unsloth is used for memory-efficient vision-language finetuning. It supports optimized model loading, LoRA adapter attachment, and gradient checkpointing, which helps reduce VRAM requirements during training.

---

## Notes

* Ensure the PDF and OCR JSON paths are correctly configured before running dataset creation
* Ensure `.pkl` files are generated before starting training
* Ensure the evaluation script points to the trained model checkpoint and correct split manifest
* Use LoRA with Unsloth for efficient training with lower VRAM usage and faster experimentation

---
