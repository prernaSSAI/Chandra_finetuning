# Chandra OCR Finetuning fo Amneal

## Overview

This repository defines the OCR model adaptation workflow for pharmaceutical document processing using the **Chandra OCR vision-language model** and the **Unsloth LoRA finetuning framework**.

The workflow is designed for client-specific document understanding, where PDF pages and their corresponding OCR JSON annotations are converted into a vision-language training dataset. The dataset is now saved in **Arrow / Hugging Face Dataset format** instead of the earlier pickle-based format. This makes the dataset easier to reload, inspect, version, and reuse across training and evaluation runs.

The updated flow also includes **lazy image wrapper support**. Instead of storing full PIL image objects directly inside serialized dataset records, the dataset stores image paths and image metadata. Images are loaded only when the training or evaluation pipeline actually needs a sample. This keeps the dataset lightweight, avoids unnecessary memory usage, and prevents the problems that came from pickling large image objects.

Evaluation has also been tightened. CER, WER, and TEDS are still the primary metrics, but failed generations, empty outputs, invalid outputs, and malformed table/HTML outputs are now handled as penalized model failures instead of being silently ignored or skipped. This gives a more realistic view of model quality.

---

## Objective

The objective is to improve OCR extraction quality on pharmaceutical document pages by:

* Converting PDF pages and OCR JSON annotations into page-level VLM training samples
* Saving the generated dataset in Arrow / Hugging Face Dataset format instead of pickle
* Using lazy image wrappers so image files are loaded only when required
* Training the Chandra OCR model using LoRA-based finetuning with Unsloth
* Evaluating OCR output quality using CER, WER, and TEDS metrics
* Penalizing invalid, empty, failed, or structurally broken model outputs during metric computation
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
  * Apache Arrow dataset storage
  * PIL / image-path based lazy loading

---

## Installation

Install dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The training and evaluation scripts expect the normal Chandra / Unsloth stack to be available. Depending on the environment, additional OCR metric dependencies such as `jiwer`, `beautifulsoup4`, `lxml`, or `apted` may also be required for full CER, WER, and TEDS support.

---

## Input Data

Before creating the dataset, prepare the following inputs:

* PDF document file
* OCR JSON annotation file

The dataset creation script reads the PDF and OCR JSON, converts each PDF page into an image, maps it with the corresponding OCR markdown text, and saves the output in training-ready format.

The updated dataset format keeps the actual page image files on disk and stores references to those files inside the dataset records. This is important because the Arrow dataset should remain lightweight and should not contain heavy in-memory image objects.

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
* Page-level metadata such as page number and image path
* A lazy image reference that resolves the image from disk when the sample is consumed

The logical training sample is still the same as before: the model receives an OCR instruction and a document page image, then learns to produce the expected OCR markdown/text output. The main change is how the image is stored and loaded.

---

## Lazy Image Wrapper Change

Earlier versions of the workflow stored PIL image objects directly inside the pickled dataset. That made the dataset heavy and tightly coupled to the local Python environment.

The updated workflow uses a lazy wrapper approach:

* The PDF page is rendered and saved as an image file under the output image directory
* The dataset record stores the image path instead of embedding the full image object
* The lazy wrapper opens the image only when training or evaluation accesses that sample
* The image can still be converted to RGB and passed to the vision-language model at runtime

This change improves the pipeline because:

* Arrow datasets remain smaller and easier to move between machines
* Dataset loading does not immediately load every page image into memory
* Training can scale to more pages without requiring all images to be resident at once
* Evaluation can resolve only the held-out samples that are actually being scored
* The same page images can be reused by dataset creation, training, and evaluation

Because of this change, do not delete the generated `page_images/` directory after creating the dataset. The Arrow dataset depends on those image paths being valid.

---

## Dataset Storage Format

The dataset format is now **Arrow / Hugging Face Dataset format**.

The old pickle format is deprecated and should not be treated as the main training input anymore.

Expected outputs are now similar to:

```text
data_arrow/page_images/
data_arrow/vlm_dataset.json
data_arrow/vlm_dataset_arrow/
```

A typical expanded output layout is:

```text
data_arrow/
├── page_images/
│   ├── page_001.png
│   ├── page_002.png
│   └── ...
├── vlm_dataset.json
└── vlm_dataset_arrow/
    ├── data-00000-of-00001.arrow
    ├── dataset_info.json
    └── state.json
```

Where:

* `page_images/` contains the rendered page images used by lazy loading
* `vlm_dataset.json` is the human-readable inspection/debug file
* `vlm_dataset_arrow/` is the Hugging Face Dataset saved to disk and used by training/evaluation

The JSON file is useful for checking whether the correct image paths, page numbers, prompts, and OCR targets were created. The Arrow directory is the actual training-ready dataset artifact.

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
* Save page images for inspection and lazy loading
* Pair each image path with its corresponding OCR markdown text
* Build page-level conversation-style VLM samples
* Save a JSON dataset for inspection
* Save the final dataset in Arrow / Hugging Face Dataset format

Expected outputs:

```text
data_arrow/page_images/
data_arrow/vlm_dataset.json
data_arrow/vlm_dataset_arrow/
```

The Arrow dataset directory contains the final training-ready dataset. The image directory must remain available because the lazy wrapper loads images from those saved paths.

---

### Step 2: Train the Model with Unsloth LoRA

After the Arrow dataset is created, point the training script to the configured Arrow dataset directory, for example:

```text
./data_arrow/vlm_dataset_arrow/
```

Then run:

```bash
python lora_train.py
```

The training script will:

* Load the Arrow / Hugging Face Dataset from the configured dataset directory
* Resolve page images lazily through the image wrapper
* Validate page-level image/text samples
* Convert samples into the expected Chandra vision-language conversation format
* Split the data into train, validation, and test sets
* Load the Chandra base model
* Attach LoRA adapters using Unsloth
* Train the model using `SFTTrainer`
* Save the best model checkpoint and split manifest
* Save run configuration, logs, and training history when enabled

Expected training outputs include:

```text
chandra_output/
├── best_model/
├── run_config.json
├── split_manifest.json
├── training_history.csv
├── training_history.json
└── train.log
```

The split manifest is important because evaluation uses it to identify the held-out pages that should be scored. Keep it with the model output folder.

---

### Step 3: Evaluate the Trained Model

After training is complete, update the evaluation script paths for:

* Arrow dataset directory
* Model checkpoint path
* Split manifest path
* Output directory

Then run:

```bash
python chandra_evaluate_cer_wer_teds.py
```

The evaluation script will:

* Load the held-out evaluation split from the split manifest
* Load the corresponding samples from the Arrow dataset
* Resolve images lazily during evaluation
* Load the trained model checkpoint
* Generate OCR predictions for evaluation samples
* Compare predictions with ground truth text
* Compute CER, WER, and TEDS metrics
* Apply metric penalties for failed, empty, invalid, or malformed outputs
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

The metric computation is stricter now. A model output should be counted as a bad result when it cannot be evaluated properly because the generation failed, the prediction is empty, or the table/HTML structure is invalid.

---

## Metric Penalization Changes (more changes needed here)

The evaluation flow now penalizes problematic predictions instead of silently dropping them from the aggregate scores.

Examples of penalized cases include:

* Empty predictions
* Failed model generations
* Outputs that contain no useful OCR content
* Invalid or malformed HTML/table outputs for TEDS scoring
* Predictions that cannot be matched cleanly against the ground truth sample
* Samples where the model response is structurally broken or incomplete

This is important because skipping failed samples can make evaluation look better than it really is. Penalizing them makes the reported CER, WER, and TEDS summary closer to real production behavior.

The intended behavior is:

* Bad text output should increase CER/WER instead of disappearing from the report
* Bad table/HTML output should receive poor TEDS treatment instead of being ignored
* Failed samples should still be visible in detailed evaluation outputs
* Aggregate metrics should reflect both quality and reliability

---

## Notes

* Ensure the PDF and OCR JSON paths are correctly configured before running dataset creation
* Ensure the Arrow dataset is generated before starting training
* Keep the generated `page_images/` directory because lazy image wrappers load images from those paths
* Ensure the training script points to the Arrow dataset directory, not the old `.pkl` directory
* Ensure the evaluation script points to the trained model checkpoint and correct split manifest
* Ensure the evaluation script uses the same dataset/split information that was produced during training
* Treat pickle outputs as deprecated unless running an older legacy script
* Use LoRA with Unsloth for efficient training with lower VRAM usage and faster experimentation
* Review `evaluation_results.csv` and `evaluation_summary.json` after evaluation to confirm penalized failures are reflected in the final metrics

---
