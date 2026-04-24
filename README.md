# Chandra OCR Finetuning

## Overview

This repository contains experiments for finetuning the **Chandra OCR vision-language model** using the **Unsloth framework**. The project explores different finetuning strategies to improve OCR performance on document images.

---

##  Goal

The goal of this project is to finetune the Chandra OCR model using:

* **Full finetuning (with frozen vision layers)**
* **LoRA-based finetuning**

The objective is to:

* Improve OCR accuracy on document-level data
* Compare efficiency and performance of different finetuning approaches
* Build a scalable and reproducible training pipeline

---

##  Base Model

* **Model:** `datalab-to/chandra`
* **Type:** Vision-Language Model (OCR)
* **Frameworks Used:**

  * Unsloth (`FastVisionModel`)
  * Hugging Face Transformers
  * TRL (`SFTTrainer`)

---

## Data Format

The dataset is stored in `.pkl` files and follows a structured conversation-style format.

### Conceptual Representation

```
User → Image  
Assistant → Extracted OCR Text
```

### Key Points

* Each sample contains:

  * One **image input**
  * One **ground truth text output**
* Data is processed at **page level**
* Images are automatically:

  * Converted to RGB
  * Resized during preprocessing

---

## ⚙️ Finetuning Approaches

### 1. Full Finetuning (Vision Frozen)

* Vision encoder is **frozen**
* Only the **language model is trained**
* No LoRA / QLoRA used

**Why:**

* More stable training
* Lower memory usage
* Strong baseline

---

### 2. LoRA Finetuning

* Uses **Low-Rank Adaptation (LoRA)**
* Enables parameter-efficient training
* Allows broader adaptation across model layers

**Why:**

* Faster experimentation
* Reduced compute cost
* Flexible fine-tuning

---

## 🧪 Summary

| Approach      | Vision Layers | Trainable Params | Efficiency | Use Case         |
| ------------- | ------------- | ---------------- | ---------- | ---------------- |
| Full Finetune | Frozen        | Medium           | Moderate   | Baseline         |
| LoRA Finetune | Trainable     | Low              | High       | Efficient tuning |

---

## 🚀 Usage

### Training

```
python <training_script>.py
```

### Evaluation

```
python <evaluation_script>.py
```

---

## 📈 Outputs

Each training run generates:

* Model checkpoints
* Training logs
* Training history
* Best model snapshot

Stored in:

```
outputs/<run_name>/
```

---

