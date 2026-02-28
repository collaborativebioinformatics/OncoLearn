# OncoLearn Multimodal Module — Code Documentation

This document describes the **multimodal learning framework** under `OncoLearn/src/multimodal`. It integrates genomics (gene expression), clinical (tabular), and imaging (DICOM) data for TCGA-BRCA cancer subtyping and staging. The pipeline achieves **around 80% performance** (e.g., accuracy/F1 on validation) for stage and subtype classification when trained with the provided setup.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Data Module](#data-module)
4. [Models](#models)
5. [Training & Evaluation](#training--evaluation)
6. [Project Structure](#project-structure)
7. [Performance](#performance)

---

## Overview

The multimodal module supports two variants:

- **V1 (imaging-present):** Gene expression + clinical tabular + DICOM imaging (MR/MG). Uses sequence-level expansion (multiple series per patient) and a 3-modality gated late-fusion classifier.
- **V2 (no-imaging):** Gene expression + clinical tabular only. Same fusion design with 2 modalities.

Tasks:

- **Stage classification:** 5 classes — Stage I, II, III, IV, Unknown.
- **Subtype classification (optional):** e.g. PAM50 or derived subtypes (HR+/HER2-, HR+/HER2+, HR-/HER2+, TNBC, Unknown), when labels are provided (BRCA labels file or PAM50).

Design choices:

- **Modality dropout** during training for robustness to missing modalities.
- **Stratified K-fold** (or external test split via `test_patients_path`) for train/val/test.
- **Class weights** for imbalanced stage/subtype labels.
- **Mixed precision (AMP)** and **early stopping** supported in training.

---

## Architecture

### High-level flow

```
[Gene features]     → RNABERTEncoder        → z_gene (128-d)
[Clinical table]   → FTTransformerEncoder  → z_clinical (128-d)
[DICOM images]     → MRMGHierarchicalImageEncoder → z_image (256-d)   [V1 only]
                          ↓
              GatedLateFusionClassifier
                    (per-modality heads + gate network)
                          ↓
              stage_logits, [subtype_logits]
```

### Fusion (Gated Late Fusion)

- Each modality is encoded to a fixed-size embedding.
- Each modality has its own **stage head** (and optionally **subtype head**) producing logits.
- A **gate network** takes the concatenation of available embeddings and outputs weights over modalities (masked for missing modalities, then softmax).
- Final logits = weighted sum of per-modality logits. This allows the model to rely more on one modality when others are missing or noisy.

---

## Data Module

### Location

`data/` (relative to `src/multimodal`).

### Components

| File | Purpose |
|------|--------|
| **cohort.py** | Load cohort index (parquet/csv), clinical table, gene set table; detect imaging presence; build V1/V2 cohorts via `get_cohort_for_variant`. |
| **labels.py** | **LabelManager:** stage/subtype label discovery and derivation (stage column discovery, PAM50/BRCA file loading, ER-PR-HER2–based subtype); class weights for stage and subtype. |
| **dicom_io.py** | DICOM read with optional cache; parse `dicom_series` JSON; list series per patient; uniform/random sampling of slices; batch loading. |
| **transforms.py** | **DICOMToTensor**, **ResizeDICOM**, **DICOMTransform:** normalize, resize (e.g. 224×224), optional augmentation (flip, rotation). |
| **pairs_dataset.py** | **TCGAPairsDataset:** paired image (patch path or directory) + precomputed omics vector (`.npy`) from a split CSV; used for simpler pipelines. |
| **datamodule.py** | **TCGADataModule:** builds train/val/test splits (stratified K-fold or from `test_patients_path`), **TCGAV1Dataset** / **TCGAV2Dataset**, and DataLoaders; collate functions for variable-length image sequences. |

### V1 dataset (imaging-present)

- **TCGAV1Dataset:** One sample per **(patient, series)** when `expand_by_sequences=True` (default). Each sample has:
  - Gene vector from gene set table (same for all series of the patient).
  - Clinical numeric vector (same for all series).
  - A fixed number of DICOM paths per series (e.g. 5), uniformly sampled from that series; loaded and transformed to 3×224×224; optional modality dropout (image/gene/clinical).
- **collate_fn_v1:** Pads image sequences to the same length per batch; stacks gene/clinical; handles missing modality via placeholders so batch size is consistent.

### V2 dataset (no-imaging)

- **TCGAV2Dataset:** One sample per patient; gene + clinical only; same modality dropout for gene/clinical.
- **collate_fn_v2:** Stacks gene and clinical; pads clinical to max length in batch.

### Splits

- **Standard:** Stratified K-fold on stage labels → train/val only (no test).
- **With test set:** If `test_patients_path` is set, test patients are loaded from that file; the rest are split into train/val (stratified when possible).

---

## Models

### Location

`src/models/` (and `data/` as above).

### Encoders

| Model | Input | Output | Description |
|-------|--------|--------|-------------|
| **RNABERTEncoder** | (B, P) gene expression | (B, 128) | Wraps IBM biomed.rna.bert.110m (e.g. `ibm-research/biomed.rna.bert.110m.mlm.multitask.v1`). Backbone can be frozen; projection to 128-d. |
| **FTTransformerEncoder** | (B, clinical_dim) | (B, 128) | TabTransformer (continuous-only); backbone frozen; projection to 128-d. |
| **MRMGHierarchicalImageEncoder** | (B, N, 3, H, W), modality_ids (B, N) | (B, 256) | Loads pretrained checkpoint (ViT or 3D ViT); per-image features → 256-d; **HierarchicalAttentionPooling** with modality embedding (MR=0, MG=1) over N images; output 256-d. Used only in V1. |

### Fusion

- **GatedLateFusionClassifier**
  - **Inputs:** Optional `gene`, `clinical`, `image`, `modality_ids`.
  - **Heads:** Per-modality stage (and optionally subtype) heads.
  - **Gate:** MLP on concatenated embeddings → mask missing modalities → softmax → weights.
  - **Output:** `stage_logits`; `subtype_logits` if `num_subtype_classes > 0`.

### Supporting modules

- **vit_3d_wrapper.py** (commented out): 3D ViT wrapper for 2D slices (pseudo-3D); used when checkpoint is 3D.
- **vit_block.py** (commented out): Transformer block used by the 3D wrapper.
- **image_encoder.py** can load 2D ViT (e.g. HuggingFace ViT) or 3D ViT from checkpoint; feature dim is projected to 256 before hierarchical pooling.

---

## Training & Evaluation

### Training (`src/train.py`)

- **build_model:** Builds RNABERTEncoder, FTTransformerEncoder, optional MRMGHierarchicalImageEncoder (V1), then GatedLateFusionClassifier.
- **train_epoch:** One epoch with optional AMP; loss = stage_loss + subtype_lambda × subtype_loss; modality dropout applied in the dataloader.
- **validate:** Computes loss and metrics (accuracy, balanced accuracy, macro F1) for stage (and subtype if present).
- **main:** Parses variant (v1_imaging / v2_no_imaging), config, data paths, fold; builds TCGADataModule; uses class-weighted CrossEntropy; AdamW + ReduceLROnPlateau; best checkpoint by validation stage F1; early stopping.

### Evaluation (`src/eval.py`)

- **evaluate:** Runs model on a given dataloader (val or test); collects stage and subtype predictions; computes accuracy, balanced accuracy, macro F1, confusion matrix; returns metrics dict and predictions DataFrame.
- **main:** Loads config and checkpoint; builds datamodule and model; runs evaluation on val or test (e.g. `--use_test`); saves metrics JSON and predictions CSV.

### Utilities (`src/utils.py`)

- **set_seed:** Reproducibility for random, numpy, torch, cudnn.
- **load_config / save_config:** YAML config with optional `_base_` inheritance and overrides.
- **setup_logging:** File + stdout logging.
- **save_jsonl:** Append JSONL lines.

---

## Project Structure

```
multimodal/
├── data/
│   ├── cohort.py          # Cohort loading, V1/V2 cohort selection
│   ├── datamodule.py      # TCGADataModule, TCGAV1/V2Dataset, collate
│   ├── dicom_io.py        # DICOM read, series listing, sampling
│   ├── labels.py          # LabelManager (stage, subtype, weights)
│   ├── pairs_dataset.py   # TCGAPairsDataset (image + omics from CSV)
│   └── transforms.py      # DICOM transforms and augmentation
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fusion.py              # GatedLateFusionClassifier
│   │   ├── gene_encoder.py        # RNABERTEncoder
│   │   ├── tab_encoder.py         # FTTransformerEncoder
│   │   ├── image_encoder.py       # MRMGHierarchicalImageEncoder
│   │   ├── vit_block.py           # (commented) Transformer block
│   │   └── vit_3d_wrapper.py      # (commented) 3D ViT wrapper
│   ├── train.py           # Training entry point
│   ├── eval.py            # Evaluation entry point
│   └── utils.py           # Config, logging, seed
├── scripts/               # Shell scripts for training / federated
├── pre_trained/           # biomed-multi-omic configs and assets
└── MULTIMODAL_DOCUMENTATION.md  # This file
```

---

## Performance

When trained with the default (or similar) configuration on TCGA-BRCA:

- The pipeline reaches **around 80%** performance on validation (e.g. accuracy or macro F1 for stage or subtype, depending on metric and split).
- Exact numbers depend on:
  - Train/val/test split and fold,
  - Use of PAM50 or BRCA subtype labels,
  - Whether V1 (with imaging) or V2 (gene + clinical only) is used,
  - Hyperparameters (learning rate, batch size, modality dropout, subtype_lambda, etc.).

For reproducible results, use the same config, seed, and data paths as in the experiments that reported ~80% performance.

---

## References (in-repo)

- **Cohort/labels:** `data/cohort.py`, `data/labels.py`
- **Fusion:** `src/models/fusion.py`
- **Encoders:** `src/models/gene_encoder.py`, `tab_encoder.py`, `image_encoder.py`
- **Training/eval:** `src/train.py`, `src/eval.py`
