# OncoLearn Multimodal Architecture — Code Documentation

This document describes the **multimodal learning framework** under `OncoLearn/src/oncolearn`. It integrates genomics (gene expression), clinical (tabular), and **imaging (DICOM)** data for TCGA-BRCA cancer subtyping and staging. The pipeline achieves **around 80% performance** (e.g., accuracy/F1 on validation) for stage and subtype classification when trained with the provided setup.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Architecture](#architecture)
4. [Data Module](#data-module)
5. [Models](#models)
6. [Training & Evaluation](#training--evaluation)
7. [Project Structure](#project-structure)
8. [Performance](#performance)

---

## Overview

The multimodal module uses **gene expression**, **clinical tabular**, and **DICOM imaging (MR/MG)** with a 3-modality gated late-fusion classifier. The dataset is expanded at the sequence level: one sample per (patient, series), with a fixed number of DICOM slices uniformly sampled per series.

Tasks:

- **Stage classification:** 5 classes — Stage I, II, III, IV, Unknown.
- **Subtype classification (optional):** e.g. PAM50 or derived subtypes (HR+/HER2-, HR+/HER2+, HR-/HER2+, TNBC, Unknown), when labels are provided (BRCA labels file or PAM50).

Design choices:

- **Modality dropout** during training for robustness to missing modalities.
- **Stratified K-fold** (or external test split via `test_patients_path`) for train/val/test.
- **Class weights** for imbalanced stage/subtype labels.
- **Mixed precision (AMP)** and **early stopping** supported in training.

---

## Data Preprocessing

This section describes how each data type is loaded, cleaned, and prepared before being fed to the model.

### 1. Input sources and loading

| Source | File format | Loading |
|-------|-------------|--------|
| **Cohort index** | `cohort_index.parquet` or `.csv` | Required columns: `patient_id`, `clinical_row_id`. Optional: `dicom_series` (JSON), `imaging_modalities`, `has_imaging` — used to select patients with imaging and to list DICOM series per patient. |
| **Clinical table** | `clinical_table.parquet` or `.csv` | Indexed by `patient_id` (set as index if provided as column). All columns kept; only **numeric** columns are used at runtime (see below). |
| **Gene set / expression** | `expression_matrix.parquet` or `.csv` (e.g. `gene_set_table.parquet`) | Indexed by `patient_id`. Rows = patients, columns = genes or gene-set scores. Loaded as-is; values are cast to `float32` when building the batch. No in-code normalization or scaling; any normalization is assumed to be done upstream (e.g. log-transform, z-score) before saving the matrix. |

- **Cohort (imaging-present):** Patients with imaging are those for whom `has_imaging` is set, or `imaging_modalities` contains `"MR"` or `"MG"`, or `dicom_series` is non-empty. Only these patients are used in this pipeline.

### 2. Label preprocessing (stage and subtype)

- **Patient ID:** TCGA sample barcodes are reduced to a 12-character patient ID by taking the first three segments (e.g. `TCGA-XX-XXXX`). Used to join cohort, clinical, gene set, and external label files.

- **Stage (5-way):**
  - **Column discovery:** The clinical table is scanned for a column whose name (lowercased) contains one of: `ajcc_pathologic_stage`, `pathologic_stage`, `stage`, `clinical_stage`. The first match is used.
  - **Normalization:** Raw values are mapped to one of: `Stage I`, `Stage II`, `Stage III`, `Stage IV`, `Unknown`. Rules: regex for “stage 1”/“stage i” → Stage I; “stage 2”/“stage ii” → Stage II; similarly for III/IV; empty/NaN or no match → `Unknown`. Class index is assigned via a fixed ordering of these five labels.

- **Subtype (optional):**
  - **Priority 1 — BRCA labels file:** If `brca_labels_file` (e.g. `BRCA-data-with-integer-labels.csv`) is provided, it is loaded. Required column: `sample_id`; label column is resolved from `Subtype` / `subtype` / `label`. Patient ID is derived from `sample_id` (first 12 characters). One subtype per patient (first occurrence if duplicates). Unique labels define the subtype classes; an extra `Unknown` class is added.
  - **Priority 2 — PAM50 file:** If `pam50_file` is provided (TSV with `Sample` and `PAM50`), patient ID is derived from `Sample` and mapped to PAM50 subtype. Unique PAM50 values (+ `Unknown`) define the classes.
  - **Priority 3 — Clinical table:** Search for a column whose name contains `pam50`, `subtype`, `molecular`, or `intrinsic`. The column is accepted only if it has 2–6 unique values and no single class exceeds 90% of samples.
  - **Priority 4 — Derived from ER/PR/HER2:** Columns containing `er_status`/`estrogen_receptor`, `pr_status`/`progesterone_receptor`, `her2_status`/`her2` are used. “Positive”/“+” and “negative”/“-” are parsed; HR = ER+ or PR+. Subtypes: HR+/HER2-, HR+/HER2+, HR-/HER2+, TNBC (HR-/HER2-); otherwise `Unknown`. If fewer than 10% of patients get a non-Unknown subtype, derivation is discarded.
  - **Class weights:** For both stage and subtype, inverse-frequency class weights are computed on the **training** patient set: `weight[c] = N / (num_classes * count[c])`, then used in `CrossEntropyLoss`.

### 3. Gene expression (omics)

- **At dataset level:** For each patient, the row corresponding to `patient_id` is taken from the gene set table and converted to a 1D float32 vector. No per-sample normalization or clipping in the dataloader.
- **Missing:** If a patient in the cohort is not in the gene set table, the dataset raises an error (no imputation).
- **Pairs dataset (alternative pipeline):** In `TCGAPairsDataset`, gene features are read from precomputed `.npy` files (one per sample); values are cast to `float32`. Again, any normalization is expected to be done before saving the `.npy` files.

### 4. Clinical (tabular) features

- **Selection:** Only **numeric** columns are used. In the datamodule, for each patient the clinical row is converted with `pd.to_numeric(..., errors='coerce')`; non-numeric entries become NaN.
- **Missing values:** All NaN values are filled with **0** before converting to a float32 tensor.
- **Variable length:** Patients can have different numbers of numeric columns depending on the table; in practice the clinical table is shared, so the length is the same for all. In **collate**, clinical vectors are padded to the **maximum length in the batch** with zero padding so they can be stacked into a tensor.

### 5. DICOM imaging

- **Reading:** `pydicom.dcmread(path)` loads the DICOM file. Pixel data is cast to `float32`. If the array is 3D (multi-slice), only the **middle slice** (`shape[0] // 2`) is used; if ndim > 3, the array is reshaped and the first 2D slice is taken.
- **Rescaling:** If the DICOM has `RescaleSlope` and `RescaleIntercept`, pixel values are updated as: `pixel_array = pixel_array * slope + intercept`.
- **Caching:** A global LRU cache (default size 1000) stores pixel arrays by path.
- **Series selection:** The cohort row’s `dicom_series` is a JSON mapping study UID → series UID → `{ "modality": "MR"|"MG", "example_paths": [...] }`. For each series, the directory from `example_paths[0]` is resolved and all `*.dcm`/`*.DCM` files are collected, sorted. Preferred modality is **MR**; otherwise other modalities are used.
- **Sampling slices:** From a series with `n_total` images, `n_dicom_samples` (e.g. 5) indices are chosen uniformly: `int(i * (n_total / n_dicom_samples))` for `i = 0..n_dicom_samples-1`. Only series with at least `n_dicom_samples` images are included when `expand_by_sequences=True`.

### 6. Image (pixel) transforms

Applied per DICOM slice after reading:

1. **To tensor:** NumPy → `torch.float32`. 2D → (1, H, W); 3D → (C, H, W); channels trimmed or duplicated to 3 for pretrained backbones.
2. **Normalization:** Min–max to **[0, 1]** per image: `(x - min) / (max - min)` when max > min.
3. **Resize:** Bilinear to **224×224**.
4. **Augmentation (train only):** `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=5)`.

Output per image: **(3, 224, 224)**. Each sample has **N** such tensors (e.g. N=5) stacked as **(N, 3, 224, 224)** plus modality IDs (0=MR, 1=MG).

### 7. Batch-level preprocessing (collate)

- **Gene:** Stacked into `(B, gene_dim)`; placeholder zeros when modality dropout is applied.
- **Clinical:** Zero-padded to max length in batch, then stacked into `(B, max_clinical_len)`.
- **Images:** Each sample has shape `(N_i, 3, 224, 224)`. Batches are padded to the same number of slices `max_N` per sample (zero tensors and zero modality IDs for padding). Result: `(B, max_N, 3, 224, 224)` and `(B, max_N)`.
- **Modality dropout (train only):** With probability `modality_dropout` (e.g. 0.3), each of gene, clinical, and image can be dropped (placeholders). At least one modality is always kept.

### 8. Pairs dataset (image + omics from CSV)

For the simpler paired pipeline (`TCGAPairsDataset`): a CSV lists `case_id`, `label`, `img_path`, `omics_path`. Image is loaded from `img_path` (file or directory; if directory, one image is chosen at random among `.png`, `.jpg`, etc.), converted to RGB, resized to `img_size` (default 224), and converted to tensor with `ToTensor()` (no extra normalization). Omics are loaded from `.npy` and cast to float32. No DICOM or series logic; no modality dropout in the dataset itself.

---

## Architecture

### High-level flow

```
[Gene features]     → RNABERTEncoder        → z_gene (128-d)
[Clinical table]   → FTTransformerEncoder  → z_clinical (128-d)
[DICOM images]     → MRMGHierarchicalImageEncoder → z_image (256-d)
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

`src/oncolearn/data/modalities/` and `src/oncolearn/data/multimodal.py`.

### Components

| File/Modality | Purpose |
|------|--------|
| **tabular** | `TabularDataModule`, dynamically loading API inputs via explicit parser objects (`XenabrowserParser`, etc.) and resolving sequences of numbers. |
| **image** | `ImageDataModule`, leveraging lazy import guards linking standard PyDicom/PIL objects to load imaging frames. Transforms slices. |
| **multimodal** | `MultimodalDataModule`, dynamically generating inner/outer dataset intersections of generic instantiated inputs via Builder pattern. |

### Dataset (imaging-included)

- **TCGAV1Dataset:** One sample per **(patient, series)** when `expand_by_sequences=True`. Each sample: gene vector, clinical numeric vector, and N uniformly sampled DICOM slices (e.g. 5) from that series, transformed to (N, 3, 224, 224). Modality dropout (image/gene/clinical) during training.
- **collate_fn_v1:** Pads image sequences to the same length per batch; stacks gene/clinical; uses placeholders for dropped modalities.

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
| **RNABERTEncoder** | (B, P) gene expression | (B, 128) | Wraps IBM biomed.rna.bert.110m. Backbone can be frozen; projection to 128-d. |
| **FTTransformerEncoder** | (B, clinical_dim) | (B, 128) | TabTransformer (continuous-only); backbone frozen; projection to 128-d. |
| **MRMGHierarchicalImageEncoder** | (B, N, 3, H, W), modality_ids (B, N) | (B, 256) | Pretrained checkpoint (ViT or 3D ViT); per-image features → 256-d; **HierarchicalAttentionPooling** with modality embedding (MR=0, MG=1) over N images. |

### Fusion

- **GatedLateFusionClassifier**
  - **Inputs:** `gene`, `clinical`, `image`, `modality_ids` (optional modality dropout at dataloader).
  - **Heads:** Per-modality stage (and optionally subtype) heads.
  - **Gate:** MLP on concatenated embeddings → mask missing modalities → softmax → weights.
  - **Output:** `stage_logits`; `subtype_logits` if `num_subtype_classes > 0`.

### Supporting modules

- **image_encoder.py:** Loads 2D ViT (e.g. HuggingFace) or 3D ViT from checkpoint; projects to 256-d before hierarchical pooling.
- **vit_3d_wrapper.py**, **vit_block.py** (commented): 3D ViT wrapper and transformer block for 3D checkpoints.

---

## Training & Evaluation

### Training (`src/train.py`)

- **build_model:** Builds RNABERTEncoder, FTTransformerEncoder, MRMGHierarchicalImageEncoder (from checkpoint), then GatedLateFusionClassifier (3 modalities).
- **train_epoch:** One epoch with optional AMP; loss = stage_loss + subtype_lambda × subtype_loss; modality dropout in the dataloader.
- **validate:** Computes loss and metrics (accuracy, balanced accuracy, macro F1) for stage (and subtype if present).
- **main:** Parses config, data paths, fold; builds TCGADataModule (imaging variant); class-weighted CrossEntropy; AdamW + ReduceLROnPlateau; best checkpoint by validation stage F1; early stopping.

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
src/oncolearn/
├── data/                  # API-first dataset pipelines
│   ├── multimodal.py      # Join builder
│   └── modalities/        # Individual feature submodules
│       ├── image/
│       │   ├── dataset.py
│       │   └── loaders/
│       └── tabular/
│           ├── dataset.py
│           └── parsers/
├── modeling/              # PyTorch Lightning logic
│   ├── fusion.py          # GatedLateFusionClassifier
│   ├── gene_encoder.py
│   ├── tab_encoder.py
│   ├── image_encoder.py
│   └── trainer.py
└── registry/              # Centralized resolution point
```

---

## Performance

When trained with the default (or similar) configuration on TCGA-BRCA:

- The pipeline reaches **around 80%** performance on validation (e.g. accuracy or macro F1 for stage or subtype, depending on metric and split).
- Exact numbers depend on:
  - Train/val/test split and fold,
  - Use of PAM50 or BRCA subtype labels,
  - Hyperparameters (learning rate, batch size, modality dropout, subtype_lambda, etc.).

For reproducible results, use the same config, seed, and data paths as in the experiments that reported ~80% performance.

---

## References (in-repo)

- **Cohort/labels:** `data/cohort.py`, `data/labels.py`
- **Fusion:** `src/models/fusion.py`
- **Encoders:** `src/models/gene_encoder.py`, `tab_encoder.py`, `image_encoder.py`
- **Training/eval:** `src/train.py`, `src/eval.py`
