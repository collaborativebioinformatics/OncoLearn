# TCGA Data Download Guide

This guide explains how to download RNA-Seq and clinical data from The Cancer Genome Atlas (TCGA) for use with OncoLearn.

## Overview

OncoLearn provides convenient download scripts for fetching TCGA data from the [UCSC Xena Browser](https://xenabrowser.net/datapages/). The scripts download:
- **RNA-Seq data**: STAR FPKM-UQ normalized gene expression data
- **Clinical data**: Patient phenotype and clinical information

## Quick Start

### Download All Cohorts

To download all available cancer cohorts at once:

```bash
bash ./scripts/data/download_all_tcga.sh
```

This will download and extract all cohort data to `data/GDCdata/`.

### Available Cancer Cohorts

The following TCGA cancer cohorts are supported:

| Cohort Code | Cancer Type | Script |
|------------|-------------|--------|
| TCGA-BRCA | Breast Invasive Carcinoma | `download_tcga_brca.sh` |
| TCGA-COAD | Colon Adenocarcinoma | `download_tcga_coad.sh` |
| TCGA-LAML | Acute Myeloid Leukemia | `download_tcga_laml.sh` |
| TCGA-LUAD | Lung Adenocarcinoma | `download_tcga_luad.sh` |
| TCGA-LUSC | Lung Squamous Cell Carcinoma | `download_tcga_lusc.sh` |
| TCGA-MESO | Mesothelioma | `download_tcga_meso.sh` |
| TCGA-SKCM | Skin Cutaneous Melanoma | `download_tcga_skcm.sh` |

### Download Individual Cohorts

To download specific cancer cohorts, use the individual scripts:

```bash
# Lung Adenocarcinoma
bash ./scripts/data/download_tcga_luad.sh

# Lung Squamous Cell Carcinoma
bash ./scripts/data/download_tcga_lusc.sh

# Breast Cancer
bash ./scripts/data/download_tcga_brca.sh

# Colon Cancer
bash ./scripts/data/download_tcga_coad.sh

# Melanoma
bash ./scripts/data/download_tcga_skcm.sh

# Mesothelioma
bash ./scripts/data/download_tcga_meso.sh

# Acute Myeloid Leukemia
bash ./scripts/data/download_tcga_laml.sh
```

### Data Size Considerations

Each cohort varies in size:
- **Small cohorts** (e.g., LAML): ~50-100 MB
- **Medium cohorts** (e.g., COAD, LUSC): 100-300 MB
- **Large cohorts** (e.g., BRCA, LUAD): 300-600 MB

**Total download size for all cohorts**: Approximately 2-3 GB

Ensure you have sufficient disk space before downloading.

## Data Storage Structure

All downloaded data is organized in the `data/GDCdata/` directory:

```
data/
└── GDCdata/
    ├── TCGA-BRCA.clinical.tsv
    ├── TCGA-BRCA.star_fpkm-uq.tsv
    ├── TCGA-COAD.clinical.tsv
    ├── TCGA-COAD.star_fpkm-uq.tsv
    ├── TCGA-LAML.clinical.tsv
    ├── TCGA-LAML.star_fpkm-uq.tsv
    └── ...
```

Each cohort has two files:
- `TCGA-{COHORT}.clinical.tsv` - Clinical and phenotype data
- `TCGA-{COHORT}.star_fpkm-uq.tsv` - Gene expression data

## Data Processing

After downloading, you can use the preprocessing notebooks to merge and prepare the data:

1. **Merge clinical and expression data**:
   - Notebook: [`notebooks/data/preprocess_merge_cohort_data.ipynb`](../notebooks/data/preprocess_merge_cohort_data.ipynb)
   - Output: Merged files in `data/processed/`

2. **Prepare multimodal data**:
   - Notebook: [`notebooks/data/preprocess_multimodal_data_fusion.ipynb`](../notebooks/data/preprocess_multimodal_data_fusion.ipynb)

3. **Exploratory data analysis**:
   - Notebook: [`notebooks/data/eda_cohorts.ipynb`](../notebooks/data/eda_cohorts.ipynb)
