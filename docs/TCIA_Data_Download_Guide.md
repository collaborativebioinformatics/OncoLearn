# TCIA Data Download Guide

This guide explains how to download medical imaging data from The Cancer Imaging Archive (TCIA) using the NBIA Data Retriever CLI tool that is pre-installed in the OncoLearn Docker container.

## Overview

The NBIA Data Retriever is a command-line tool for downloading DICOM imaging data from TCIA. Our Docker container comes with version 4.4.3 pre-installed and ready to use.

## TCGA-BRCA Collection

The Cancer Genome Atlas Breast Invasive Carcinoma (TCGA-BRCA) collection is particularly relevant for oncology research:

- **Collection**: TCGA-BRCA
- **Size**: 88.13 GB
- **Subjects**: 139 patients
- **Studies**: 164 imaging studies
- **Series**: 1,877 image series
- **Images**: 230,167 DICOM files
- **Modalities**: MR (Magnetic Resonance), MG (Mammography)
- **License**: CC BY 3.0

**Collection URL**: https://www.cancerimagingarchive.net/collection/tcga-brca/

### Matched Genomic Data

TCGA-BRCA images can be matched with genomic and clinical data from:
- [Genomic Data Commons (GDC)](https://portal.gdc.cancer.gov/projects/TCGA-BRCA)
- Patient identifiers are consistent across TCIA and TCGA databases

### Step 1: Obtain a Manifest File

Before downloading data, you need to create a manifest file (`.tcia` format) that specifies which images to download. We provice some in the `data` folder.

### Step 2: Transfer Manifest to Container

If you downloaded the manifest on your host machine, transfer it to the container:

```bash
# From your host, copy to the workspace volume
docker cp manifest-xxx.tcia <container-name>:/workspace/data/

# Or place it in your workspace directory before starting the container
```

### Step 3: Run the NBIA Data Retriever CLI

#### Basic Usage

**Important**: Use the `--cli` flag to run in command-line mode.

```bash
nbia-data-retriever --cli /path/to/manifest.tcia -d /workspace/data/tcia-downloads
```

#### Common Options

```bash
# Download with a specific output directory
nbia-data-retriever --cli /workspace/data/tcga-brca-manifest.tcia \
  -d /workspace/data/TCGA-BRCA-images

# Download with verbose output
nbia-data-retriever --cli /workspace/data/TCIA_TCGA-BRCA_09-16-2015.tcia \
  -d /workspace/data/TCIA_BRCA \
  -v

# Download with descriptive format and verbose output
nbia-data-retriever --cli /workspace/data/TCIA_TCGA-BRCA_09-16-2015_part1_of_4.tcia \
  -d /workspace/data/TCIA_BRCA \
  -v -f

# Download with credential file (for restricted collections)
nbia-data-retriever --cli /workspace/data/manifest.tcia \
  -d /workspace/data/tcia-downloads \
  -l /workspace/credentials.txt \
  -v
```