#!/bin/bash

# Script to download TCGA SKCM (Melanoma) data and extract to data/GDCdata directory

# Exit on error
set -e

# Define target directory
TARGET_DIR="data/GDCdata"

# Create directory if it doesn't exist
echo "Creating target directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

# Define URLs
FPKM_URL="https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-SKCM.star_fpkm-uq.tsv.gz"
PHENOTYPE_URL="https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-SKCM.clinical.tsv.gz"

# Download FPKM data
echo "Downloading TCGA-SKCM FPKM data..."
wget -P "$TARGET_DIR" "$FPKM_URL"

# Download phenotype data (includes cancer subtypes)
echo "Downloading TCGA-SKCM phenotype data..."
wget -P "$TARGET_DIR" "$PHENOTYPE_URL"

# Unzip FPKM data
echo "Extracting FPKM data..."
gunzip -f "$TARGET_DIR/TCGA-SKCM.star_fpkm-uq.tsv.gz"

# Unzip phenotype data
echo "Extracting phenotype data..."
gunzip -f "$TARGET_DIR/TCGA-SKCM.clinical.tsv.gz"

echo "Download and extraction complete!"
echo "Files saved to: $TARGET_DIR"
ls -lh "$TARGET_DIR"/TCGA-SKCM*.tsv
