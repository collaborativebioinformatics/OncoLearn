#!/bin/bash

# Script to download TCGA BRCA data and extract to data/GDCdata directory

# Exit on error
set -e

# Define target directory
TARGET_DIR="data/GDCdata"

# Create directory if it doesn't exist
echo "Creating target directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

# Define URLs
FPKM_URL="https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.star_fpkm-uq.tsv.gz"
CLINICAL_URL="https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.clinical.tsv.gz"

# Download FPKM data
echo "Downloading TCGA-BRCA FPKM data..."
wget -P "$TARGET_DIR" "$FPKM_URL"

# Download clinical data
echo "Downloading TCGA-BRCA clinical data..."
wget -P "$TARGET_DIR" "$CLINICAL_URL"

# Unzip FPKM data
echo "Extracting FPKM data..."
gunzip -f "$TARGET_DIR/TCGA-BRCA.star_fpkm-uq.tsv.gz"

# Unzip clinical data
echo "Extracting clinical data..."
gunzip -f "$TARGET_DIR/TCGA-BRCA.clinical.tsv.gz"

echo "Download and extraction complete!"
echo "Files saved to: $TARGET_DIR"
ls -lh "$TARGET_DIR"/*.tsv
