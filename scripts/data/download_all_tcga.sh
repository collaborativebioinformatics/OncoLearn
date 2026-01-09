#!/bin/bash

# Master script to download all TCGA cancer cohort data

# Exit on error
set -e

echo "=========================================="
echo "Downloading all TCGA cancer cohort data"
echo "=========================================="

# Make all scripts executable
chmod +x ./scripts/data/*.sh

# Breast Cancer
echo ""
echo "========== BRCA (Breast Cancer) =========="
bash ./scripts/data/download_tcga_brca.sh

# Lung Cancer cohorts
echo ""
echo "========== LUAD (Lung Adenocarcinoma) =========="
bash ./scripts/data/download_tcga_luad.sh

echo ""
echo "========== LUSC (Lung Squamous Cell Carcinoma) =========="
bash ./scripts/data/download_tcga_lusc.sh

echo ""
echo "========== SKCM (Melanoma) =========="
bash ./scripts/data/download_tcga_skcm.sh

echo ""
echo "========== MESO (Mesothelioma) =========="
bash ./scripts/data/download_tcga_meso.sh

# Colorectal Cancer
echo ""
echo "========== COAD (Colon Cancer) =========="
bash ./scripts/data/download_tcga_coad.sh

# Leukemia
echo ""
echo "========== LAML (Acute Myeloid Leukemia) =========="
bash ./scripts/data/download_tcga_laml.sh

echo ""
echo "=========================================="
echo "All downloads complete!"
echo "=========================================="
echo "Files are stored in: data/GDCdata/"
ls -lh data/GDCdata/*.tsv
