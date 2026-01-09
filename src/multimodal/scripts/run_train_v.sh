#!/bin/bash
# Training script for V1 (imaging-present) variant

set -e

# Configuration
VARIANT="v1_imaging"
CONFIG="configs/v1_imaging.yaml"
DATA_DIR="data/processed"  # Update with your data directory
OUT_DIR="outputs"
N_FOLDS=5
SEED=42
PAM50_FILE="${PAM50_FILE:-data/pam50.txt}"  # PAM50 labels file (optional)
BRCA_LABELS_FILE="${BRCA_LABELS_FILE:-data/BRCA-data-with-integer-labels.csv}"  # BRCA labels file (highest priority)

# Create output directory
mkdir -p ${OUT_DIR}

# Train each fold
for FOLD in $(seq 0 $((N_FOLDS-1))); do
    echo "Training fold ${FOLD}..."
    python3 src/train.py \
        --variant ${VARIANT} \
        --config ${CONFIG} \
        --data_dir ${DATA_DIR} \
        --out_dir ${OUT_DIR} \
        --fold ${FOLD} \
        --n_folds ${N_FOLDS} \
        --seed ${SEED} \
        --pam50_file ${PAM50_FILE} \
        --brca_labels_file ${BRCA_LABELS_FILE}
done

echo "Training complete for all folds!"

