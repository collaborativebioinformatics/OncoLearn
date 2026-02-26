#!/bin/bash
# NVFlare Federated Learning execution script
# Multiple clients collaborate with distributed data

set -e

echo "=== NVFlare Federated Learning Execution ==="

# Configuration
N_CLIENTS=${N_CLIENTS:-5}           # Number of clients
NUM_ROUNDS=${NUM_ROUNDS:-10}        # Federated learning rounds
SPLIT_DIR=${SPLIT_DIR:-data/federated_splits}  # Federated data split directory
DATA_ROOT=${DATA_ROOT:-data/processed}  # Data root path
JOB_ROOT=${JOB_ROOT:-outputs/nvflare_jobs}  # NVFlare job config storage path
WORKSPACE=${WORKSPACE:-outputs/nvflare_workspace}  # NVFlare workspace path

# Check if federated split is prepared
if [ ! -d "$SPLIT_DIR" ]; then
    echo "ERROR: Federated data split directory not found: $SPLIT_DIR"
    echo ""
    echo "Please prepare federated data split first:"
    echo "  bash scripts/prepare_federated_data.sh"
    exit 1
fi

# Check for site-*.csv files
SITE_FILES=$(ls ${SPLIT_DIR}/site-*.csv 2>/dev/null | wc -l)
if [ "$SITE_FILES" -eq 0 ]; then
    echo "ERROR: site-*.csv files not found in: $SPLIT_DIR"
    exit 1
fi

echo "Found clients: $SITE_FILES"
echo "Configured clients: $N_CLIENTS"

if [ "$SITE_FILES" -lt "$N_CLIENTS" ]; then
    echo "WARNING: Found clients ($SITE_FILES) is less than configured clients ($N_CLIENTS)."
    echo "Adjusting configured clients to $SITE_FILES."
    N_CLIENTS=$SITE_FILES
fi

# Check gene expression dimension (from gene_set_table)
if [ -f "${DATA_ROOT}/gene_set_table.parquet" ]; then
    OMICS_DIM=$(python3 << EOF
import pandas as pd
df = pd.read_parquet('${DATA_ROOT}/gene_set_table.parquet')
print(df.shape[1])
EOF
)
elif [ -f "${DATA_ROOT}/expression_matrix.parquet" ]; then
    OMICS_DIM=$(python3 << EOF
import pandas as pd
df = pd.read_parquet('${DATA_ROOT}/expression_matrix.parquet')
print(df.shape[1])
EOF
)
else
    echo "WARNING: gene_set_table.parquet not found. Using default value 60660."
    OMICS_DIM=60660
fi

echo "Gene expression dimension: $OMICS_DIM"

# Create and run NVFlare job
echo ""
echo "=== NVFlare Job Creation and Execution ==="
echo "Number of clients: $N_CLIENTS"
echo "Number of rounds: $NUM_ROUNDS"
echo "Split directory: $SPLIT_DIR"
echo "Data root: $DATA_ROOT"
echo ""

python3 job.py \
    --n_clients $N_CLIENTS \
    --num_rounds $NUM_ROUNDS \
    --split_dir $SPLIT_DIR \
    --data_root $DATA_ROOT \
    --omics_dim $OMICS_DIM \
    --job_root $JOB_ROOT \
    --workspace $WORKSPACE

echo ""
echo "=== Federated Learning Complete ==="
echo "Results saved to:"
echo "  - Job config: $JOB_ROOT"
echo "  - Workspace: $WORKSPACE"

