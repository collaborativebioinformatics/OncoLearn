"""
NVFLARE Federated Learning - AWS EC2 Deployment
Modified for distributed deployment across EC2 instances with local data storage
"""

import os

from nvflare import FedJob
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.xgboost.tree_based.bagging_aggregator import XGBBaggingAggregator
from nvflare.app_opt.xgboost.tree_based.model_persistor import XGBModelPersistor
from nvflare.app_opt.xgboost.tree_based.shareable_generator import (
    XGBModelShareableGenerator,
)
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

if __name__ == "__main__":
    print("=" * 80)
    print("Creating NVFLARE Federated Learning Job Configuration")
    print("=" * 80)

    # ========================================
    # Configuration
    # ========================================
    BASE_DIR = "/home/ec2-user/nvflare"
    LOCAL_DATA_DIR = f"{BASE_DIR}/data"
    LOCAL_OUTPUT_DIR = f"{BASE_DIR}/outputs"

    JOB_EXPORT_DIR = "/tmp/nvflare_job"

    # Federated Learning Configuration
    n_clients = 2
    num_rounds = 100
    train_script = "scripts/local-XGBoost.py"

    # Training parameters
    random_state = 0
    test_size = 0.2
    num_client_bagging = n_clients

    # IMPORTANT:
    # If site-*.csv already contains a header row, set SKIP_ROWS=1
    # If site-*.csv has NO header row (pure data), set SKIP_ROWS=None
    SKIP_ROWS = 1

    # If labels are strings / not 0..K-1, you can enable this
    ENCODE_LABELS = False

    # Validate input files exist (fail early with clear error)
    for i in range(1, n_clients + 1):
        data_file = f"{LOCAL_DATA_DIR}/site-{i}.csv"
        header_file = f"{LOCAL_DATA_DIR}/site-{i}_header.csv"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Missing data file: {data_file}")
        if not os.path.exists(header_file):
            raise FileNotFoundError(f"Missing header file: {header_file}")
        print(f"[OK] Found {data_file} and {header_file}")

    # Script arguments passed to local-XGBoost.py
    script_args = (
    f"--data_root_dir {LOCAL_DATA_DIR} "
    f"--output_dir {LOCAL_OUTPUT_DIR} "
    f"--random_state {random_state} "
    f"--test_size {test_size} "
    f"--num_client_bagging {num_client_bagging} "
    f"--skip_rows 1 "
    )

    if SKIP_ROWS is not None:
        script_args += f"--skip_rows {SKIP_ROWS} "

    if ENCODE_LABELS:
        script_args += "--encode_labels "

    print("\nScript args:\n", script_args)

    aggregator_id = "aggregator"
    persistor_id = "persistor"
    shareable_generator_id = "shareable_generator"

    print("\nCreating job components...")
    job = FedJob("xgboost_tree_brca")

    # Server components
    job.to(XGBModelPersistor(), "server", id=persistor_id)
    job.to(XGBModelShareableGenerator(), "server", id=shareable_generator_id)
    job.to(XGBBaggingAggregator(), "server", id=aggregator_id)

    ctrl = ScatterAndGather(
        min_clients=n_clients,
        num_rounds=num_rounds,
        start_round=0,
        wait_time_after_min_received=10,
        aggregator_id=aggregator_id,
        persistor_id=persistor_id,
        shareable_generator_id=shareable_generator_id,
        train_task_name="train",
        train_timeout=600,
        allow_empty_global_weights=True,
    )
    job.to(ctrl, "server")

    # Client runners
    for i in range(1, n_clients + 1):
        runner = ScriptRunner(
            script=train_script,
            script_args=script_args,
            framework=FrameworkType.RAW,
        )
        job.to(runner, f"site-{i}")
        print(f"[OK] Added site-{i}")

    # Export Job
    print("\nExporting job configuration...")
    os.makedirs(JOB_EXPORT_DIR, exist_ok=True)
    job.export_job(JOB_EXPORT_DIR)
    print(f"[OK] Job exported to {JOB_EXPORT_DIR}")
