"""
NVFlare Job Configuration for Cancer Subtyping Federated Learning.

This script creates a federated learning job using NVFlare's Job API.
"""

from nvflare import FedJob, ScriptExecutor
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import (
    InTimeAccumulateWeightedAggregator,
)
from nvflare.app_common.workflows.fedavg import FedAvg


def create_federated_job(
    job_name: str = "cancer_subtyping_federated",
    num_clients: int = 3,
    num_rounds: int = 50,
    clients_per_round: int = 3,
    script_path: str = "train_nvflare.py",
):
    """
    Create NVFlare federated learning job.

    Args:
        job_name: Name of the federated learning job
        num_clients: Number of client sites
        num_rounds: Number of federated learning rounds
        clients_per_round: Number of clients to train per round
        script_path: Path to the training script

    Returns:
        FedJob object
    """
    # Create job
    job = FedJob(name=job_name)

    # Create FedAvg workflow for server
    aggregator = InTimeAccumulateWeightedAggregator(
        expected_data_kind="WEIGHTS",
    )

    fedavg_controller = FedAvg(
        num_clients=num_clients,
        num_rounds=num_rounds,
        aggregator=aggregator,
    )

    # Add controller to server
    job.to_server(fedavg_controller)

    # Add script executor to each client
    for i in range(num_clients):
        client_name = f"site-{i+1}"

        # Create executor that runs the training script
        executor = ScriptExecutor(
            task_script_path=script_path,
            task_script_args="",  # Args can be added here or in script
        )

        # Add executor to client
        job.to(executor, target=client_name)

    return job


def main():
    """Create and export the job."""
    import argparse

    parser = argparse.ArgumentParser(description="Create NVFlare job")
    parser.add_argument("--job_name", type=str, default="cancer_subtyping_fl",
                        help="Job name")
    parser.add_argument("--num_clients", type=int, default=3,
                        help="Number of client sites")
    parser.add_argument("--num_rounds", type=int, default=50,
                        help="Number of federated rounds")
    parser.add_argument("--output_dir", type=str, default="nvflare_jobs",
                        help="Output directory for job")

    args = parser.parse_args()

    print(f"Creating NVFlare job: {args.job_name}")
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds: {args.num_rounds}")

    # Create job
    job = create_federated_job(
        job_name=args.job_name,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        clients_per_round=args.num_clients,
    )

    # Export job
    job.export_job(args.output_dir)

    print(f"[OK] Job created and exported to {args.output_dir}/{args.job_name}")
    print("\nTo submit the job:")
    print("  1. Start NVFlare server and clients")
    print(
        f"  2. Submit job: nvflare job submit {args.output_dir}/{args.job_name}")


if __name__ == "__main__":
    main()
