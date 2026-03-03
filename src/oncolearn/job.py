import argparse
import os

from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner

import torch.nn as nn
from oncolearn.modeling.encoders import RNABERTEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_root", type=str, default="/tmp/nvflare/jobs/job_config")
    ap.add_argument("--workspace", type=str, default="/tmp/nvflare/jobs/workdir")
    ap.add_argument("--n_clients", type=int, default=2)
    ap.add_argument("--num_rounds", type=int, default=2)
    ap.add_argument("--split_dir", type=str, required=True, help="Directory containing site-*.csv split files")
    ap.add_argument("--data_root", type=str, required=True, help="Root path that makes img/omics paths resolvable on each site")
    ap.add_argument("--omics_dim", type=int, required=True, help="Dimension of omics vectors (.npy) used by seq encoder")
    ap.add_argument("--export_only", action="store_true", help="If set, export job config only (no simulator run).")
    args = ap.parse_args()

    n_clients = args.n_clients
    num_rounds = args.num_rounds

    # 1) Create job
    job = FedJob(name="tcga_multimodal_omics_encoder_fedavg", min_clients=n_clients)

    # 2) Server workflow: FedAvg
    controller = FedAvg(num_clients=n_clients, num_rounds=num_rounds)
    job.to_server(controller)

    # 3) Initial global model: Gene Encoder MLP + Gate Network
    # Gene Encoder MLP
    init_gene_encoder = GeneSetMLPEncoder(
        input_dim=args.omics_dim,
        hidden_dim=256,
        output_dim=128,
        dropout=0.3
    )

    # Gate Network (3-modality mode: gene + clinical + image)
    # Gate input: 128 (gene) + 128 (clinical) + 256 (image) = 512
    gate_network = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 3)  # 3 modalities
    )

    # Wrapper model to share both components
    class SharedModel(nn.Module):
        def __init__(self, gene_encoder, gate_network):
            super().__init__()
            self.gene_encoder = gene_encoder
            self.gate_network = gate_network

        def state_dict(self):
            state = {}
            gene_state = self.gene_encoder.state_dict()
            gate_state = self.gate_network.state_dict()

            for key, value in gene_state.items():
                state[f"gene_encoder.{key}"] = value

            for key, value in gate_state.items():
                state[f"fusion.gate_network.{key}"] = value

            return state

        def load_state_dict(self, state_dict, strict=True):
            gene_state = {}
            gate_state = {}

            for key, value in state_dict.items():
                if key.startswith("gene_encoder."):
                    gene_key = key[len("gene_encoder."):]
                    gene_state[gene_key] = value
                elif key.startswith("fusion.gate_network."):
                    gate_key = key[len("fusion.gate_network."):]
                    gate_state[gate_key] = value
                elif key.startswith("gate_network."):
                    gate_key = key[len("gate_network."):]
                    gate_state[gate_key] = value
                else:
                    gene_state[key] = value

            if gene_state:
                self.gene_encoder.load_state_dict(gene_state, strict=False)
            if gate_state:
                self.gate_network.load_state_dict(gate_state, strict=False)

    init_shared_model = SharedModel(init_gene_encoder, gate_network)
    job.to_server(PTModel(init_shared_model))

    # 4) Optional model selection widget
    job.to_server(IntimeModelSelector(key_metric="accuracy"))

    # 5) Client training script (Client API)
    train_script = "src/oncolearn/train_nvflare.py"
    split_dir_abs = os.path.abspath(args.split_dir)
    data_root_abs = os.path.abspath(args.data_root)

    script_args = (
        f"--split_dir {split_dir_abs} "
        f"--variant v1_imaging "
        f"--epochs 1 "
        f"--batch_size 16"
    )
    runner = ScriptRunner(script=train_script, script_args=script_args)
    job.to_clients(runner)

    os.makedirs(args.job_root, exist_ok=True)
    os.makedirs(args.workspace, exist_ok=True)

    job.export_job(args.job_root)
    print(f"[OK] Exported job to: {args.job_root}")

    if not args.export_only:
        job.simulator_run(args.workspace, n_clients=n_clients)
        print(f"[OK] Simulator run finished. Workspace: {args.workspace}")

if __name__ == "__main__":
    main()
