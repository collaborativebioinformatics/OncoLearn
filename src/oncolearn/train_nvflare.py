import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import nvflare.client as flare

from oncolearn.trainer import OncoTrainer
from oncolearn.modeling.gene_encoder import GeneSetMLPEncoder, RNABERTEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_trainable_params(model):
    """Enable gradients only for parts of the model being federated."""
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze Gene Encoder MLP / Projection (assuming it exists in Tabular pipeline)
    # The new pipeline uses FTTransformer for tabular data (clinical) and possibly MLP for gene
    # In V2, we have gene and clinical encoders if we have multiple tabular modalities
    # Let's unfreeze the gate network and whatever tabular encoders we want federated
    if hasattr(model, 'gate_network'):
        for param in model.gate_network.parameters():
            param.requires_grad = True

    if hasattr(model, 'gene_encoder'):
        for param in model.gene_encoder.parameters():
            param.requires_grad = True
            
    # Collect trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return trainable_params


def load_shared_params(model, shared_state_dict):
    """Load received FL weights strictly into the submodules."""
    gene_params = {}
    gate_params = {}
    for key, value in shared_state_dict.items():
        if key.startswith("gene_encoder."):
            gene_params[key.replace("gene_encoder.", "")] = value
        elif key.startswith("fusion.gate_network."):
            gate_params[key.replace("fusion.gate_network.", "")] = value
        else:
            gene_params[key] = value

    if gene_params and hasattr(model, 'gene_encoder'):
        model.gene_encoder.load_state_dict(gene_params, strict=False)
    if gate_params and hasattr(model, 'gate_network'):
        model.gate_network.load_state_dict(gate_params, strict=False)


def get_shared_params(model):
    """Extract updated metrics to return to the NVFlare FL server."""
    shared = {}
    if hasattr(model, 'gene_encoder'):
        for name, param in model.gene_encoder.named_parameters():
            if param.requires_grad:
                shared[f"gene_encoder.{name}"] = param.cpu()
                
    if hasattr(model, 'gate_network'):
        for name, param in model.gate_network.named_parameters():
            if param.requires_grad:
                shared[f"fusion.gate_network.{name}"] = param.cpu()
    return shared


def main():
    parser = argparse.ArgumentParser(description="NVFlare client training pipeline")
    parser.add_argument("--split_dir", type=str, required=True, help="Directory containing site-{i} subsets (Ignored in OncoLearn V2 refactor due to config)")
    parser.add_argument("--variant", type=str, default="v2_no_imaging", choices=["v1_imaging", "v2_no_imaging"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    flare.init()
    logger.info("FLARE client initialized.")
    
    # Establish local pipeline using OncoTrainer
    modalities = ['image', 'tabular'] if args.variant == 'v1_imaging' else ['tabular']
    trainer = OncoTrainer(
        modalities=modalities,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        variant=args.variant
    )
    
    # Needs explicit datamodule setup for NVFlare FL execution stepping
    trainer.datamodule.setup(stage='fit')
    
    # Inject trainability configurations
    trainable_params = setup_trainable_params(trainer.model)
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)

    while flare.is_running():
        # Receive global payload
        in_model = flare.receive()
        if in_model and in_model.params:
            logger.info("Received aggregate global model.")
            load_shared_params(trainer.model, in_model.params)
            
            # Epoch simulation execution locally!
            logger.info("Running local federation epoch...")
            trainer.model.train()
            for _ in range(args.epochs):
                train_metrics = trainer.train_epoch(
                    loader=trainer.datamodule.train_dataloader(),
                    optimizer=optimizer,
                    scaler=torch.cuda.amp.GradScaler() if trainer.device.type == 'cuda' else None,
                    criterion_stage=nn.CrossEntropyLoss().to(trainer.device),
                    criterion_subtype=None,
                    use_amp=(trainer.device.type == 'cuda')
                )
                logger.info(f"Local train metrics: {train_metrics['loss']:.4f}")
            
            val_metrics = trainer.validate_epoch(
                loader=trainer.datamodule.val_dataloader(),
                criterion_stage=nn.CrossEntropyLoss().to(trainer.device),
                criterion_subtype=None
            )
            
            logger.info(f"Local validation accuracy: {val_metrics['stage_acc']:.4f}")
            
            # Send updated parameters back
            out_model = flare.FLModel(
                params=get_shared_params(trainer.model),
                metrics={"accuracy": val_metrics['stage_acc']}
            )
            flare.send(out_model)
        else:
            logger.info("Bypassed execution window due to missing Global Network parameters.")

if __name__ == "__main__":
    main()
