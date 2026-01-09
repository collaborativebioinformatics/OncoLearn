"""
NVFlare client training script for federated learning.
Uses current data structure and model architecture.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import nvflare.client as flare

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.cohort import load_clinical_table, load_gene_set_table
from src.data.datamodule import TCGAV1Dataset, TCGAV2Dataset, collate_fn_v1, collate_fn_v2, TCGADataModule
from src.data.labels import LabelManager
from src.models.fusion import GatedLateFusionClassifier
from src.models.gene_encoder import GeneSetMLPEncoder, RNABERTEncoder
from src.models.tab_encoder import FTTransformerEncoder
from src.models.image_encoder import MRMGHierarchicalImageEncoder
from src.utils import load_config, set_seed

# Use print instead of logger for direct output
class SimpleLogger:
    """Simple logger that uses print for direct output."""
    def info(self, *args, **kwargs):
        print(*args, **kwargs, flush=True)
    
    def warning(self, *args, **kwargs):
        print("WARNING:", *args, **kwargs, flush=True)
    
    def error(self, *args, **kwargs):
        print("ERROR:", *args, **kwargs, flush=True)
    
    def debug(self, *args, **kwargs):
        print("DEBUG:", *args, **kwargs, flush=True)

logger = SimpleLogger()

# Global round counter for logging
_round_counter = 0


def get_gene_encoder_mlp_params(gene_encoder):
    """
    Extract MLP parameters from gene encoder.
    For GeneSetMLPEncoder: all parameters
    For RNABERTEncoder: only projection layer parameters
    """
    params = {}
    if isinstance(gene_encoder, GeneSetMLPEncoder):
        # All parameters are MLP
        for name, param in gene_encoder.named_parameters():
            params[f"gene_encoder.{name}"] = param
    elif isinstance(gene_encoder, RNABERTEncoder):
        # Only projection layer
        for name, param in gene_encoder.projection.named_parameters():
            params[f"gene_encoder.projection.{name}"] = param
    return params


def get_gate_network_params(model):
    """Extract gate network parameters from fusion model."""
    params = {}
    for name, param in model.gate_network.named_parameters():
        params[f"fusion.gate_network.{name}"] = param
    return params


def get_shared_params(model):
    """
    Extract shared parameters: Gene Encoder MLP + Gate Network
    Returns a dict with prefixed parameter names for server communication.
    """
    shared_params = {}
    
    # Gene Encoder MLP parameters
    gene_mlp_params = get_gene_encoder_mlp_params(model.gene_encoder)
    shared_params.update(gene_mlp_params)
    
    # Gate Network parameters
    gate_params = get_gate_network_params(model)
    shared_params.update(gate_params)
    
    return shared_params


def load_shared_params(model, shared_state_dict):
    """
    Load shared parameters into model.
    Handles both old format (gene_encoder only) and new format (gene_encoder + gate_network).
    """
    # Separate gene encoder and gate network parameters
    gene_params = {}
    gate_params = {}
    
    for key, value in shared_state_dict.items():
        if key.startswith("gene_encoder."):
            # Remove "gene_encoder." prefix for loading
            gene_key = key[len("gene_encoder."):]
            gene_params[gene_key] = value
        elif key.startswith("fusion.gate_network."):
            # Remove "fusion.gate_network." prefix for loading
            gate_key = key[len("fusion.gate_network."):]
            gate_params[gate_key] = value
        elif key.startswith("gate_network."):
            # Handle old format without fusion prefix
            gate_key = key[len("gate_network."):]
            gate_params[gate_key] = value
        else:
            # Assume it's gene encoder parameter (backward compatibility)
            gene_params[key] = value
    
    # Load gene encoder MLP parameters
    if gene_params:
        if isinstance(model.gene_encoder, GeneSetMLPEncoder):
            model.gene_encoder.load_state_dict(gene_params, strict=False)
        elif isinstance(model.gene_encoder, RNABERTEncoder):
            # Only load projection layer
            model.gene_encoder.projection.load_state_dict(gene_params, strict=False)
    
    # Load gate network parameters
    if gate_params:
        model.gate_network.load_state_dict(gate_params, strict=False)


def freeze_all_encoders(model):
    """Freeze all encoder parameters (Gene, Clinical, Image)."""
    # Freeze Gene Encoder
    for param in model.gene_encoder.parameters():
        param.requires_grad = False
    
    # Freeze Clinical Encoder
    for param in model.clinical_encoder.parameters():
        param.requires_grad = False
    
    # Freeze Image Encoder (if exists)
    if model.image_encoder is not None:
        for param in model.image_encoder.parameters():
            param.requires_grad = False
    
    # Freeze per-modality heads
    for param in model.gene_stage_head.parameters():
        param.requires_grad = False
    for param in model.clinical_stage_head.parameters():
        param.requires_grad = False
    if model.has_image:
        for param in model.image_stage_head.parameters():
            param.requires_grad = False
    
    if model.has_subtype:
        for param in model.gene_subtype_head.parameters():
            param.requires_grad = False
        for param in model.clinical_subtype_head.parameters():
            param.requires_grad = False
        if model.has_image:
            for param in model.image_subtype_head.parameters():
                param.requires_grad = False


def setup_trainable_params(model):
    """
    Set up trainable parameters: only Gene Encoder MLP + Gate Network.
    Returns list of trainable parameters for optimizer.
    """
    # First freeze all encoders
    freeze_all_encoders(model)
    
    # Unfreeze Gene Encoder MLP
    if isinstance(model.gene_encoder, GeneSetMLPEncoder):
        # All parameters are MLP
        for param in model.gene_encoder.parameters():
            param.requires_grad = True
    elif isinstance(model.gene_encoder, RNABERTEncoder):
        # Only projection layer
        for param in model.gene_encoder.projection.parameters():
            param.requires_grad = True
    
    # Unfreeze Gate Network
    for param in model.gate_network.parameters():
        param.requires_grad = True
    
    # Collect trainable parameters
    trainable_params = []
    trainable_params.extend([p for p in model.gene_encoder.parameters() if p.requires_grad])
    trainable_params.extend([p for p in model.gate_network.parameters() if p.requires_grad])
    
    return trainable_params


def build_model(config: dict, label_manager, variant: str, device: torch.device):
    """Build model for federated learning."""
    # Gene encoder - use RNA BERT if specified, otherwise MLP
    use_rna_bert = config['model'].get('use_rna_bert', False)  # Default: False (MLP encoder)
    gene_input_dim = config['model']['gene_input_dim']
    
    if use_rna_bert:
        gene_encoder = RNABERTEncoder(
            model_name=config['model'].get('rna_bert_model', 'ibm-research/biomed.rna.bert.110m.mlm.multitask.v1'),
            output_dim=128,
            freeze_backbone=config['model'].get('freeze_rna_bert', True),
            device=str(device) if device else None
        )
        logger.info("Using RNA BERT encoder for gene expression")
    else:
        gene_encoder = GeneSetMLPEncoder(
            input_dim=gene_input_dim,
            hidden_dim=256,
            output_dim=128,
            dropout=0.3
        )
        logger.info("Using MLP encoder for gene expression")
    
    # Clinical encoder
    clinical_input_dim = config['model']['clinical_input_dim']
    clinical_encoder = FTTransformerEncoder(
        input_dim=1,  # Per-feature projection
        dim=128,
        num_heads=4,
        depth=2,
        dropout=0.2,
        output_dim=128
    )
    
    # Image encoder (V1 only)
    image_encoder = None
    if variant == 'v1_imaging':
        checkpoint_path = config['model'].get('image_checkpoint_path', None)
        image_encoder = MRMGHierarchicalImageEncoder(
            checkpoint_path=checkpoint_path,
            freeze_backbone=config['model'].get('freeze_backbone', True),
            output_dim=256
        )
    
    # Fusion model
    num_stage_classes = len(label_manager.stage_classes)
    num_subtype_classes = len(label_manager.subtype_classes) if label_manager.has_subtype else 0
    
    model = GatedLateFusionClassifier(
        gene_encoder=gene_encoder,
        clinical_encoder=clinical_encoder,
        image_encoder=image_encoder,
        gene_dim=128,
        clinical_dim=128,
        image_dim=256 if image_encoder else 0,
        num_stage_classes=num_stage_classes,
        num_subtype_classes=num_subtype_classes,
        dropout=0.2
    )
    
    return model


def accuracy_from_logits(logits, y):
    """Compute accuracy from logits."""
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()


@torch.no_grad()
def evaluate_model(model, dataloader, label_manager, device, criterion_stage, criterion_subtype, subtype_lambda):
    """Evaluate model on validation/test dataset."""
    model.eval()
    
    total_loss = 0.0
    stage_loss_sum = 0.0
    stage_correct = 0
    stage_total = 0
    subtype_correct = 0
    subtype_total = 0
    
    for batch in dataloader:
        # Move to device
        gene = batch['gene'].to(device) if batch['gene'] is not None else None
        clinical = batch['clinical'].to(device) if batch['clinical'] is not None else None
        image = batch['image'].to(device) if batch['image'] is not None else None
        stage_labels = batch['stage_label'].to(device)
        subtype_labels = batch['subtype_label'].to(device) if batch['subtype_label'] is not None else None
        
        modality_ids = batch.get('modality_ids')
        if modality_ids is not None:
            modality_ids = modality_ids.to(device)
        
        # Forward pass
        outputs = model(gene=gene, clinical=clinical, image=image, modality_ids=modality_ids)
        
        # Stage loss and accuracy
        stage_logits = outputs['stage_logits']
        loss_stage = criterion_stage(stage_logits, stage_labels)
        stage_preds = torch.argmax(stage_logits, dim=1)
        stage_correct += (stage_preds == stage_labels).sum().item()
        stage_total += stage_labels.size(0)
        
        # Subtype loss and accuracy (if available)
        loss_subtype = 0.0
        if criterion_subtype is not None and 'subtype_logits' in outputs and subtype_labels is not None:
            subtype_logits = outputs['subtype_logits']
            loss_subtype = criterion_subtype(subtype_logits, subtype_labels)
            subtype_preds = torch.argmax(subtype_logits, dim=1)
            subtype_correct += (subtype_preds == subtype_labels).sum().item()
            subtype_total += subtype_labels.size(0)
        
        # Total loss: Subtype is main task, stage is auxiliary
        if loss_subtype > 0:
            loss = loss_subtype + (1.0 - subtype_lambda) * loss_stage if subtype_lambda < 1.0 else loss_subtype
        else:
            loss = loss_stage
        
        total_loss += loss.item()
        stage_loss_sum += loss_stage.item()
    
    # Main accuracy is subtype if available, otherwise stage
    main_acc = subtype_correct / subtype_total if subtype_total > 0 else (stage_correct / stage_total if stage_total > 0 else 0.0)
    
    metrics = {
        'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0.0,
        'stage_loss': stage_loss_sum / len(dataloader) if len(dataloader) > 0 else 0.0,
        'stage_acc': stage_correct / stage_total if stage_total > 0 else 0.0,
        'subtype_acc': subtype_correct / subtype_total if subtype_total > 0 else 0.0,
        'main_acc': main_acc,  # Main task accuracy (subtype if available)
    }
    
    return metrics


def main():
    global _round_counter  # Declare global variable at the start of the function
    
    parser = argparse.ArgumentParser(description="NVFlare client training")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing processed data")
    parser.add_argument("--split_dir", type=str, required=True, help="Directory containing site-{i} subdirectories")
    parser.add_argument("--site_id", type=int, default=None, help="Site ID (1-based). If not provided, will be extracted from NVFlare client name.")
    parser.add_argument("--config", type=str, default="configs/v1_imaging.yaml", help="Config file path")
    parser.add_argument("--variant", type=str, default="v1_imaging", choices=["v1_imaging", "v2_no_imaging"])
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per round (overrides config if provided)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config if provided)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config if provided)")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers (overrides config if provided)")
    parser.add_argument("--pam50_file", type=str, default=None, help="PAM50 labels file")
    parser.add_argument("--brca_labels_file", type=str, default=None, help="BRCA labels file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get site_id from NVFlare client name if not provided
    if args.site_id is None:
        import os
        import re
        
        # Method 1: Try to get from environment variables
        client_name = (
            os.environ.get('FL_CLIENT_NAME') or
            os.environ.get('NVFLARE_CLIENT_NAME') or
            os.environ.get('CLIENT_NAME') or
            os.environ.get('FL_SITE_NAME') or
            ''
        )
        
        if client_name and client_name.startswith('site-'):
            try:
                args.site_id = int(client_name.split('-')[1])
                logger.info(f"Extracted site_id={args.site_id} from client name: {client_name}")
            except (ValueError, IndexError):
                pass
        
        # Method 2: Try to extract from current working directory path
        if args.site_id is None:
            cwd = os.getcwd()
            match = re.search(r'site-(\d+)', cwd)
            if match:
                args.site_id = int(match.group(1))
                logger.info(f"Extracted site_id={args.site_id} from path: {cwd}")
        
        # Method 3: Try to extract from script file path (__file__)
        if args.site_id is None:
            try:
                script_path = __file__
                match = re.search(r'site-(\d+)', script_path)
                if match:
                    args.site_id = int(match.group(1))
                    logger.info(f"Extracted site_id={args.site_id} from script path: {script_path}")
            except:
                pass
        
        # Method 4: Try to extract from sys.argv (if called with site name in path)
        if args.site_id is None:
            import sys
            for arg in sys.argv:
                match = re.search(r'site-(\d+)', arg)
                if match:
                    args.site_id = int(match.group(1))
                    logger.info(f"Extracted site_id={args.site_id} from argument: {arg}")
                    break
        
        if args.site_id is None:
            raise ValueError("--site_id is required. Could not determine automatically from environment, path, script path, or arguments.")
    
    # Load config
    config = load_config(args.config)
    training_config = config.get('training', {})
    
    # Override with command line args only if explicitly provided
    # This allows yaml config to be the default, with optional CLI overrides
    if args.batch_size is not None:
        training_config['batch_size'] = args.batch_size
    if args.lr is not None:
        training_config['lr'] = args.lr
    if args.num_workers is not None:
        training_config['num_workers'] = args.num_workers
    # Use epochs from config if not provided via CLI
    if args.epochs is None:
        # Use num_epochs from config, default to 1 if not in config
        args.epochs = training_config.get('num_epochs', 1)
    
    # Load data for this site
    site_dir = Path(args.split_dir) / f"site-{args.site_id}"
    if not site_dir.exists():
        raise ValueError(f"Site directory not found: {site_dir}")
    
    logger.info(f"Loading data from site directory: {site_dir}")
    
    # Load tables
    cohort_index_path = site_dir / "cohort_index.parquet"
    clinical_table_path = site_dir / "clinical_table.parquet"
    gene_set_table_path = site_dir / "gene_set_table.parquet"
    
    if not cohort_index_path.exists():
        raise FileNotFoundError(f"Cohort index not found: {cohort_index_path}")
    if not clinical_table_path.exists():
        raise FileNotFoundError(f"Clinical table not found: {clinical_table_path}")
    if not gene_set_table_path.exists():
        raise FileNotFoundError(f"Gene set table not found: {gene_set_table_path}")
    
    import pandas as pd
    cohort_df = pd.read_parquet(cohort_index_path)
    clinical_table = load_clinical_table(str(clinical_table_path))
    gene_set_table = load_gene_set_table(str(gene_set_table_path))
    
    # Get actual dimensions
    gene_input_dim = gene_set_table.shape[1]
    clinical_input_dim = clinical_table.select_dtypes(include=[np.number]).shape[1]
    
    # Update config with actual dimensions
    config['model']['gene_input_dim'] = gene_input_dim
    config['model']['clinical_input_dim'] = clinical_input_dim
    
    logger.info(f"Gene input dim: {gene_input_dim}, Clinical input dim: {clinical_input_dim}")
    
    # Load labels
    label_manager = LabelManager(
        clinical_table=clinical_table,
        subtype_lambda=training_config.get('subtype_lambda', 0.3),
        pam50_file=args.pam50_file,
        brca_labels_file=args.brca_labels_file
    )
    
    # DICOM root path - try to find it relative to data_root
    dicom_root = Path(args.data_root).parent / "tcia"
    if not dicom_root.exists():
        # Try absolute path from cohort_index
        if 'dicom_series' in cohort_df.columns and len(cohort_df) > 0:
            # Extract path from first patient's DICOM series
            import json
            first_series = cohort_df.iloc[0].get('dicom_series', '{}')
            if first_series:
                try:
                    series_dict = json.loads(first_series)
                    if series_dict:
                        # Get first example path
                        first_study = list(series_dict.values())[0]
                        first_series_info = list(first_study.values())[0]
                        example_path = first_series_info.get('example_paths', [None])[0]
                        if example_path:
                            dicom_root = Path(example_path).parent.parent.parent.parent
                except:
                    pass
    
    # Check for train/val/test patient lists
    train_patients_path = site_dir / "train_patients.csv"
    val_patients_path = site_dir / "val_patients.csv"
    test_patients_path = site_dir / "test_patients.csv"
    
    # Load patient splits if available
    train_patient_ids = None
    val_patient_ids = None
    test_patient_ids = None
    
    if train_patients_path.exists():
        train_patient_ids = set(pd.read_csv(train_patients_path)['patient_id'].tolist())
        logger.info(f"Loaded train patients: {len(train_patient_ids)}")
    
    if val_patients_path.exists():
        val_patient_ids = set(pd.read_csv(val_patients_path)['patient_id'].tolist())
        logger.info(f"Loaded val patients: {len(val_patient_ids)}")
    
    if test_patients_path.exists():
        test_patient_ids = set(pd.read_csv(test_patients_path)['patient_id'].tolist())
        logger.info(f"Loaded test patients: {len(test_patient_ids)}")
    
    # Build train dataset
    if train_patient_ids is not None:
        train_cohort_df = cohort_df[cohort_df['patient_id'].isin(train_patient_ids)]
    else:
        train_cohort_df = cohort_df  # Use all if no split file
    
    if args.variant == 'v1_imaging':
        train_dataset = TCGAV1Dataset(
            cohort_df=train_cohort_df,
            clinical_table=clinical_table,
            gene_set_table=gene_set_table,
            label_manager=label_manager,
            dicom_root=str(dicom_root),
            dicom_transform=None,  # Will use default
            n_dicom_samples=training_config.get('n_dicom_samples', 5),
            modality_dropout=training_config.get('modality_dropout', 0.3),
            mode='train',
            expand_by_sequences=True
        )
        collate_fn = collate_fn_v1
    else:
        train_dataset = TCGAV2Dataset(
            cohort_df=train_cohort_df,
            clinical_table=clinical_table,
            gene_set_table=gene_set_table,
            label_manager=label_manager,
            modality_dropout=training_config.get('modality_dropout', 0.3),
            mode='train'
        )
        collate_fn = collate_fn_v2
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config.get('num_workers', 0),
        collate_fn=collate_fn,
        drop_last=True
    )
    logger.info(f"Created training dataloader with {len(train_dataset)} samples (sequences)")
    
    # Build validation dataset
    val_dataloader = None
    if val_patient_ids is not None and len(val_patient_ids) > 0:
        val_cohort_df = cohort_df[cohort_df['patient_id'].isin(val_patient_ids)]
        if args.variant == 'v1_imaging':
            val_dataset = TCGAV1Dataset(
                cohort_df=val_cohort_df,
                clinical_table=clinical_table,
                gene_set_table=gene_set_table,
                label_manager=label_manager,
                dicom_root=str(dicom_root),
                dicom_transform=None,
                n_dicom_samples=training_config.get('n_dicom_samples', 5),
                modality_dropout=0.0,  # No dropout in eval
                mode='val',
                expand_by_sequences=True
            )
        else:
            val_dataset = TCGAV2Dataset(
                cohort_df=val_cohort_df,
                clinical_table=clinical_table,
                gene_set_table=gene_set_table,
                label_manager=label_manager,
                modality_dropout=0.0,
                mode='val'
            )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=training_config.get('num_workers', 0),
            collate_fn=collate_fn,
            drop_last=False
        )
        logger.info(f"Created validation dataloader with {len(val_dataset)} samples (sequences)")
    
    # Build test dataset
    test_dataloader = None
    if test_patient_ids is not None and len(test_patient_ids) > 0:
        test_cohort_df = cohort_df[cohort_df['patient_id'].isin(test_patient_ids)]
        if args.variant == 'v1_imaging':
            test_dataset = TCGAV1Dataset(
                cohort_df=test_cohort_df,
                clinical_table=clinical_table,
                gene_set_table=gene_set_table,
                label_manager=label_manager,
                dicom_root=str(dicom_root),
                dicom_transform=None,
                n_dicom_samples=training_config.get('n_dicom_samples', 5),
                modality_dropout=0.0,  # No dropout in eval
                mode='test',
                expand_by_sequences=True
            )
        else:
            test_dataset = TCGAV2Dataset(
                cohort_df=test_cohort_df,
                clinical_table=clinical_table,
                gene_set_table=gene_set_table,
                label_manager=label_manager,
                modality_dropout=0.0,
                mode='test'
            )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=training_config.get('num_workers', 0),
            collate_fn=collate_fn,
            drop_last=False
        )
        logger.info(f"Created test dataloader with {len(test_dataset)} samples (sequences)")
    
    # Build model
    model = build_model(config, label_manager, args.variant, device)
    model.to(device)
    
    # For federated learning, we share only:
    # - Gene Encoder MLP (or projection layer for RNA BERT)
    # - Gate Network
    # All encoders (Gene, Clinical, Image) are frozen
    # Per-modality heads are frozen
    
    # Setup trainable parameters: only MLP + Gate Network
    trainable_params = setup_trainable_params(model)
    
    logger.info(f"Trainable parameters:")
    logger.info(f"  - Gene Encoder MLP: {sum(p.numel() for p in model.gene_encoder.parameters() if p.requires_grad):,}")
    logger.info(f"  - Gate Network: {sum(p.numel() for p in model.gate_network.parameters() if p.requires_grad):,}")
    logger.info(f"  - Total trainable: {sum(p.numel() for p in trainable_params):,}")
    
    # Loss functions
    criterion_stage = nn.CrossEntropyLoss()
    criterion_subtype = None
    if label_manager.has_subtype:
        criterion_subtype = nn.CrossEntropyLoss()
    
    # Optimizer - only train MLP + Gate Network
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(training_config.get('lr', 1e-4)),
        weight_decay=float(training_config.get('weight_decay', 1e-4))
    )
    
    # Initialize NVFlare
    flare.init()
    
    logger.info("Starting federated learning rounds...")
    
    # Federated learning loop
    while flare.is_running():
        # Receive global model weights (gene encoder only)
        in_model = flare.receive()
        
        if in_model and in_model.params:
            # Load global shared weights (Gene Encoder MLP + Gate Network)
            # Handles backward compatibility with old format (gene_encoder only)
            load_shared_params(model, in_model.params)
            logger.info(f"=== Round {_round_counter + 1}: Received AGGREGATED global model ===")
            logger.info(f"  Shared parameters: Gene Encoder MLP + Gate Network")
            # Log parameter stats to verify aggregation is working
            first_param_name = next(iter(in_model.params.keys()))
            first_param = in_model.params[first_param_name]
            logger.info(f"  Parameter '{first_param_name}': shape={first_param.shape}, mean={first_param.mean().item():.6f}, std={first_param.std().item():.6f}")
            
            # Evaluate aggregated global model BEFORE local training
            logger.info("--- Evaluating AGGREGATED GLOBAL MODEL (before local training) ---")
            model.eval()
            global_val_metrics = None
            global_test_metrics = None
            subtype_lambda = training_config.get('subtype_lambda', 1.0)
            
            if val_dataloader is not None:
                global_val_metrics = evaluate_model(
                    model, val_dataloader, label_manager, device,
                    criterion_stage, criterion_subtype, subtype_lambda
                )
                logger.info(f"Global Val metrics: loss={global_val_metrics['loss']:.4f}, stage_acc={global_val_metrics['stage_acc']:.4f}, "
                           f"subtype_acc={global_val_metrics['subtype_acc']:.4f}")
            
            if test_dataloader is not None:
                global_test_metrics = evaluate_model(
                    model, test_dataloader, label_manager, device,
                    criterion_stage, criterion_subtype, subtype_lambda
                )
                logger.info(f"Global Test metrics: loss={global_test_metrics['loss']:.4f}, stage_acc={global_test_metrics['stage_acc']:.4f}, "
                           f"subtype_acc={global_test_metrics['subtype_acc']:.4f}")
        else:
            logger.info("No global model received, using local initialization")
        
        # Local training
        model.train()
        steps = 0
        total_loss = 0.0
        stage_loss_sum = 0.0
        last_acc = 0.0
        t0 = time.time()
        
        for epoch in range(args.epochs):
            for batch in train_dataloader:
                # Move to device
                gene = batch['gene'].to(device) if batch['gene'] is not None else None
                clinical = batch['clinical'].to(device) if batch['clinical'] is not None else None
                image = batch['image'].to(device) if batch['image'] is not None else None
                stage_labels = batch['stage_label'].to(device)
                subtype_labels = batch['subtype_label'].to(device) if batch['subtype_label'] is not None else None
                
                modality_ids = batch.get('modality_ids')
                if modality_ids is not None:
                    modality_ids = modality_ids.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(gene=gene, clinical=clinical, image=image, modality_ids=modality_ids)
                
                # Stage loss
                stage_logits = outputs['stage_logits']
                loss_stage = criterion_stage(stage_logits, stage_labels)
                
                # Subtype loss (if available) - MAIN TASK
                loss_subtype = 0.0
                if criterion_subtype is not None and 'subtype_logits' in outputs and subtype_labels is not None:
                    subtype_logits = outputs['subtype_logits']
                    loss_subtype = criterion_subtype(subtype_logits, subtype_labels)
                
                # Total loss: Subtype is main task, stage is auxiliary
                subtype_lambda = training_config.get('subtype_lambda', 1.0)
                if loss_subtype > 0:
                    # Subtype is main task
                    loss = loss_subtype + (1.0 - subtype_lambda) * loss_stage if subtype_lambda < 1.0 else loss_subtype
                else:
                    # Fallback to stage if no subtype
                    loss = loss_stage
                
                # Backward
                loss.backward()
                optimizer.step()
                
                steps += 1
                total_loss += loss.item()
                stage_loss_sum += loss_stage.item()
                
                # Accuracy: Use subtype if available, otherwise stage
                if criterion_subtype is not None and 'subtype_logits' in outputs and subtype_labels is not None:
                    last_acc = accuracy_from_logits(subtype_logits.detach(), subtype_labels.detach())
                else:
                    last_acc = accuracy_from_logits(stage_logits.detach(), stage_labels.detach())
        
        # Evaluation on validation/test sets
        subtype_lambda = training_config.get('subtype_lambda', 0.3)
        val_metrics = None
        test_metrics = None
        
        if val_dataloader is not None:
            val_metrics = evaluate_model(
                model, val_dataloader, label_manager, device,
                criterion_stage, criterion_subtype, subtype_lambda
            )
            logger.info(f"Val metrics: loss={val_metrics['loss']:.4f}, stage_acc={val_metrics['stage_acc']:.4f}, "
                       f"subtype_acc={val_metrics['subtype_acc']:.4f}")
        
        if test_dataloader is not None:
            test_metrics = evaluate_model(
                model, test_dataloader, label_manager, device,
                criterion_stage, criterion_subtype, subtype_lambda
            )
            logger.info(f"Test metrics: loss={test_metrics['loss']:.4f}, stage_acc={test_metrics['stage_acc']:.4f}, "
                       f"subtype_acc={test_metrics['subtype_acc']:.4f}")
        
        # Calculate main accuracy (subtype if available, otherwise stage)
        main_val_acc = None
        main_test_acc = None
        if val_metrics is not None:
            main_val_acc = val_metrics.get('main_acc', val_metrics['subtype_acc'] if val_metrics['subtype_acc'] > 0 else val_metrics['stage_acc'])
        if test_metrics is not None:
            main_test_acc = test_metrics.get('main_acc', test_metrics['subtype_acc'] if test_metrics['subtype_acc'] > 0 else test_metrics['stage_acc'])
        
        # Use validation accuracy as main accuracy for NVFlare (if available, otherwise use train accuracy)
        main_accuracy = main_val_acc if main_val_acc is not None else last_acc
        
        # Get number of training samples for weighted aggregation
        num_train_samples = len(train_dataset) if train_dataset else steps * args.batch_size
        
        # Send shared parameters (Gene Encoder MLP + Gate Network) back to server for aggregation
        out_params = get_shared_params(model)
        out_params = {k: v.cpu() for k, v in out_params.items()}
        logger.info(f"Sending shared parameters to server for aggregation (Round {_round_counter + 1})")
        logger.info(f"  Shared components: Gene Encoder MLP + Gate Network")
        logger.info(f"  Number of parameter groups: {len(out_params)}, Training samples: {num_train_samples}")
        
        meta = {
            "NUM_STEPS_CURRENT_ROUND": steps,
            "num_samples": int(num_train_samples),  # Number of training samples for weighted FedAvg
            "accuracy": float(main_accuracy),  # Main accuracy for NVFlare IntimeModelSelector
            "train_accuracy": float(last_acc),
            "train_stage_accuracy": float(last_acc),  # Alias for compatibility
            "train_loss": float(total_loss / steps) if steps > 0 else 0.0,
            "train_stage_loss": float(stage_loss_sum / steps) if steps > 0 else 0.0,
            "train_time_sec": float(time.time() - t0),
        }
        
        # Add validation metrics (main task: subtype)
        if val_metrics is not None:
            meta.update({
                "val_accuracy": float(main_val_acc),  # Main validation accuracy (subtype)
                "val_stage_accuracy": float(val_metrics['stage_acc']),  # Stage accuracy
                "val_loss": float(val_metrics['loss']),
                "val_stage_loss": float(val_metrics['stage_loss']),
                "val_stage_acc": float(val_metrics['stage_acc']),
                "val_subtype_acc": float(val_metrics['subtype_acc']),
            })
        
        # Add test metrics (main task: subtype)
        if test_metrics is not None:
            meta.update({
                "test_accuracy": float(main_test_acc),  # Main test accuracy (subtype)
                "test_stage_accuracy": float(test_metrics['stage_acc']),  # Stage accuracy
                "test_loss": float(test_metrics['loss']),
                "test_stage_loss": float(test_metrics['stage_loss']),
                "test_stage_acc": float(test_metrics['stage_acc']),
                "test_subtype_acc": float(test_metrics['subtype_acc']),
            })
        
        # Track round number manually
        _round_counter += 1
        
        # Print formatted metrics (main task: subtype)
        round_info = f"Round {_round_counter}"
        logger.info("="*80)
        logger.info(f"{round_info} Complete - Site {args.site_id}")
        logger.info("-"*80)
        
        # Global model metrics (before local training)
        if 'global_val_metrics' in locals() and global_val_metrics is not None:
            global_main_val_acc = global_val_metrics.get('main_acc', global_val_metrics['subtype_acc'] if global_val_metrics['subtype_acc'] > 0 else global_val_metrics['stage_acc'])
            logger.info(f"Global Model Metrics (Before Local Training):")
            logger.info(f"  Global Val Accuracy (Subtype): {global_main_val_acc:.4f}")
            logger.info(f"  Global Val Loss: {global_val_metrics['loss']:.4f}")
            logger.info(f"  Global Val Stage Accuracy: {global_val_metrics['stage_acc']:.4f}")
            logger.info(f"  Global Val Subtype Accuracy: {global_val_metrics['subtype_acc']:.4f}")
        
        if 'global_test_metrics' in locals() and global_test_metrics is not None:
            global_main_test_acc = global_test_metrics.get('main_acc', global_test_metrics['subtype_acc'] if global_test_metrics['subtype_acc'] > 0 else global_test_metrics['stage_acc'])
            logger.info(f"  Global Test Accuracy (Subtype): {global_main_test_acc:.4f}")
            logger.info(f"  Global Test Loss: {global_test_metrics['loss']:.4f}")
            logger.info(f"  Global Test Stage Accuracy: {global_test_metrics['stage_acc']:.4f}")
            logger.info(f"  Global Test Subtype Accuracy: {global_test_metrics['subtype_acc']:.4f}")
        
        logger.info("-"*80)
        logger.info(f"Training Metrics:")
        logger.info(f"  Steps: {steps}")
        logger.info(f"  Train Accuracy (Subtype): {last_acc:.4f}")  # Main task: Subtype
        logger.info(f"  Train Loss: {meta['train_loss']:.4f}")
        logger.info(f"  Train Stage Loss: {meta['train_stage_loss']:.4f}")
        logger.info(f"  Time: {meta['train_time_sec']:.2f}s")
        
        if val_metrics:
            main_val_acc = val_metrics.get('main_acc', val_metrics['subtype_acc'] if val_metrics['subtype_acc'] > 0 else val_metrics['stage_acc'])
            logger.info(f"Local Model Metrics (After Local Training):")
            logger.info(f"  Val Accuracy (Subtype): {main_val_acc:.4f}")  # Main task: Subtype
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"  Val Stage Loss: {val_metrics['stage_loss']:.4f}")
            logger.info(f"  Val Stage Accuracy: {val_metrics['stage_acc']:.4f}")
            logger.info(f"  Val Subtype Accuracy: {val_metrics['subtype_acc']:.4f}")
        
        if test_metrics:
            main_test_acc = test_metrics.get('main_acc', test_metrics['subtype_acc'] if test_metrics['subtype_acc'] > 0 else test_metrics['stage_acc'])
            logger.info(f"  Test Accuracy (Subtype): {main_test_acc:.4f}")  # Main task: Subtype
            logger.info(f"  Test Loss: {test_metrics['loss']:.4f}")
            logger.info(f"  Test Stage Loss: {test_metrics['stage_loss']:.4f}")
            logger.info(f"  Test Stage Accuracy: {test_metrics['stage_acc']:.4f}")
            logger.info(f"  Test Subtype Accuracy: {test_metrics['subtype_acc']:.4f}")
        
        logger.info("="*80)
        
        # Also log a summary line for easy parsing (main task: subtype)
        summary_parts = [f"Round {_round_counter}", f"Site {args.site_id}"]
        
        # Global model metrics (before local training)
        if 'global_val_metrics' in locals() and global_val_metrics is not None:
            global_main_val_acc = global_val_metrics.get('main_acc', global_val_metrics['subtype_acc'] if global_val_metrics['subtype_acc'] > 0 else global_val_metrics['stage_acc'])
            summary_parts.append(f"Global_Val_Acc={global_main_val_acc:.4f}")
        if 'global_test_metrics' in locals() and global_test_metrics is not None:
            global_main_test_acc = global_test_metrics.get('main_acc', global_test_metrics['subtype_acc'] if global_test_metrics['subtype_acc'] > 0 else global_test_metrics['stage_acc'])
            summary_parts.append(f"Global_Test_Acc={global_main_test_acc:.4f}")
        
        # Local model metrics (after local training)
        summary_parts.append(f"Train_Acc={last_acc:.4f}")
        if val_metrics:
            main_val_acc = val_metrics.get('main_acc', val_metrics['subtype_acc'] if val_metrics['subtype_acc'] > 0 else val_metrics['stage_acc'])
            summary_parts.append(f"Local_Val_Acc={main_val_acc:.4f}")
        if test_metrics:
            main_test_acc = test_metrics.get('main_acc', test_metrics['subtype_acc'] if test_metrics['subtype_acc'] > 0 else test_metrics['stage_acc'])
            summary_parts.append(f"Local_Test_Acc={main_test_acc:.4f}")
        
        logger.info("SUMMARY: " + " | ".join(summary_parts))
        
        flare.send(flare.FLModel(params=out_params, meta=meta))


if __name__ == "__main__":
    main()
