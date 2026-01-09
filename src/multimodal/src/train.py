"""
Training script for TCGA-BRCA V1 and V2 variants.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.datamodule import TCGADataModule
from data.labels import LabelManager
from models import (
    FTTransformerEncoder,
    GatedLateFusionClassifier,
    MRMGHierarchicalImageEncoder,
    RNABERTEncoder,
)
from utils import load_config, save_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def build_model(config: Dict, label_manager: LabelManager, variant: str, device: torch.device = None) -> nn.Module:
    """Build model for specified variant."""

    # Gene encoder 
    use_rna_bert = config['model'].get('use_rna_bert', False)
    gene_input_dim = config['model']['gene_input_dim']
    
    gene_encoder = RNABERTEncoder(
        model_name=config['model'].get('rna_bert_model', 'ibm-research/biomed.rna.bert.110m.mlm.multitask.v1'),
        output_dim=128,
        freeze_backbone=config['model'].get('freeze_rna_bert', True),
        device=str(device) if device else None
    )
    logger.info("Using RNA BERT encoder for gene expression")
    
    # Clinical(Tab) encoder 
    clinical_input_dim = config['model']['clinical_input_dim']
    clinical_encoder = FTTransformerEncoder(
        input_dim=1,  # Per-feature projection
        dim=128,
        num_heads=4,
        depth=2,
        dropout=0.2,
        output_dim=128
    )
    
    # Image encoder (requires checkpoint)
    image_encoder = None
    if variant == 'v1_imaging':
        checkpoint_path = config['model'].get('image_checkpoint_path', None)
        if checkpoint_path is None:
            raise ValueError("image_checkpoint_path must be provided for v1_imaging variant")
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


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion_stage: nn.Module,
    criterion_subtype: nn.Module = None,
    subtype_lambda: float = 0.3,
    device: torch.device = None,
    use_amp: bool = False,
    scaler: GradScaler = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    stage_loss_sum = 0.0
    subtype_loss_sum = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        gene = batch['gene'].to(device) if batch['gene'] is not None else None
        clinical = batch['clinical'].to(device) if batch['clinical'] is not None else None
        image = batch['image'].to(device) if batch['image'] is not None else None
        stage_labels = batch['stage_label'].to(device)
        subtype_labels = batch['subtype_label'].to(device) if batch['subtype_label'] is not None else None
        
        # Modality IDs for images
        modality_ids = batch.get('modality_ids')
        if modality_ids is not None:
            modality_ids = modality_ids.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            # Forward
            outputs = model(gene=gene, clinical=clinical, image=image, modality_ids=modality_ids)
            
            # Stage loss
            stage_logits = outputs['stage_logits']
            loss_stage = criterion_stage(stage_logits, stage_labels)
            
            # Subtype loss (if available)
            loss_subtype = 0.0
            if criterion_subtype is not None and 'subtype_logits' in outputs:
                subtype_logits = outputs['subtype_logits']
                loss_subtype = criterion_subtype(subtype_logits, subtype_labels)
            
            # Total loss
            loss = loss_stage + subtype_lambda * loss_subtype
        
        # Backward
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        stage_loss_sum += loss_stage.item()
        if loss_subtype > 0:
            subtype_loss_sum += loss_subtype.item()
        n_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'stage': f"{loss_stage.item():.4f}",
            'subtype': f"{loss_subtype:.4f}" if loss_subtype > 0 else "N/A"
        })
    
    return {
        'loss': total_loss / n_batches,
        'stage_loss': stage_loss_sum / n_batches,
        'subtype_loss': subtype_loss_sum / n_batches if subtype_loss_sum > 0 else 0.0
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion_stage: nn.Module,
    criterion_subtype: nn.Module = None,
    subtype_lambda: float = 0.3,
    device: torch.device = None
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    stage_loss_sum = 0.0
    subtype_loss_sum = 0.0
    n_batches = 0
    
    all_stage_preds = []
    all_stage_labels = []
    all_subtype_preds = []
    all_subtype_labels = []
    
    for batch in tqdm(dataloader, desc="Validation"):
        # Move to device
        gene = batch['gene'].to(device) if batch['gene'] is not None else None
        clinical = batch['clinical'].to(device) if batch['clinical'] is not None else None
        image = batch['image'].to(device) if batch['image'] is not None else None
        stage_labels = batch['stage_label'].to(device)
        subtype_labels = batch['subtype_label'].to(device) if batch['subtype_label'] is not None else None
        
        modality_ids = batch.get('modality_ids')
        if modality_ids is not None:
            modality_ids = modality_ids.to(device)
        
        # Forward
        outputs = model(gene=gene, clinical=clinical, image=image, modality_ids=modality_ids)
        
        # Losses
        stage_logits = outputs['stage_logits']
        loss_stage = criterion_stage(stage_logits, stage_labels)
        
        loss_subtype = 0.0
        if criterion_subtype is not None and 'subtype_logits' in outputs:
            subtype_logits = outputs['subtype_logits']
            loss_subtype = criterion_subtype(subtype_logits, subtype_labels)
        
        loss = loss_stage + subtype_lambda * loss_subtype
        
        # Accumulate
        total_loss += loss.item()
        stage_loss_sum += loss_stage.item()
        if loss_subtype > 0:
            subtype_loss_sum += loss_subtype.item()
        n_batches += 1
        
        # Predictions
        stage_preds = stage_logits.argmax(dim=-1).cpu().numpy()
        all_stage_preds.extend(stage_preds)
        all_stage_labels.extend(stage_labels.cpu().numpy())
        
        if 'subtype_logits' in outputs:
            subtype_preds = outputs['subtype_logits'].argmax(dim=-1).cpu().numpy()
            all_subtype_preds.extend(subtype_preds)
            all_subtype_labels.extend(subtype_labels.cpu().numpy())
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    
    stage_acc = accuracy_score(all_stage_labels, all_stage_preds)
    stage_bal_acc = balanced_accuracy_score(all_stage_labels, all_stage_preds)
    stage_f1 = f1_score(all_stage_labels, all_stage_preds, average='macro')
    
    metrics = {
        'loss': total_loss / n_batches,
        'stage_loss': stage_loss_sum / n_batches,
        'subtype_loss': subtype_loss_sum / n_batches if subtype_loss_sum > 0 else 0.0,
        'stage_acc': stage_acc,
        'stage_bal_acc': stage_bal_acc,
        'stage_f1': stage_f1,
    }
    
    if all_subtype_preds:
        subtype_acc = accuracy_score(all_subtype_labels, all_subtype_preds)
        subtype_bal_acc = balanced_accuracy_score(all_subtype_labels, all_subtype_preds)
        subtype_f1 = f1_score(all_subtype_labels, all_subtype_preds, average='macro')
        metrics.update({
            'subtype_acc': subtype_acc,
            'subtype_bal_acc': subtype_bal_acc,
            'subtype_f1': subtype_f1,
        })
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train TCGA-BRCA model")
    parser.add_argument('--variant', type=str, required=True, choices=['v1_imaging', 'v2_no_imaging'],
                       help='Training variant')
    parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--fold', type=int, default=0, help='Fold number (0-indexed)')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--pam50_file', type=str, default=None, help='Path to PAM50 labels file (optional)')
    parser.add_argument('--brca_labels_file', type=str, default=None, help='Path to BRCA-data-with-integer-labels.csv (optional, highest priority)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Setup directories
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / 'checkpoints' / args.variant / f'fold_{args.fold}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / 'logs' / args.variant
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(log_dir, args.variant, args.fold)
    writer = SummaryWriter(log_dir / f'tensorboard_fold_{args.fold}')
    
    # Data paths
    data_dir = Path(args.data_dir)
    cohort_index_path = data_dir / 'cohort_index.parquet'
    clinical_table_path = data_dir / 'clinical_table.parquet'
    gene_set_path = data_dir / 'expression_matrix.parquet'  # or gene_set_table.parquet
    dicom_root = str(data_dir / 'tcia') if args.variant == 'v1_imaging' else None
    
    # Create data module
    datamodule = TCGADataModule(
        cohort_index_path=str(cohort_index_path),
        clinical_table_path=str(clinical_table_path),
        gene_set_path=str(gene_set_path),
        dicom_root=dicom_root,
        variant=args.variant,
        n_folds=args.n_folds,
        fold=args.fold,
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4),
        n_dicom_samples=config['training'].get('n_dicom_samples', 10),
        modality_dropout=config['training'].get('modality_dropout', 0.3),
        seed=args.seed,
        pam50_file=args.pam50_file,
        brca_labels_file=args.brca_labels_file
    )
    
    # Get label manager
    label_manager = datamodule.label_manager
    
    # Get actual data dimensions
    actual_gene_dim = datamodule.gene_set_table.shape[1]
    actual_clinical_dim = len(datamodule.clinical_table.select_dtypes(include=[np.number]).columns)
    
    # Update config with actual dimensions
    config['model']['gene_input_dim'] = actual_gene_dim
    config['model']['clinical_input_dim'] = actual_clinical_dim
    
    logger.info(f"Actual gene input dim: {actual_gene_dim}")
    logger.info(f"Actual clinical input dim: {actual_clinical_dim}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    model = build_model(config, label_manager, args.variant, device=device)
    model = model.to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss functions
    train_patient_ids = datamodule.train_patient_ids
    stage_weights = torch.from_numpy(label_manager.get_class_weights_stage(train_patient_ids)).to(device)
    criterion_stage = nn.CrossEntropyLoss(weight=stage_weights)
    
    criterion_subtype = None
    if label_manager.has_subtype:
        subtype_weights = label_manager.get_class_weights_subtype(train_patient_ids)
        if subtype_weights is not None:
            subtype_weights = torch.from_numpy(subtype_weights).to(device)
            criterion_subtype = nn.CrossEntropyLoss(weight=subtype_weights)
    
    # Optimizer
    lr = float(config['training']['lr'])
    weight_decay = float(config['training'].get('weight_decay', 1e-4))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=config['training'].get('patience', 5)
    )
    
    # Mixed precision
    use_amp = config['training'].get('use_amp', False)
    scaler = GradScaler() if use_amp else None
    
    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    max_patience = config['training'].get('early_stopping_patience', 10)
    num_epochs = config['training'].get('num_epochs', 100)
    subtype_lambda = config['training'].get('subtype_lambda', 0.3)
    
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, datamodule.train_dataloader(), optimizer,
            criterion_stage, criterion_subtype, subtype_lambda,
            device, use_amp, scaler
        )
        
        # Validate
        val_metrics = validate(
            model, datamodule.val_dataloader(),
            criterion_stage, criterion_subtype, subtype_lambda,
            device
        )
        
        # Log
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val Stage F1: {val_metrics['stage_f1']:.4f}")
        
        # TensorBoard
        for key, value in train_metrics.items():
            writer.add_scalar(f'Train/{key}', value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1,
            'val_metrics': val_metrics,
        }
        torch.save(checkpoint, checkpoint_dir / 'latest.pt')
        
        # Save best
        if val_metrics['stage_f1'] > best_f1:
            best_f1 = val_metrics['stage_f1']
            torch.save(checkpoint, checkpoint_dir / 'best.pt')
            patience_counter = 0
            logger.info(f"New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        # Scheduler step
        scheduler.step(val_metrics['stage_f1'])
        
        # Early stopping
        if patience_counter >= max_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    writer.close()
    logger.info("Training complete!")


if __name__ == '__main__':
    main()

