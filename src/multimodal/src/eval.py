"""
Evaluation script for TCGA-BRCA V1 and V2 variants.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

from data.datamodule import TCGADataModule
from data.labels import LabelManager
from models import (
    FTTransformerEncoder,
    GatedLateFusionClassifier,
    GeneSetMLPEncoder,
    MRMGHierarchicalImageEncoder,
)
from train import build_model
from utils import load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    label_manager: LabelManager,
    device: torch.device
) -> Dict:
    """Evaluate model and return predictions and metrics."""
    model.eval()
    
    all_patient_ids = []
    all_stage_preds = []
    all_stage_labels = []
    all_subtype_preds = []
    all_subtype_labels = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
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
        
        # Predictions
        stage_preds = outputs['stage_logits'].argmax(dim=-1).cpu().numpy()
        all_stage_preds.extend(stage_preds)
        all_stage_labels.extend(stage_labels.cpu().numpy())
        all_patient_ids.extend(batch['patient_ids'])
        
        if 'subtype_logits' in outputs:
            subtype_preds = outputs['subtype_logits'].argmax(dim=-1).cpu().numpy()
            all_subtype_preds.extend(subtype_preds)
            if subtype_labels is not None:
                all_subtype_labels.extend(subtype_labels.cpu().numpy())
    
    # Compute metrics for stage
    stage_acc = accuracy_score(all_stage_labels, all_stage_preds)
    stage_bal_acc = balanced_accuracy_score(all_stage_labels, all_stage_preds)
    stage_f1 = f1_score(all_stage_labels, all_stage_preds, average='macro')
    stage_cm = confusion_matrix(all_stage_labels, all_stage_preds)
    
    metrics = {
        'stage': {
            'accuracy': stage_acc,
            'balanced_accuracy': stage_bal_acc,
            'macro_f1': stage_f1,
            'confusion_matrix': stage_cm.tolist(),
            'class_names': label_manager.stage_classes,
        }
    }
    
    # Compute metrics for subtype (if available)
    if all_subtype_preds and label_manager.has_subtype:
        subtype_acc = accuracy_score(all_subtype_labels, all_subtype_preds)
        subtype_bal_acc = balanced_accuracy_score(all_subtype_labels, all_subtype_preds)
        subtype_f1 = f1_score(all_subtype_labels, all_subtype_preds, average='macro')
        subtype_cm = confusion_matrix(all_subtype_labels, all_subtype_preds)
        
        metrics['subtype'] = {
            'accuracy': subtype_acc,
            'balanced_accuracy': subtype_bal_acc,
            'macro_f1': subtype_f1,
            'confusion_matrix': subtype_cm.tolist(),
            'class_names': label_manager.subtype_classes,
        }
    
    # Create predictions DataFrame
    preds_df = pd.DataFrame({
        'patient_id': all_patient_ids,
        'stage_true': [label_manager.idx_to_stage[i] for i in all_stage_labels],
        'stage_pred': [label_manager.idx_to_stage[i] for i in all_stage_preds],
    })
    
    if all_subtype_preds and label_manager.has_subtype:
        preds_df['subtype_true'] = [label_manager.idx_to_subtype[i] for i in all_subtype_labels]
        preds_df['subtype_pred'] = [label_manager.idx_to_subtype[i] for i in all_subtype_preds]
    
    return metrics, preds_df


def print_metrics(metrics: Dict):
    """Print evaluation metrics."""
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    # Stage metrics
    stage_metrics = metrics['stage']
    print("\nStage Classification:")
    print(f"  Accuracy: {stage_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {stage_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1: {stage_metrics['macro_f1']:.4f}")
    
    print("\nConfusion Matrix (Stage):")
    cm = np.array(stage_metrics['confusion_matrix'])
    class_names = stage_metrics['class_names']
    print("  " + " ".join([f"{name:>10}" for name in class_names]))
    for i, name in enumerate(class_names):
        print(f"  {name:>10} " + " ".join([f"{cm[i,j]:>10}" for j in range(len(class_names))]))
    
    # Subtype metrics (if available)
    if 'subtype' in metrics:
        subtype_metrics = metrics['subtype']
        print("\nSubtype Classification:")
        print(f"  Accuracy: {subtype_metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {subtype_metrics['balanced_accuracy']:.4f}")
        print(f"  Macro F1: {subtype_metrics['macro_f1']:.4f}")
        
        print("\nConfusion Matrix (Subtype):")
        cm = np.array(subtype_metrics['confusion_matrix'])
        class_names = subtype_metrics['class_names']
        print("  " + " ".join([f"{name:>10}" for name in class_names]))
        for i, name in enumerate(class_names):
            print(f"  {name:>10} " + " ".join([f"{cm[i,j]:>10}" for j in range(len(class_names))]))
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TCGA-BRCA model")
    parser.add_argument('--variant', type=str, required=True, choices=['v1_imaging', 'v2_no_imaging'],
                       help='Training variant')
    parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--fold', type=int, default=0, help='Fold number (0-indexed)')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--pam50_file', type=str, default=None, help='Path to PAM50 labels file (optional)')
    parser.add_argument('--brca_labels_file', type=str, default=None, help='Path to BRCA-data-with-integer-labels.csv (optional, highest priority)')
    parser.add_argument('--test_patients_path', type=str, default=None, help='Path to test_patients.csv (for federated splits, optional)')
    parser.add_argument('--use_test', action='store_true', help='Use test dataset instead of validation dataset for evaluation')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Setup directories
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = out_dir / 'preds' / args.variant
    preds_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / 'logs' / args.variant
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(log_dir, args.variant, args.fold)
    
    # Data paths
    data_dir = Path(args.data_dir)
    cohort_index_path = data_dir / 'cohort_index.parquet'
    clinical_table_path = data_dir / 'clinical_table.parquet'
    gene_set_path = data_dir / 'expression_matrix.parquet'
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
        modality_dropout=0.0,  # No dropout in eval
        seed=args.seed,
        pam50_file=args.pam50_file,
        brca_labels_file=args.brca_labels_file,
        test_patients_path=args.test_patients_path
    )
    
    # Get label manager
    label_manager = datamodule.label_manager
    
    # Build model
    model = build_model(config, label_manager, args.variant)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Determine which dataset to use for evaluation
    if args.use_test:
        test_loader = datamodule.test_dataloader()
        if test_loader is None:
            logger.warning("Test dataset not available. Falling back to validation dataset.")
            eval_loader = datamodule.val_dataloader()
            eval_split = 'val'
        else:
            eval_loader = test_loader
            eval_split = 'test'
            logger.info("Using TEST dataset for evaluation")
    else:
        eval_loader = datamodule.val_dataloader()
        eval_split = 'val'
        logger.info("Using VALIDATION dataset for evaluation")
    
    # Evaluate
    metrics, preds_df = evaluate(model, eval_loader, label_manager, device)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save predictions
    preds_file = preds_dir / f'preds_{eval_split}_fold_{args.fold}.csv'
    preds_df.to_csv(preds_file, index=False)
    logger.info(f"Saved predictions to {preds_file}")
    
    # Save metrics
    import json
    metrics_file = preds_dir / f'metrics_{eval_split}_fold_{args.fold}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")


if __name__ == '__main__':
    main()

