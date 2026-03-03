import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from oncolearn.registry import get_model
from oncolearn.data.multimodal import MultimodalDataModule
from oncolearn.data.modalities.tabular.dataset import TabularDataModule
from oncolearn.data.modalities.image.dataset import ImageDataModule
from oncolearn.modeling.fusion import GatedLateFusionClassifier
from oncolearn.modeling.gene_encoder import RNABERTEncoder
from oncolearn.modeling.tab_encoder import FTTransformerEncoder
from oncolearn.modeling.image_encoder import MRMGHierarchicalImageEncoder

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(config: Dict, variant: str = 'v1_imaging', device: torch.device = None) -> nn.Module:
    """Builds RNABERTEncoder, FTTransformerEncoder, MRMGHierarchicalImageEncoder, then GatedLateFusionClassifier."""
    model_config = config.get('model', {})
    
    # Gene encoder 
    gene_encoder = RNABERTEncoder(
        model_name=model_config.get('rna_bert_model', 'ibm-research/biomed.rna.bert.110m.mlm.multitask.v1'),
        output_dim=128,
        freeze_backbone=model_config.get('freeze_rna_bert', True),
        device=str(device) if device else None
    )
    
    # Clinical encoder 
    clinical_encoder = FTTransformerEncoder(
        input_dim=1,
        dim=128,
        num_heads=4,
        depth=2,
        dropout=0.2,
        output_dim=128
    )
    
    # Image encoder 
    image_encoder = None
    if variant == 'v1_imaging':
        checkpoint_path = model_config.get('image_checkpoint_path', None)
        if checkpoint_path is None:
            logger.warning("No image_checkpoint_path provided, using scratch image encoder.")
            # For robustness, we allow it without checkpoint if needed, but MULTIMODAL.md says "from checkpoint"
            # It expects one to be provided if fully running.
            pass
        if checkpoint_path is not None:
            image_encoder = MRMGHierarchicalImageEncoder(
                checkpoint_path=checkpoint_path,
                freeze_backbone=model_config.get('freeze_backbone', True),
                output_dim=256
            )
    
    model = GatedLateFusionClassifier(
        gene_encoder=gene_encoder,
        clinical_encoder=clinical_encoder,
        image_encoder=image_encoder,
        gene_dim=128,
        clinical_dim=128,
        image_dim=256 if image_encoder else 0,
        num_stage_classes=model_config.get('num_stage_classes', 5),
        num_subtype_classes=model_config.get('num_subtype_classes', 0),
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
    scaler: GradScaler = None,
    device_type: str = 'cpu',
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    stage_loss_sum = 0.0
    subtype_loss_sum = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Map MultimodalDataModule batches. Assume 'tabular' -> gene, 'clinical' -> clinical if provided
        gene = batch.get('tabular', batch.get('gene'))
        if gene is not None:
            gene = gene.to(device)
            
        clinical = batch.get('clinical')
        if clinical is not None:
            clinical = clinical.to(device)
            
        image = batch.get('image')
        if image is not None:
            image = image.to(device)
            
        # Get labels
        stage_labels = batch.get('label')
        if stage_labels is None:
            stage_labels = batch.get('stage_label')
        if stage_labels is not None:
            stage_labels = stage_labels.to(device)
            
        subtype_labels = batch.get('subtype_label')
        if subtype_labels is not None:
            subtype_labels = subtype_labels.to(device)
            
        modality_ids = batch.get('modality_ids')
        if modality_ids is not None:
            modality_ids = modality_ids.to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type=device_type, enabled=use_amp):
            outputs = model(gene=gene, clinical=clinical, image=image, modality_ids=modality_ids)
            
            stage_logits = outputs['stage_logits']
            loss_stage = criterion_stage(stage_logits, stage_labels)
            
            loss_subtype = 0.0
            if criterion_subtype is not None and 'subtype_logits' in outputs and subtype_labels is not None:
                subtype_logits = outputs['subtype_logits']
                loss_subtype = criterion_subtype(subtype_logits, subtype_labels)
            
            loss = loss_stage + subtype_lambda * loss_subtype
        
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        stage_loss_sum += loss_stage.item()
        if type(loss_subtype) == torch.Tensor:
            subtype_loss_sum += loss_subtype.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'stage': f"{loss_stage.item():.4f}"})
    
    return {
        'loss': total_loss / max(1, n_batches),
        'stage_loss': stage_loss_sum / max(1, n_batches),
        'subtype_loss': subtype_loss_sum / max(1, n_batches) if subtype_loss_sum > 0 else 0.0
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
    """Validate model and return metrics."""
    model.eval()
    total_loss = 0.0
    stage_loss_sum = 0.0
    subtype_loss_sum = 0.0
    n_batches = 0
    
    all_stage_preds = []
    all_stage_labels = []
    
    for batch in tqdm(dataloader, desc="Validation"):
        gene = batch.get('tabular', batch.get('gene'))
        if gene is not None: gene = gene.to(device)
        clinical = batch.get('clinical')
        if clinical is not None: clinical = clinical.to(device)
        image = batch.get('image')
        if image is not None: image = image.to(device)
            
        stage_labels = batch.get('label', batch.get('stage_label'))
        if stage_labels is not None: stage_labels = stage_labels.to(device)
        subtype_labels = batch.get('subtype_label')
        if subtype_labels is not None: subtype_labels = subtype_labels.to(device)
        
        modality_ids = batch.get('modality_ids')
        if modality_ids is not None: modality_ids = modality_ids.to(device)
        
        outputs = model(gene=gene, clinical=clinical, image=image, modality_ids=modality_ids)
        
        stage_logits = outputs['stage_logits']
        loss_stage = criterion_stage(stage_logits, stage_labels)
        
        loss_subtype = 0.0
        if criterion_subtype is not None and 'subtype_logits' in outputs and subtype_labels is not None:
            subtype_logits = outputs['subtype_logits']
            loss_subtype = criterion_subtype(subtype_logits, subtype_labels)
        
        loss = loss_stage + subtype_lambda * loss_subtype
        total_loss += loss.item()
        stage_loss_sum += loss_stage.item()
        if type(loss_subtype) == torch.Tensor:
            subtype_loss_sum += loss_subtype.item()
        n_batches += 1
        
        stage_preds = stage_logits.argmax(dim=-1).cpu().numpy()
        all_stage_preds.extend(stage_preds)
        all_stage_labels.extend(stage_labels.cpu().numpy())
    
    stage_acc = accuracy_score(all_stage_labels, all_stage_preds) if all_stage_labels else 0.0
    stage_bal_acc = balanced_accuracy_score(all_stage_labels, all_stage_preds) if all_stage_labels else 0.0
    stage_f1 = f1_score(all_stage_labels, all_stage_preds, average='macro') if all_stage_labels else 0.0
    
    return {
        'loss': total_loss / max(1, n_batches),
        'stage_loss': stage_loss_sum / max(1, n_batches),
        'subtype_loss': subtype_loss_sum / max(1, n_batches) if subtype_loss_sum > 0 else 0.0,
        'stage_acc': stage_acc,
        'stage_bal_acc': stage_bal_acc,
        'stage_f1': stage_f1,
    }


class OncoTrainer:
    """
    A builder pipeline that merges the intention of the OncoTrainer structure 
    with the expected manual metric tracking and components from MULTIMODAL.md.
    """
    def __init__(
        self,
        modalities: List[str],
        model_name: str = "gated_late_fusion",
        max_epochs: int = 10,
        accelerator: str = "auto",
        devices: int = 1,
        join_on: str = "patient_id",
        strategy: str = "inner",
        model_kwargs: dict = None,
        data_kwargs: dict = None
    ):
        self.modalities_requested = modalities
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        
        self.datamodule = MultimodalDataModule(
            modalities=self.modalities_requested,
            join_on=join_on,
            strategy=strategy,
            **(data_kwargs or {})
        )
        
        # Set label column on underlying datamodules if they support it
        for name, dm in self.datamodule.datamodules.items():
            if hasattr(dm, 'label_column'):
                dm.label_column = 'label'
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and accelerator != "cpu" else 'cpu')
        
        config = {'model': model_kwargs or {}}
        variant = 'v1_imaging' if 'image' in modalities else 'v2_no_imaging'
        
        self.model = build_model(config, variant=variant, device=self.device)
        self.model = self.model.to(self.device)
        
    def train(self):
        print(f"Starting OncoTrainer execution with manual loop | Modalities: {self.modalities_requested}")
        
        self.datamodule.prepare_data()
        self.datamodule.setup(stage='fit')
        
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        
        # Determine class weighting (basic uniform initialization, ideally extracted from datamodule)
        criterion_stage = nn.CrossEntropyLoss().to(self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        device_type = self.device.type
        use_amp = device_type == 'cuda'
        scaler = GradScaler(device_type) if use_amp else None

        best_f1 = 0.0
        patience_counter = 0
        max_patience = 10

        for epoch in range(self.max_epochs):
            print(f"Epoch {epoch+1}/{self.max_epochs}")
            train_metrics = train_epoch(
                self.model, train_loader, optimizer,
                criterion_stage, None, 0.3,
                self.device, use_amp=use_amp, scaler=scaler,
                device_type=device_type,
            )
            
            val_metrics = validate(
                self.model, val_loader,
                criterion_stage, None, 0.3,
                self.device
            )
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Stage F1: {val_metrics['stage_f1']:.4f}, Val Acc: {val_metrics['stage_acc']:.4f}")
            
            if val_metrics['stage_f1'] > best_f1:
                best_f1 = val_metrics['stage_f1']
                patience_counter = 0
                print(f"New best F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                
            scheduler.step(val_metrics['stage_f1'])
            
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    def test(self):
        self.datamodule.setup(stage='test')
        test_loader = self.datamodule.test_dataloader()
        criterion_stage = nn.CrossEntropyLoss().to(self.device)
        
        metrics = validate(self.model, test_loader, criterion_stage, None, 0.3, self.device)
        print("Test Metrics:", metrics)


def main():
    """Main function matching docs expectations."""
    parser = argparse.ArgumentParser(description="Evaluate TCGA-BRCA model with new unified pipeline.")
    parser.add_argument('--variant', type=str, default='v1_imaging', choices=['v1_imaging', 'v2_no_imaging'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    set_seed(42)
    modalities = ['image', 'tabular'] if args.variant == 'v1_imaging' else ['tabular']
    
    trainer = OncoTrainer(
        modalities=modalities,
        max_epochs=args.epochs,
        data_kwargs={'batch_size': args.batch_size}
    )
    trainer.train()

if __name__ == '__main__':
    main()
