"""
Gated late fusion classifier supporting 2-modality and 3-modality modes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from oncolearn.registry import register_model


class GatedLateFusionClassifier(nn.Module):
    """
    Gated late fusion with per-modality heads and gating network.
    
    Supports:
    - 3-modality mode: gene, clinical, image (V1)
    - 2-modality mode: gene, clinical (V2)
    
    Architecture:
    - Per-modality encoders produce embeddings
    - Per-modality heads produce logits for each task
    - Gate network produces alphas over available modalities
    - Missing-modality masking applied
    - Final logits = weighted sum of per-modality logits
    """
    
    def __init__(
        self,
        gene_encoder: nn.Module = None,
        clinical_encoder: nn.Module = None,
        image_encoder: nn.Module = None,
        gene_dim: int = 128,
        clinical_dim: int = 128,
        image_dim: int = 256,
        num_stage_classes: int = 5,
        num_subtype_classes: int = 0,  # 0 means no subtype task
        dropout: float = 0.2
    ):
        super().__init__()
        self.gene_encoder = gene_encoder
        self.clinical_encoder = clinical_encoder
        self.image_encoder = image_encoder
        
        self.gene_dim = gene_dim
        self.clinical_dim = clinical_dim
        self.image_dim = image_dim
        self.num_stage_classes = num_stage_classes
        self.num_subtype_classes = num_subtype_classes
        self.has_subtype = num_subtype_classes > 0
        self.has_image = image_encoder is not None
        
        # Per-modality heads for stage
        self.gene_stage_head = nn.Linear(gene_dim, num_stage_classes)
        self.clinical_stage_head = nn.Linear(clinical_dim, num_stage_classes)
        if self.has_image:
            self.image_stage_head = nn.Linear(image_dim, num_stage_classes)
        
        # Per-modality heads for subtype (if enabled)
        if self.has_subtype:
            self.gene_subtype_head = nn.Linear(gene_dim, num_subtype_classes)
            self.clinical_subtype_head = nn.Linear(clinical_dim, num_subtype_classes)
            if self.has_image:
                self.image_subtype_head = nn.Linear(image_dim, num_subtype_classes)
        
        self.has_clinical = clinical_encoder is not None
        self.has_gene = gene_encoder is not None
        
        gate_input_dim = 0
        num_mods = 0
        if self.has_gene:
            gate_input_dim += gene_dim
            num_mods += 1
        if self.has_clinical:
            gate_input_dim += clinical_dim
            num_mods += 1
        if self.has_image:
            gate_input_dim += image_dim
            num_mods += 1
            
        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, 128 if num_mods == 3 else 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128 if num_mods == 3 else 64, num_mods)
        )
    
    def forward(
        self,
        gene: torch.Tensor = None,
        clinical: torch.Tensor = None,
        image: torch.Tensor = None,
        modality_ids: torch.Tensor = None
    ) -> dict:
        """
        Forward pass with optional modalities.
        
        Args:
            gene: (B, gene_input_dim) gene features
            clinical: (B, clinical_input_dim) clinical features
            image: (B, N, C, H, W) image sequence (V1 only)
            modality_ids: (B, N) modality IDs (V1 only)
        
        Returns:
            dict with 'stage_logits' and optionally 'subtype_logits'
        """
        B = None
        available_modalities = []
        modality_embeddings = []
        
        # Encode gene
        if gene is not None:
            B = gene.shape[0]
            z_gene = self.gene_encoder(gene)  # (B, gene_dim)
            available_modalities.append('gene')
            modality_embeddings.append(z_gene)
        
        # Encode clinical
        if clinical is not None:
            if B is None:
                B = clinical.shape[0]
            z_clinical = self.clinical_encoder(clinical)  # (B, clinical_dim)
            available_modalities.append('clinical')
            modality_embeddings.append(z_clinical)
        
        # Encode image (V1 only)
        if image is not None and self.has_image:
            if B is None:
                B = image.shape[0]
            if modality_ids is None:
                # Default to MR (0)
                modality_ids = torch.zeros(B, image.shape[1], dtype=torch.long, device=image.device)
            z_image = self.image_encoder(image, modality_ids)  # (B, image_dim)
            available_modalities.append('image')
            modality_embeddings.append(z_image)
        
        if not modality_embeddings:
            raise ValueError("At least one modality must be provided")
        
        # Validate all embeddings have same batch size
        batch_sizes = [emb.shape[0] for emb in modality_embeddings]
        if len(set(batch_sizes)) > 1:
            raise ValueError(f"Inconsistent batch sizes in modality embeddings: {dict(zip(available_modalities, batch_sizes))}")
        
        # Concatenate available embeddings for gate
        gate_input = torch.cat(modality_embeddings, dim=-1)  # (B, sum(dims))
        
        # Compute gate weights
        gate_logits = self.gate_network(gate_input)  # (B, num_modalities)
        
        # Create mask for missing modalities
        # Create mask for missing modalities
        mask = torch.zeros(B, len(available_modalities), device=gate_logits.device)
        for i, mod in enumerate(available_modalities):
            mask[:, i] = 1.0
        
        # Apply mask and softmax
        gate_logits = gate_logits * mask + (1 - mask) * (-1e9)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, num_modalities)
        
        # Per-modality stage logits
        stage_logits_list = []
        if 'gene' in available_modalities:
            stage_logits_list.append(self.gene_stage_head(z_gene))
        if 'clinical' in available_modalities:
            stage_logits_list.append(self.clinical_stage_head(z_clinical))
        if 'image' in available_modalities:
            stage_logits_list.append(self.image_stage_head(z_image))
        
        # Weighted combination
        stage_logits = torch.stack(stage_logits_list, dim=1)  # (B, num_available, num_classes)
        gate_weights_expanded = gate_weights[:, :len(available_modalities)].unsqueeze(-1)  # (B, num_available, 1)
        stage_logits = (stage_logits * gate_weights_expanded).sum(dim=1)  # (B, num_classes)
        
        result = {'stage_logits': stage_logits}
        
        # Subtype logits (if enabled)
        if self.has_subtype:
            subtype_logits_list = []
            if 'gene' in available_modalities:
                subtype_logits_list.append(self.gene_subtype_head(z_gene))
            if 'clinical' in available_modalities:
                subtype_logits_list.append(self.clinical_subtype_head(z_clinical))
            if 'image' in available_modalities:
                subtype_logits_list.append(self.image_subtype_head(z_image))
            
            subtype_logits = torch.stack(subtype_logits_list, dim=1)  # (B, num_available, num_classes)
            subtype_logits = (subtype_logits * gate_weights_expanded).sum(dim=1)  # (B, num_classes)
            result['subtype_logits'] = subtype_logits
        
        return result


@register_model("gated_late_fusion", modalities=["image", "tabular"])
class GatedLateFusionLightning(pl.LightningModule):
    """
    End-to-End PyTorch Lightning wrapper for the GatedLateFusionClassifier.
    Requires tabular and image modalities to be present in the training batches.
    """
    def __init__(
        self, 
        model: GatedLateFusionClassifier,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        # We expect MultimodalDataModule batches to have "tabular" and "image"
        # Since tabular combines gene+clinical conceptually here, we adapt the inputs.
        # For a clean implementation, you'd map "tabular" features to gene/clinical encoders.
        # Assuming for now 'tabular' directly maps to gene, and clinical is None.
        gene_features = batch.get("tabular")
        image_features = batch.get("image")
        
        return self.model(
            gene=gene_features,
            clinical=None,  # Adjust based on tabular feature splitting
            image=image_features
        )

    def training_step(self, batch, batch_idx):
        preds = self.forward(batch)
        labels = batch.get("label", torch.zeros(preds['stage_logits'].shape[0], dtype=torch.long, device=self.device))
        loss = self.loss_fn(preds['stage_logits'], labels)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        preds = self.forward(batch)
        loss = self.loss_fn(preds['stage_logits'], labels)
        
        preds_class = preds['stage_logits'].argmax(dim=1)
        acc = (preds_class == labels).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
