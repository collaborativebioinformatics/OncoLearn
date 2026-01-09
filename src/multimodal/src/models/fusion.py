"""
Gated late fusion classifier supporting 2-modality and 3-modality modes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        gene_encoder: nn.Module,
        clinical_encoder: nn.Module,
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
        
        # Gate network
        if self.has_image:
            # 3-modality mode
            gate_input_dim = gene_dim + clinical_dim + image_dim
            self.gate_network = nn.Sequential(
                nn.Linear(gate_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 3)  # 3 modalities
            )
        else:
            # 2-modality mode
            gate_input_dim = gene_dim + clinical_dim
            self.gate_network = nn.Sequential(
                nn.Linear(gate_input_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2)  # 2 modalities
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
        if self.has_image:
            # 3-modality mode: [gene, clinical, image]
            mask = torch.zeros(B, 3, device=gate_logits.device)
            if 'gene' in available_modalities:
                mask[:, 0] = 1.0
            if 'clinical' in available_modalities:
                mask[:, 1] = 1.0
            if 'image' in available_modalities:
                mask[:, 2] = 1.0
        else:
            # 2-modality mode: [gene, clinical]
            mask = torch.zeros(B, 2, device=gate_logits.device)
            if 'gene' in available_modalities:
                mask[:, 0] = 1.0
            if 'clinical' in available_modalities:
                mask[:, 1] = 1.0
        
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

