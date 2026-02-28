"""
FM-BCMRI pretrained models from checkpoint.
Loads pretrained checkpoint and uses as frozen feature extractor.
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HierarchicalAttentionPooling(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Modality embeddings (MR, MG)
        self.modality_embed = nn.Embedding(2, hidden_dim)  # 0=MR, 1=MG
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.output_proj = nn.Linear(input_dim + hidden_dim, output_dim)
    
    def forward(self, features: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, N, input_dim) where N is number of images
            modality_ids: (B, N) modality IDs (0=MR, 1=MG)
        
        Returns:
            (B, output_dim) patient-level embedding
        """
        B, N, C = features.shape
        
        mod_emb = self.modality_embed(modality_ids)  # (B, N, hidden_dim)
        
        x = torch.cat([features, mod_emb], dim=-1)  # (B, N, input_dim + hidden_dim)
        
        attn_weights = self.attention(x)  # (B, N, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        x = (x * attn_weights).sum(dim=1)  # (B, input_dim + hidden_dim)
        
        x = self.output_proj(x)  # (B, output_dim)
        
        return x


class MRMGHierarchicalImageEncoder(nn.Module):
    """
    Pretrained model encoder from checkpoint with hierarchical attention pooling.
    
    Architecture:
    - Pretrained model from checkpoint (frozen by default)
    - Hierarchical attention pooling with modality embeddings
    - Output: patient embedding z_i in R^256
    
    Args:
        checkpoint_path: Path to pretrained checkpoint (required)
        freeze_backbone: Whether to freeze the pretrained model (default: True)
        output_dim: Output embedding dimension (default: 256)
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        freeze_backbone: bool = True,
        output_dim: int = 256,
        backbone: str = None,
        pretrained: bool = None
    ):
        super().__init__()
        self.output_dim = output_dim
        self.checkpoint_path = checkpoint_path
        
        # Checkpoint path is required
        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise ValueError(
                f"Checkpoint path must be provided and exist. "
                f"Given path: {checkpoint_path}"
            )
        
        # Load from checkpoint
        logger.info(f"Loading image encoder from checkpoint: {checkpoint_path}")
        self.backbone, backbone_dim = self._load_from_checkpoint(checkpoint_path, freeze_backbone)
        self.is_vit = hasattr(self.backbone, 'patch_embed') or hasattr(self.backbone, 'blocks') or hasattr(self.backbone, 'transformer')
        
        # Feature projection to consistent dim
        self.feature_proj = nn.Linear(backbone_dim, 256)
        
        # Hierarchical attention pooling
        self.attention_pool = HierarchicalAttentionPooling(
            input_dim=256,
            hidden_dim=256,
            output_dim=output_dim
        )
    
    def _load_from_checkpoint(self, checkpoint_path: str, freeze_backbone: bool = True):

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Try different checkpoint formats
        state_dict = None
        arch = None
        
        # Common checkpoint formats
        if isinstance(checkpoint, dict):
            # Format 1: {'state_dict': ..., 'arch': ...}
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                arch = checkpoint.get('arch', checkpoint.get('architecture', 'unknown'))
            # Format 2: {'model': ..., 'arch': ...}
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                arch = checkpoint.get('arch', checkpoint.get('architecture', 'unknown'))
            # Format 3: Direct state_dict with prefix
            elif any(k.startswith(('backbone.', 'base_encoder.', 'encoder.', 'model.')) for k in checkpoint.keys()):
                state_dict = checkpoint
                arch = checkpoint.get('arch', checkpoint.get('architecture', 'unknown'))
            # Format 4: FM-BCMRI might use different keys
            else:
                state_dict = checkpoint
                arch = checkpoint.get('arch', checkpoint.get('architecture', 'unknown'))
        
        if state_dict is None:
            raise ValueError(f"Could not extract state_dict from checkpoint: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint with architecture: {arch}")
        
        # Try to identify model type and extract backbone
        # Option 1: FM-BCMRI or models with 'backbone' prefix
        backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone.')]
        # Option 2: Models with 'base_encoder' prefix
        base_encoder_keys = [k for k in state_dict.keys() if k.startswith('base_encoder.')]
        # Option 3: Models with 'encoder' prefix
        encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder.')]
        # Option 4: Models with 'model' prefix
        model_keys = [k for k in state_dict.keys() if k.startswith('model.')]
        
        # Extract backbone state dict
        backbone_dict = {}
        prefix = None
        
        if backbone_keys:
            prefix = 'backbone.'
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    new_key = key.replace(prefix, '')
                    backbone_dict[new_key] = value
        elif base_encoder_keys:
            prefix = 'base_encoder.'
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    new_key = key.replace(prefix, '')
                    backbone_dict[new_key] = value
        elif encoder_keys:
            prefix = 'encoder.'
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    new_key = key.replace(prefix, '')
                    backbone_dict[new_key] = value
        elif model_keys:
            prefix = 'model.'
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    new_key = key.replace(prefix, '')
                    backbone_dict[new_key] = value
        else:
            # No prefix, assume it's the backbone itself
            backbone_dict = state_dict
        
        if not backbone_dict:
            logger.warning(f"No backbone found with common prefixes. Trying to load as direct state_dict.")
            backbone_dict = state_dict
        
        # Check if it's a ViT (has patch_embed, blocks, cls_token, etc.)
        is_vit = any(k in backbone_dict for k in ['patch_embed', 'cls_token', 'blocks.0', 'pos_embed', 'transformer.blocks.0'])
        # Check if it's a 3D ViT
        is_3d = 'vit_3d' in arch.lower() if arch else False
        is_3d = is_3d or '3d' in arch.lower() if arch else False
        is_3d = is_3d or any('3d' in k.lower() for k in backbone_dict.keys())
        
        if is_3d:
            logger.info("Detected 3D ViT model. Creating 3D ViT wrapper.")
            from .vit_3d_wrapper import ViT3DWrapper
            vit_model = ViT3DWrapper(backbone_dict, freeze_backbone)
            backbone_dim = 768  # Default for base model
            return vit_model, backbone_dim
        elif is_vit:
            # Try to load as 2D ViT
            logger.info("Detected 2D ViT model.")
            try:
                from transformers import ViTModel, ViTConfig
                
                # Infer config from checkpoint
                hidden_size = 768  # Default
                if 'cls_token' in backbone_dict:
                    hidden_size = backbone_dict['cls_token'].shape[-1]
                elif 'transformer.cls_token' in backbone_dict:
                    hidden_size = backbone_dict['transformer.cls_token'].shape[-1]
                
                num_layers = 12  # Default
                layer_keys = [k for k in backbone_dict.keys() if k.startswith('blocks.') or k.startswith('transformer.blocks.')]
                if layer_keys:
                    layer_nums = []
                    for k in layer_keys:
                        parts = k.split('.')
                        # Handle both 'blocks.0' and 'transformer.blocks.0'
                        for i, part in enumerate(parts):
                            if part == 'blocks' and i + 1 < len(parts) and parts[i+1].isdigit():
                                layer_nums.append(int(parts[i+1]))
                    if layer_nums:
                        num_layers = max(layer_nums) + 1
                
                config = ViTConfig(
                    image_size=224,
                    patch_size=16,
                    num_channels=3,
                    hidden_size=hidden_size,
                    num_hidden_layers=num_layers,
                    num_attention_heads=12,
                    intermediate_size=3072,
                )
                
                # Create model
                vit_model = ViTModel(config)
                
                # Handle transformer prefix if exists
                model_backbone_dict = {}
                for k, v in backbone_dict.items():
                    if k.startswith('transformer.'):
                        new_key = k.replace('transformer.', '')
                        model_backbone_dict[new_key] = v
                    else:
                        model_backbone_dict[k] = v
                
                # Load weights (with flexibility for key mismatches)
                try:
                    vit_model.load_state_dict(model_backbone_dict, strict=False)
                    logger.info("Loaded 2D ViT model from checkpoint")
                except Exception as e:
                    logger.warning(f"Could not load all weights: {e}. Using partial loading.")
                    model_dict = vit_model.state_dict()
                    matching_dict = {k: v for k, v in model_backbone_dict.items() if k in model_dict}
                    model_dict.update(matching_dict)
                    vit_model.load_state_dict(model_dict)
                
                # Freeze if requested
                if freeze_backbone:
                    for param in vit_model.parameters():
                        param.requires_grad = False
                
                # Return the encoder part
                backbone = vit_model.encoder
                backbone_dim = config.hidden_size
                
                return backbone, backbone_dim
                
            except ImportError:
                logger.warning("transformers package not available. Trying alternative ViT loading.")
                # Fallback: try to use custom ViT wrapper
                from .vit_3d_wrapper import ViT3DWrapper
                vit_model = ViT3DWrapper(backbone_dict, freeze_backbone)
                backbone_dim = 768
                return vit_model, backbone_dim
        else:
            # Unknown architecture
            raise NotImplementedError(
                f"Could not identify model architecture from checkpoint. "
                f"Please ensure the checkpoint is from FM-BCMRI or ViT. "
                f"Found keys: {list(backbone_dict.keys())[:10]}"
            )
    
    def forward(self, images: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, N, C, H, W) batch of image sequences
            modality_ids: (B, N) modality IDs (0=MR, 1=MG)
        
        Returns:
            (B, output_dim) patient-level embedding
        """
        B, N, C, H, W = images.shape
        
        # Flatten batch and sequence dimensions
        images_flat = images.view(B * N, C, H, W)
        
        # Extract features with backbone
        if self.is_vit:
            # ViT forward pass
            # For ViT, we need to handle CLS token or pooling
            if hasattr(self.backbone, '__call__'):
                # Direct model call (for ViTModel.encoder)
                features = self.backbone(images_flat)  # Might return tuple or dict
                if isinstance(features, dict):
                    features = features.last_hidden_state[:, 0]  # CLS token
                elif isinstance(features, tuple):
                    features = features[0][:, 0]  # CLS token from tuple
                elif len(features.shape) == 3:
                    features = features[:, 0]  # CLS token
                # features should be (B*N, hidden_size)
            else:
                # Custom wrapper
                features = self.backbone(images_flat)  # (B*N, hidden_size)
        else:
            # For non-ViT models, assume they return features directly
            features = self.backbone(images_flat)  # (B*N, backbone_dim)
            if len(features.shape) > 2:
                # If spatial dimensions exist, use global average pooling
                features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        
        # Project to consistent dim
        features = self.feature_proj(features)  # (B*N, 256)
        
        # Reshape back to (B, N, 256)
        features = features.view(B, N, 256)
        
        # Hierarchical attention pooling
        patient_embedding = self.attention_pool(features, modality_ids)  # (B, output_dim)
        
        return patient_embedding
    
    def forward_single_image(self, image: torch.Tensor, modality_id: int = 0) -> torch.Tensor:
        """
        Forward pass for single image (for inference).
        
        Args:
            image: (B, C, H, W) single image
            modality_id: modality ID (0=MR, 1=MG)
        
        Returns:
            (B, output_dim) image embedding
        """
        B, C, H, W = image.shape
        
        # Extract features
        if self.is_vit:
            if hasattr(self.backbone, '__call__'):
                features = self.backbone(image)
                if isinstance(features, dict):
                    features = features.last_hidden_state[:, 0]
                elif isinstance(features, tuple):
                    features = features[0][:, 0]
                elif len(features.shape) == 3:
                    features = features[:, 0]
            else:
                features = self.backbone(image)
        else:
            features = self.backbone(image)  # (B, backbone_dim, ...)
            if len(features.shape) > 2:
                features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        
        # Project
        features = self.feature_proj(features)  # (B, 256)
        
        # Add modality embedding and project
        modality_ids = torch.full((B,), modality_id, dtype=torch.long, device=image.device)
        mod_emb = self.attention_pool.modality_embed(modality_ids)  # (B, 256)
        
        x = torch.cat([features, mod_emb], dim=-1)  # (B, 512)
        x = self.attention_pool.output_proj(x)  # (B, output_dim)
        
        return x
