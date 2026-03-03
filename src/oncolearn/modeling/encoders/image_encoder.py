"""
FM-BCMRI pretrained image encoder with hierarchical attention pooling.
Loads a pretrained checkpoint and uses it as a frozen feature extractor.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from oncolearn.registry import register_encoder
from .base import BaseEncoder

if TYPE_CHECKING:
    from oncolearn.config import OncoLearnConfig

logger = logging.getLogger(__name__)


class HierarchicalAttentionPooling(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Modality embeddings (MR=0, MG=1)
        self.modality_embed = nn.Embedding(2, hidden_dim)

        self.attention = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.output_proj = nn.Linear(input_dim + hidden_dim, output_dim)

    def forward(self, features: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, N, input_dim)
            modality_ids: (B, N) modality IDs (0=MR, 1=MG)

        Returns:
            (B, output_dim) patient-level embedding
        """
        mod_emb = self.modality_embed(modality_ids)        # (B, N, hidden_dim)
        x = torch.cat([features, mod_emb], dim=-1)         # (B, N, input_dim + hidden_dim)
        attn_weights = F.softmax(self.attention(x), dim=1) # (B, N, 1)
        x = (x * attn_weights).sum(dim=1)                  # (B, input_dim + hidden_dim)
        return self.output_proj(x)                         # (B, output_dim)


@register_encoder("image")
class MRMGHierarchicalImageEncoder(BaseEncoder):
    """
    Pretrained image encoder from checkpoint with hierarchical attention pooling.

    Architecture:
    - Pretrained backbone from checkpoint (frozen by default)
    - Feature projection to a consistent internal dimension
    - Hierarchical attention pooling with modality embeddings (MR / MG)
    - Output: patient embedding in R^{output_dim}

    Args:
        config: Full experiment config. Reads ``model.freeze_encoders`` for the
                backbone freeze flag.
        output_dim: Output embedding dimension.
        checkpoint_path: Path to pretrained checkpoint file (required).
    """

    def __init__(
        self,
        config: "OncoLearnConfig",
        output_dim: int = 256,
        checkpoint_path: str = None,
        **kwargs,
    ):
        super().__init__(config, output_dim=output_dim, huggingface_models=None, **kwargs)

        # Resolve checkpoint path: kwarg takes priority, then huggingface config.
        if not checkpoint_path and config.huggingface:
            checkpoint_path = config.huggingface.image_checkpoint

        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise ValueError(
                f"A valid checkpoint_path must be provided for the image encoder. "
                f"Given: {checkpoint_path}"
            )

        logger.info(f"Loading image encoder from checkpoint: {checkpoint_path}")
        self.backbone, backbone_dim = self._load_from_checkpoint(checkpoint_path)
        self.is_vit = (
            hasattr(self.backbone, "patch_embed")
            or hasattr(self.backbone, "blocks")
            or hasattr(self.backbone, "transformer")
        )

        self.feature_proj = nn.Linear(backbone_dim, 256)

        self.attention_pool = HierarchicalAttentionPooling(
            input_dim=256,
            hidden_dim=256,
            output_dim=output_dim,
        )

    def _load_from_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        state_dict = None
        arch = None

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                arch = checkpoint.get("arch", checkpoint.get("architecture", "unknown"))
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
                arch = checkpoint.get("arch", checkpoint.get("architecture", "unknown"))
            elif any(
                k.startswith(("backbone.", "base_encoder.", "encoder.", "model."))
                for k in checkpoint.keys()
            ):
                state_dict = checkpoint
                arch = checkpoint.get("arch", checkpoint.get("architecture", "unknown"))
            else:
                state_dict = checkpoint
                arch = checkpoint.get("arch", checkpoint.get("architecture", "unknown"))

        if state_dict is None:
            raise ValueError(f"Could not extract state_dict from checkpoint: {checkpoint_path}")

        logger.info(f"Loading checkpoint with architecture: {arch}")

        backbone_keys = [k for k in state_dict.keys() if k.startswith("backbone.")]
        base_encoder_keys = [k for k in state_dict.keys() if k.startswith("base_encoder.")]
        encoder_keys = [k for k in state_dict.keys() if k.startswith("encoder.")]
        model_keys = [k for k in state_dict.keys() if k.startswith("model.")]

        backbone_dict = {}
        if backbone_keys:
            backbone_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
        elif base_encoder_keys:
            backbone_dict = {k.replace("base_encoder.", ""): v for k, v in state_dict.items() if k.startswith("base_encoder.")}
        elif encoder_keys:
            backbone_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
        elif model_keys:
            backbone_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        else:
            backbone_dict = state_dict

        if not backbone_dict:
            logger.warning("No backbone found with common prefixes. Trying to load as direct state_dict.")
            backbone_dict = state_dict

        is_vit = any(
            k in backbone_dict
            for k in ["patch_embed", "cls_token", "blocks.0", "pos_embed", "transformer.blocks.0"]
        )
        is_3d = ("vit_3d" in arch.lower() if arch else False) or any(
            "3d" in k.lower() for k in backbone_dict.keys()
        )

        if is_3d:
            logger.info("Detected 3D ViT model. Creating 3D ViT wrapper.")
            from oncolearn.modeling.modules.vit_3d_wrapper import ViT3DWrapper
            backbone = ViT3DWrapper(backbone_dict, self.freeze_encoders)
            return backbone, 768

        if is_vit:
            logger.info("Detected 2D ViT model.")
            try:
                from transformers import ViTModel, ViTConfig

                hidden_size = 768
                if "cls_token" in backbone_dict:
                    hidden_size = backbone_dict["cls_token"].shape[-1]
                elif "transformer.cls_token" in backbone_dict:
                    hidden_size = backbone_dict["transformer.cls_token"].shape[-1]

                num_layers = 12
                layer_keys = [
                    k for k in backbone_dict.keys()
                    if k.startswith("blocks.") or k.startswith("transformer.blocks.")
                ]
                if layer_keys:
                    nums = []
                    for k in layer_keys:
                        parts = k.split(".")
                        for i, part in enumerate(parts):
                            if part == "blocks" and i + 1 < len(parts) and parts[i + 1].isdigit():
                                nums.append(int(parts[i + 1]))
                    if nums:
                        num_layers = max(nums) + 1

                vit_cfg = ViTConfig(
                    image_size=224, patch_size=16, num_channels=3,
                    hidden_size=hidden_size, num_hidden_layers=num_layers,
                    num_attention_heads=12, intermediate_size=3072,
                )
                vit_model = ViTModel(vit_cfg)

                model_backbone_dict = {
                    (k.replace("transformer.", "") if k.startswith("transformer.") else k): v
                    for k, v in backbone_dict.items()
                }

                try:
                    vit_model.load_state_dict(model_backbone_dict, strict=False)
                    logger.info("Loaded 2D ViT model from checkpoint")
                except Exception as e:
                    logger.warning(f"Could not load all weights: {e}. Using partial loading.")
                    model_dict = vit_model.state_dict()
                    model_dict.update({k: v for k, v in model_backbone_dict.items() if k in model_dict})
                    vit_model.load_state_dict(model_dict)

                if self.freeze_encoders:
                    self._freeze(vit_model)

                return vit_model.encoder, vit_cfg.hidden_size

            except ImportError:
                logger.warning("transformers not available. Falling back to ViT3DWrapper.")
                from oncolearn.modeling.modules.vit_3d_wrapper import ViT3DWrapper
                backbone = ViT3DWrapper(backbone_dict, self.freeze_encoders)
                return backbone, 768

        raise NotImplementedError(
            f"Could not identify model architecture from checkpoint. "
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
        images_flat = images.view(B * N, C, H, W)

        if self.is_vit:
            features = self.backbone(images_flat)
            features = self._pool_hf_outputs(features, strategy="cls")
        else:
            features = self.backbone(images_flat)
            if len(features.shape) > 2:
                features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)

        features = self.feature_proj(features)      # (B*N, 256)
        features = features.view(B, N, 256)
        return self.attention_pool(features, modality_ids)

    def forward_single_image(self, image: torch.Tensor, modality_id: int = 0) -> torch.Tensor:
        """Forward pass for a single image (inference convenience method)."""
        B, C, H, W = image.shape

        if self.is_vit:
            features = self.backbone(image)
            features = self._pool_hf_outputs(features, strategy="cls")
        else:
            features = self.backbone(image)
            if len(features.shape) > 2:
                features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)

        features = self.feature_proj(features)  # (B, 256)

        modality_ids = torch.full((B,), modality_id, dtype=torch.long, device=image.device)
        mod_emb = self.attention_pool.modality_embed(modality_ids)
        x = torch.cat([features, mod_emb], dim=-1)
        return self.attention_pool.output_proj(x)
