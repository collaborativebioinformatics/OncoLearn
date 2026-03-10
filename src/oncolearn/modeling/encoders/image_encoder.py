"""
FM-BCMRI pretrained image encoder with hierarchical attention pooling.
Loads a pretrained checkpoint and uses it as a frozen feature extractor.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from oncolearn.registry import register_config, register_encoder
from .base import BaseEncoder

if TYPE_CHECKING:
    from oncolearn.config import OncoLearnConfig

logger = logging.getLogger(__name__)


@register_config("oncolearn.encoder.multimodal.MRMGHierarchicalImageEncoder")
@dataclass
class MRMGHierarchicalImageEncoderConfig:
    """Configuration for the FM-BCMRI hierarchical image encoder."""

    output_dim: int = 256
    checkpoint_path: Optional[str] = None
    # Internal dim used by feature_proj (backbone → pool) and attention pooling.
    backbone_feature_dim: int = 256
    # Fallback cube size (px) for 3D ViT when pos_embed is absent in the checkpoint.
    vit_3d_default_target_size: int = 48
    # Input channels fed to the 3D ViT (FM-BCMRI expects single-channel grayscale).
    vit_3d_in_channels: int = 1


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


@register_encoder("oncolearn.encoder.multimodal.MRMGHierarchicalImageEncoder")
class MRMGHierarchicalImageEncoder(BaseEncoder):
    """
    Pretrained image encoder from checkpoint with hierarchical attention pooling.

    Architecture:
    - Pretrained backbone from checkpoint (frozen by default)
    - Feature projection to a consistent internal dimension
    - Hierarchical attention pooling with modality embeddings (MR / MG)
    - Output: patient embedding in R^{output_dim}

    Args:
        config: Full experiment config.  Encoder-specific parameters are read via
                :func:`~oncolearn.registry.resolve_encoder_config` which merges
                :class:`MRMGHierarchicalImageEncoderConfig` defaults with any YAML
                overrides (e.g. ``checkpoint_path``).
    """

    def __init__(self, config: "OncoLearnConfig") -> None:
        from oncolearn.registry import resolve_encoder_config

        enc_cfg: MRMGHierarchicalImageEncoderConfig = resolve_encoder_config(type(self), config)

        super().__init__(config, output_dim=enc_cfg.output_dim, huggingface_models=None)

        if not enc_cfg.checkpoint_path or not Path(enc_cfg.checkpoint_path).exists():
            raise ValueError(
                f"A valid checkpoint_path must be provided for the image encoder. "
                f"Given: {enc_cfg.checkpoint_path}"
            )

        logger.info(f"Loading image encoder from checkpoint: {enc_cfg.checkpoint_path}")
        self._3d_vit_target_size = None
        self.backbone, backbone_dim = self._load_from_checkpoint(
            enc_cfg.checkpoint_path,
            vit_3d_default_target_size=enc_cfg.vit_3d_default_target_size,
            vit_3d_in_channels=enc_cfg.vit_3d_in_channels,
        )
        self.is_vit = (
            hasattr(self.backbone, "patch_embed")
            or hasattr(self.backbone, "blocks")
            or hasattr(self.backbone, "transformer")
        )

        feat_dim = enc_cfg.backbone_feature_dim
        self.feature_proj = nn.Linear(backbone_dim, feat_dim)

        self.attention_pool = HierarchicalAttentionPooling(
            input_dim=feat_dim,
            hidden_dim=feat_dim,
            output_dim=enc_cfg.output_dim,
        )
        self.output_norm = nn.LayerNorm(enc_cfg.output_dim)

        # Stored for use in forward()
        self._backbone_feature_dim = feat_dim

    def _load_from_checkpoint(
        self,
        checkpoint_path: str,
        vit_3d_default_target_size: int = 48,
        vit_3d_in_channels: int = 1,
    ):
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
            logger.info("Detected 3D ViT model. Loading via vit_3d module.")
            from oncolearn.modeling.modules.vit_3d import vit_3d_base_patchsize8

            # Infer the spatial cube size from the positional embedding.
            # FM-BCMRI uses patch_size=8, so img_size = grid_edge * 8.
            # pos_embed shape: (1, num_patches + 1, embed_dim) — subtract 1 for CLS.
            target_size = vit_3d_default_target_size
            if "pos_embed" in backbone_dict:
                num_patches = backbone_dict["pos_embed"].shape[1] - 1
                grid = round(num_patches ** (1 / 3))
                if grid ** 3 != num_patches:
                    logger.warning(
                        f"3D ViT pos_embed has {num_patches} patches — not a perfect cube. "
                        f"Inferred grid edge {grid} (grid³={grid**3}). "
                        f"Using default target_size={vit_3d_default_target_size}."
                    )
                    grid = round(vit_3d_default_target_size / 8)
                target_size = grid * 8

            model = vit_3d_base_patchsize8(img_size=target_size, in_chans=vit_3d_in_channels)
            model.load_state_dict(backbone_dict, strict=False)
            if self.freeze_encoders:
                self._freeze(model)
            self._3d_vit_target_size = target_size
            return model, 768

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

            except ImportError as e:
                raise ImportError(
                    "The 'transformers' package is required to load 2D ViT checkpoints. "
                    "Install it with: pip install transformers"
                ) from e

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

        if modality_ids is None:
            modality_ids = torch.zeros(B, N, dtype=torch.long, device=images.device)

        if self._3d_vit_target_size is not None:
            # FM-BCMRI 3D ViT path: treat N slices as the depth dimension.
            # Convert to single-channel grayscale and stack into a 3D volume.
            gray = images.mean(dim=2, keepdim=True) if C != 1 else images  # (B, N, 1, H, W)
            # Rearrange to (B, 1, H, W, N) — FM-BCMRI PatchEmbed3D expects (B, C, H, W, D)
            vol = gray.permute(0, 2, 3, 4, 1)  # (B, 1, H, W, N)
            # F.interpolate requires (B, C, D, H, W); permute depth to dim-2 temporarily
            vol_dhw = vol.permute(0, 1, 4, 2, 3)  # (B, 1, N, H, W)
            ts = self._3d_vit_target_size
            vol_dhw = F.interpolate(vol_dhw, size=(ts, ts, ts), mode="trilinear", align_corners=False)
            # Back to FM-BCMRI format (B, C, H, W, D)
            vol_fmbcmri = vol_dhw.permute(0, 1, 3, 4, 2)  # (B, 1, ts, ts, ts)
            # CLS token from forward_features: (B, seq_len, 768) → take index 0
            features = self.backbone.forward_features(vol_fmbcmri)[:, 0]  # (B, 768)
            features = self.feature_proj(features)                          # (B, 256)
            # Wrap as a single-token sequence for the attention pool
            features = features.unsqueeze(1)                                # (B, 1, 256)
            return self.output_norm(self.attention_pool(features, modality_ids[:, :1]))

        images_flat = images.view(B * N, C, H, W)

        if self.is_vit:
            features = self.backbone(images_flat)
            features = self._pool_hf_outputs(features, strategy="cls")
        else:
            features = self.backbone(images_flat)
            if len(features.shape) > 2:
                features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)

        features = self.feature_proj(features)                                   # (B*N, feat_dim)
        features = features.view(B, N, self._backbone_feature_dim)
        return self.output_norm(self.attention_pool(features, modality_ids))

    def forward_single_image(self, image: torch.Tensor, modality_id: int = 0) -> torch.Tensor:
        """Forward pass for a single image (inference convenience method)."""
        B = image.shape[0]

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
        return self.output_norm(self.attention_pool.output_proj(x))
