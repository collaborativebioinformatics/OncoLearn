"""
Simple cancer subtyping model using pretrained components:
- YOLO (COCO pretrained) for feature extraction
- PyTorch Transformer for attention
- Stable Diffusion VAE (pretrained)
- Vision Transformer (pretrained) for classification

This is a proof-of-concept architecture designed to work with NVFlare.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from transformers import ViTForImageClassification

from .yolo_feature_extractor import YOLOFeatureExtractor


class CancerSubtypingModel(nn.Module):
    """
    Simple pipeline: YOLO → Transformer → VAE → ViT Classifier
    All components are pretrained and ready for transfer learning.
    """

    def __init__(
        self,
        # YOLO settings
        yolo_model: str = "yolov8n.pt",
        freeze_yolo: bool = True,

        # Attention settings
        num_attention_layers: int = 2,
        num_attention_heads: int = 8,
        attention_dim: int = 512,

        # VAE settings
        vae_model: str = "stabilityai/sd-vae-ft-mse",
        freeze_vae: bool = True,
        latent_dim: int = 128,

        # ViT classifier settings
        vit_model: str = "google/vit-base-patch16-224",
        num_classes: int = 5,
    ):
        """
        Initialize cancer subtyping model with pretrained components.

        Args:
            yolo_model: YOLO model name (pretrained on COCO)
            freeze_yolo: Whether to freeze YOLO weights
            num_attention_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            attention_dim: Attention embedding dimension
            vae_model: Hugging Face VAE model name
            freeze_vae: Whether to freeze VAE weights
            latent_dim: Final latent dimension
            vit_model: Hugging Face ViT model name
            num_classes: Number of cancer subtypes to classify
        """
        super().__init__()

        # 1. YOLO Feature Extractor (pretrained on COCO)
        print(f"Loading YOLO feature extractor: {yolo_model}")
        self.yolo = YOLOFeatureExtractor(
            model_name=yolo_model,
            pretrained=True,
            freeze_backbone=freeze_yolo
        )

        # Get YOLO output dimension and add adaptive pooling
        self.pool_size = (8, 8)  # Pool each scale to 8x8
        self.yolo_adaptive_pool = nn.AdaptiveAvgPool2d(self.pool_size)
        yolo_dim = sum(self.yolo.feature_dims) * self.pool_size[0] * self.pool_size[1]

        # Project YOLO features to attention dimension
        self.yolo_projection = nn.Linear(yolo_dim, attention_dim)

        # 2. PyTorch Transformer Encoder (trainable for adaptation)
        print(f"Initializing Transformer with {num_attention_layers} layers")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attention_dim,
            nhead=num_attention_heads,
            dim_feedforward=attention_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_attention_layers,
            norm=nn.LayerNorm(attention_dim)
        )
        # Transformer is NOT frozen - will adapt during training

        # Global pooling for transformer output
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 3. Pretrained VAE (Stable Diffusion)
        print(f"Loading pretrained VAE: {vae_model}")
        try:
            self.vae = AutoencoderKL.from_pretrained(vae_model)
            if freeze_vae:
                for param in self.vae.parameters():
                    param.requires_grad = False
                print("VAE weights frozen")
        except Exception as e:
            print(
                f"Warning: Could not load VAE ({e}). Using identity mapping.")
            self.vae = None

        # VAE adapter: project attention output to spatial format for VAE
        self.to_vae_spatial = nn.Sequential(
            nn.Linear(attention_dim, 3 * 64 * 64),  # Project to RGB spatial
            nn.Unflatten(1, (3, 64, 64))
        )

        # Project VAE latent to final latent dim
        if self.vae is not None:
            vae_latent_size = 4 * 8 * 8  # VAE downsamples 64x64 to 8x8 with 4 channels
        else:
            vae_latent_size = attention_dim

        self.latent_projection = nn.Sequential(
            nn.Linear(vae_latent_size, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

        # 4. Pretrained ViT Classifier (transfer learning)
        print(f"Loading pretrained ViT: {vit_model}")
        self.vit = ViTForImageClassification.from_pretrained(
            vit_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True  # Allow classifier head to be replaced
        )

        # Replace ViT's input projection to accept our latent features
        # We'll create a dummy image from latent features
        self.latent_to_vit = nn.Sequential(
            nn.Linear(latent_dim, 224 * 224 * 3),
            nn.Unflatten(1, (3, 224, 224))
        )

        self.num_classes = num_classes

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the pipeline.

        Args:
            pixel_values: Input images [B, C, H, W]
            labels: Optional labels for computing loss [B]

        Returns:
            Dictionary with logits, loss (if labels provided), and intermediate features
        """
        batch_size = pixel_values.size(0)

        # 1. YOLO feature extraction - returns dict with multi-scale features
        yolo_output = self.yolo(pixel_values)
        multi_scale_features = yolo_output['multi_scale_features']
        
        # Pool each scale to same size and concatenate
        pooled_features = []
        for feat in multi_scale_features:
            pooled = self.yolo_adaptive_pool(feat)  # [B, C, 8, 8]
            pooled_features.append(pooled)
        
        # Concatenate and flatten: [B, total_channels, 8, 8] -> [B, total_channels * 64]
        yolo_features = torch.cat(pooled_features, dim=1).flatten(1)

        # Add sequence dimension for transformer [B, 1, attention_dim]
        features = self.yolo_projection(yolo_features).unsqueeze(1)

        # 2. Transformer attention (trainable)
        features = self.transformer(features)  # [B, 1, attention_dim]

        # Pool to single vector
        pooled = features.squeeze(1)  # [B, attention_dim]

        # 3. VAE encoding (optional, pretrained)
        if self.vae is not None:
            # Convert to spatial format
            spatial = self.to_vae_spatial(pooled)  # [B, 3, 64, 64]

            # Encode with pretrained VAE
            with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.vae.parameters())):
                latent_dist = self.vae.encode(spatial).latent_dist
                vae_latent = latent_dist.mode()  # Use mode instead of sample for stability

            # Flatten VAE latent (use reshape instead of view for non-contiguous tensors)
            vae_latent_flat = vae_latent.reshape(batch_size, -1)
            latent = self.latent_projection(vae_latent_flat)
        else:
            # Skip VAE if not available
            latent = self.latent_projection(pooled)

        # 4. ViT classification (pretrained, fine-tuned)
        # Convert latent to image format for ViT
        vit_input = self.latent_to_vit(latent)  # [B, 3, 224, 224]

        # Forward through ViT
        vit_outputs = self.vit(pixel_values=vit_input, labels=labels)

        return {
            'logits': vit_outputs.logits,
            'loss': vit_outputs.loss if labels is not None else None,
            'yolo_features': yolo_features,
            'transformer_features': pooled,
            'latent': latent
        }

    def predict(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (class probabilities).

        Args:
            pixel_values: Input images [B, C, H, W]

        Returns:
            Class probabilities [B, num_classes]
        """
        with torch.no_grad():
            outputs = self.forward(pixel_values)
            probs = torch.softmax(outputs['logits'], dim=-1)
        return probs


class SimplifiedCancerSubtypingModel(nn.Module):
    """
    Even simpler version without VAE for maximum simplicity.
    YOLO → Transformer → ViT Classifier
    """

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        freeze_yolo: bool = True,
        num_attention_layers: int = 2,
        num_attention_heads: int = 8,
        vit_model: str = "google/vit-base-patch16-224",
        num_classes: int = 5,
    ):
        """Initialize simplified model without VAE."""
        super().__init__()

        # 1. YOLO
        print(f"Loading YOLO: {yolo_model}")
        self.yolo = YOLOFeatureExtractor(
            model_name=yolo_model,
            pretrained=True,
            freeze_backbone=freeze_yolo
        )

        yolo_dim = self.yolo.get_feature_dim()

        # 2. Transformer
        print("Initializing Transformer")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=yolo_dim,
            nhead=num_attention_heads,
            dim_feedforward=yolo_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_attention_layers
        )

        # 3. ViT
        print(f"Loading ViT: {vit_model}")
        self.vit = ViTForImageClassification.from_pretrained(
            vit_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        # Project features to ViT input format
        self.to_vit = nn.Sequential(
            nn.Linear(yolo_dim, 224 * 224 * 3),
            nn.Unflatten(1, (3, 224, 224))
        )

        self.num_classes = num_classes

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # YOLO features
        features = self.yolo(pixel_values).unsqueeze(1)  # [B, 1, dim]

        # Transformer
        features = self.transformer(features).squeeze(1)  # [B, dim]

        # ViT
        vit_input = self.to_vit(features)
        outputs = self.vit(pixel_values=vit_input, labels=labels)

        return {
            'logits': outputs.logits,
            'loss': outputs.loss if labels is not None else None,
            'features': features
        }

    def predict(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities."""
        with torch.no_grad():
            outputs = self.forward(pixel_values)
            return torch.softmax(outputs['logits'], dim=-1)
