"""
Hybrid model combining YOLO, Attention, VAE, and ViT classifier.
This is the main model for cancer image subtyping.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .attention import StackedAttention
from .classifier import ViTClassifier
from .vae import PretrainedVAE, VariationalAutoEncoder, vae_loss
from .yolo_feature_extractor import MultiScaleFeatureAdapter, YOLOFeatureExtractor


class CancerSubtypingModel(nn.Module):
    """
    Complete pipeline for cancer image subtyping:
    1. YOLO Feature Extraction (backbone without head)
    2. Multi-Head Self-Attention (feature enhancement)
    3. Variational AutoEncoder (distribution learning & dimension reduction)
    4. ViT-style Classifier (final predictions)
    """

    def __init__(
        self,
        # YOLO parameters
        yolo_model_name: str = "yolov8n.pt",
        yolo_pretrained: bool = True,
        freeze_yolo: bool = False,

        # Attention parameters
        num_attention_layers: int = 2,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,

        # VAE parameters
        latent_dim: int = 128,
        vae_hidden_dims: Optional[list] = None,
        vae_dropout: float = 0.1,
        use_pretrained_vae: bool = True,  # NEW: Use Hugging Face pretrained VAE
        pretrained_vae_name: str = "stabilityai/sd-vae-ft-mse",  # NEW
        freeze_pretrained_vae: bool = True,  # NEW

        # Classifier parameters
        num_classes: int = 5,
        classifier_hidden_dim: Optional[int] = None,
        classifier_dropout: float = 0.1,

        # Training parameters
        use_vae_loss: bool = True,
        kld_weight: float = 0.1,
        
        # Feature fusion
        feature_adapter_dim: int = 768
    ):
        """
        Initialize the hybrid cancer subtyping model.

        Args:
            yolo_model_name: Ultralytics YOLO model name (e.g., 'yolov8n.pt')
            yolo_pretrained: Whether to use pretrained YOLO weights
            freeze_yolo: Whether to freeze YOLO backbone
            num_attention_layers: Number of attention blocks
            num_attention_heads: Number of attention heads per block
            attention_dropout: Dropout in attention layers
            latent_dim: VAE latent dimension
            vae_hidden_dims: VAE encoder/decoder hidden dimensions (for custom VAE)
            vae_dropout: Dropout in VAE (for custom VAE)
            use_pretrained_vae: Whether to use pretrained VAE from Hugging Face
            pretrained_vae_name: Hugging Face model name for pretrained VAE
            freeze_pretrained_vae: Whether to freeze pretrained VAE weights
            num_classes: Number of cancer subtypes
            classifier_hidden_dim: Hidden dimension in classifier
            classifier_dropout: Dropout in classifier
            use_vae_loss: Whether to include VAE reconstruction loss
            kld_weight: Weight for KLD term in VAE loss
            feature_adapter_dim: Dimension for multi-scale feature adapter
        """
        super().__init__()

        # 1. YOLO Feature Extractor (multi-scale)
        self.yolo = YOLOFeatureExtractor(
            model_name=yolo_model_name,
            pretrained=yolo_pretrained,
            freeze_backbone=freeze_yolo
        )
        
        # 2. Multi-scale Feature Adapter (converts to sequence format)
        self.feature_adapter = MultiScaleFeatureAdapter(
            yolo_extractor=self.yolo,
            target_dim=feature_adapter_dim,
            num_tokens=50  # Reduced for efficiency
        )
        
        feature_dim = feature_adapter_dim

        # 2. Attention Layers
        self.attention = StackedAttention(
            embed_dim=feature_dim,
            num_layers=num_attention_layers,
            num_heads=num_attention_heads,
            dropout=attention_dropout
        )

        # Pooling to get single vector from sequence
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 3. Variational AutoEncoder (Pretrained or Custom)
        if use_pretrained_vae:
            print(f"Using pretrained VAE: {pretrained_vae_name}")
            self.vae = PretrainedVAE(
                input_dim=feature_dim,
                latent_dim=latent_dim,
                model_name=pretrained_vae_name,
                freeze_vae=freeze_pretrained_vae
            )
        else:
            print("Using custom VAE")
            self.vae = VariationalAutoEncoder(
                input_dim=feature_dim,
                latent_dim=latent_dim,
                hidden_dims=vae_hidden_dims,
                dropout=vae_dropout
            )

        # 4. ViT-style Classifier
        self.classifier = ViTClassifier(
            input_dim=latent_dim,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=classifier_dropout
        )

        self.use_vae_loss = use_vae_loss
        self.kld_weight = kld_weight

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete pipeline.

        Args:
            pixel_values: Input images [batch_size, channels, height, width]
            return_features: Whether to return intermediate features
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing:
                - logits: Classification logits [batch_size, num_classes]
                - mu: VAE latent mean (for VAE loss)
                - logvar: VAE latent log variance (for VAE loss)
                - reconstruction: VAE reconstruction (if use_vae_loss)
                - yolo_features: YOLO multi-scale features (if return_features)
                - attention_features: Attention features (if return_features)
                - latent: VAE latent features (if return_features)
        """
        # 1. Extract multi-scale features with YOLO and convert to sequence
        yolo_features = self.feature_adapter(pixel_values)  # [B, num_tokens, feature_dim]

        # 2. Apply attention
        if return_attention:
            attention_features, attention_weights = self.attention(
                yolo_features,
                return_all_attention=True
            )
        else:
            attention_features = self.attention(yolo_features)
            attention_weights = None

        # 3. Pool to single vector
        # Shape: [B, num_tokens, feature_dim] -> [B, feature_dim, num_tokens] -> [B, feature_dim, 1]
        pooled = self.pool(attention_features.transpose(1, 2)).squeeze(-1)  # [B, feature_dim]

        # 4. Pass through VAE
        vae_outputs = self.vae(pooled, return_latent=True)
        mu = vae_outputs['mu']
        logvar = vae_outputs['logvar']
        latent = vae_outputs['z']
        reconstruction = vae_outputs['reconstruction']

        # 5. Classify
        logits = self.classifier(latent)

        # Prepare outputs
        outputs = {
            'logits': logits,
            'mu': mu,
            'logvar': logvar
        }

        if self.use_vae_loss:
            outputs['reconstruction'] = reconstruction
            outputs['pooled_features'] = pooled

        if return_features:
            outputs['yolo_features'] = yolo_features
            outputs['attention_features'] = attention_features
            outputs['latent'] = latent

        if return_attention:
            outputs['attention_weights'] = attention_weights

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        classification_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss (classification + VAE).

        Args:
            outputs: Model outputs from forward pass
            labels: Ground truth labels [batch_size]
            classification_weight: Weight for classification loss

        Returns:
            Dictionary with individual and total losses
        """
        # Classification loss
        classification_loss = nn.functional.cross_entropy(
            outputs['logits'], labels)

        losses = {
            'classification_loss': classification_loss
        }

        # VAE loss (if enabled)
        if self.use_vae_loss and 'reconstruction' in outputs:
            vae_total_loss, recon_loss, kld_loss = vae_loss(
                reconstruction=outputs['reconstruction'],
                x=outputs['pooled_features'],
                mu=outputs['mu'],
                logvar=outputs['logvar'],
                kld_weight=self.kld_weight,
                reconstruction_loss='mse'
            )

            losses['vae_loss'] = vae_total_loss
            losses['reconstruction_loss'] = recon_loss
            losses['kld_loss'] = kld_loss

            # Total loss
            losses['total_loss'] = classification_weight * \
                classification_loss + vae_total_loss
        else:
            losses['total_loss'] = classification_loss

        return losses

    def predict(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (returns class probabilities).

        Args:
            pixel_values: Input images [batch_size, channels, height, width]

        Returns:
            Class probabilities [batch_size, num_classes]
        """
        with torch.no_grad():
            outputs = self.forward(pixel_values)
            probs = torch.softmax(outputs['logits'], dim=-1)
        return probs

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space (for feature extraction).

        Args:
            pixel_values: Input images [batch_size, channels, height, width]

        Returns:
            Latent features [batch_size, latent_dim]
        """
        with torch.no_grad():
            # Extract multi-scale YOLO features
            yolo_features = self.feature_adapter(pixel_values)

            # Apply attention
            attention_features = self.attention(yolo_features)

            # Pool
            pooled = self.pool(attention_features.transpose(1, 2)).squeeze(-1)

            # Encode to latent space (use mean, not sample)
            mu, _ = self.vae.encode(pooled)

        return mu


class SimplifiedCancerSubtypingModel(nn.Module):
    """
    Simplified version without VAE for faster training/inference.
    YOLO (multi-scale) -> Feature Adapter -> Attention -> Pooling -> Classifier
    """

    def __init__(
        self,
        yolo_model_name: str = "yolov8n.pt",
        yolo_pretrained: bool = True,
        freeze_yolo: bool = False,
        num_attention_layers: int = 2,
        num_attention_heads: int = 8,
        num_classes: int = 5,
        dropout: float = 0.1,
        feature_adapter_dim: int = 768
    ):
        """Initialize simplified model."""
        super().__init__()

        # YOLO backbone (multi-scale)
        self.yolo = YOLOFeatureExtractor(
            model_name=yolo_model_name,
            pretrained=yolo_pretrained,
            freeze_backbone=freeze_yolo
        )
        
        # Multi-scale Feature Adapter
        self.feature_adapter = MultiScaleFeatureAdapter(
            yolo_extractor=self.yolo,
            target_dim=feature_adapter_dim,
            num_tokens=50
        )

        feature_dim = feature_adapter_dim

        # Attention
        self.attention = StackedAttention(
            embed_dim=feature_dim,
            num_layers=num_attention_layers,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = ViTClassifier(
            input_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Extract multi-scale features
        features = self.feature_adapter(pixel_values)

        # Apply attention
        features = self.attention(features)

        # Pool
        pooled = self.pool(features.transpose(1, 2)).squeeze(-1)

        # Classify
        logits = self.classifier(pooled)

        return logits
