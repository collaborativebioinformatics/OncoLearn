"""
YOLO-based Feature Extractor (without detection head).
Uses Ultralytics YOLO models to extract features from 3 detection scales.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from ultralytics import YOLO


class YOLOFeatureExtractor(nn.Module):
    """
    YOLO Feature Extractor. Extracts multi-scale features from backbone before 
    the detection head. Features are extracted from 3 different scales (P3, P4, P5).
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        extract_from_layers: Optional[List[int]] = None
    ):
        """
        Initialize YOLO feature extractor.

        Args:
            model_name: Ultralytics model name
                Options: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
                Or 'yolov9c.pt', 'yolov10n.pt', etc.
            pretrained: Whether to load pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
            extract_from_layers: Which layers to extract features from (None = auto-detect from neck)
        """
        super().__init__()

        # Load YOLO model
        if pretrained:
            self.yolo = YOLO(model_name)
        else:
            # Load architecture without weights
            self.yolo = YOLO(model_name.replace('.pt', '.yaml'))
        
        # Get the actual model
        self.model = self.yolo.model
        
        # Store model name for reference
        self.model_name = model_name
        
        # Auto-detect feature extraction layers from neck outputs
        # In YOLOv8, the backbone outputs to layers 15, 18, 21 (for nano/small)
        # These correspond to P3 (80x80), P4 (40x40), P5 (20x20) feature maps
        if extract_from_layers is None:
            # Default: extract from the 3 scales before detection head
            # These layer indices work for YOLOv8
            self.extract_from_layers = [15, 18, 21]  # Will be validated
        else:
            self.extract_from_layers = extract_from_layers
        
        # Calculate feature dimensions
        self._calculate_feature_dims()
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()

    def _calculate_feature_dims(self):
        """Calculate output dimensions from each extraction layer."""
        # Run a dummy forward pass to get dimensions
        dummy_input = torch.zeros(1, 3, 640, 640)
        with torch.no_grad():
            features = self._extract_features(dummy_input)
        
        self.feature_dims = [f.shape[1] for f in features]  # Channel dimensions
        self.feature_shapes = [f.shape[2:] for f in features]  # Spatial dimensions
        self.total_channels = sum(self.feature_dims)
        
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        # Freeze all model parameters up to the neck
        for i, module in enumerate(self.model.model):
            if i < min(self.extract_from_layers):
                for param in module.parameters():
                    param.requires_grad = False

    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from specified layers.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            List of feature tensors from different scales
        """
        features = []
        
        # Forward through model and collect intermediate features
        y = []
        for i, m in enumerate(self.model.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            
            x = m(x)  # run module
            y.append(x if m.i in self.model.save else None)  # save output
            
            # Extract features from specified layers
            if i in self.extract_from_layers:
                features.append(x if isinstance(x, torch.Tensor) else x[0])
        
        return features

    def forward(self, pixel_values: torch.Tensor) -> dict:
        """
        Forward pass through YOLO backbone.

        Args:
            pixel_values: Input images [batch_size, channels, height, width]
                Expected input: RGB images, will be normalized internally

        Returns:
            Dictionary containing:
                - features_p3: Features from scale P3 (highest resolution) [B, C1, H/8, W/8]
                - features_p4: Features from scale P4 (medium resolution) [B, C2, H/16, W/16]
                - features_p5: Features from scale P5 (lowest resolution) [B, C3, H/32, W/32]
                - multi_scale_features: List of all feature tensors
                - feature_dims: List of channel dimensions for each scale
        """
        # Normalize input if needed (YOLO expects 0-1 range)
        if pixel_values.max() > 1.0:
            pixel_values = pixel_values / 255.0
        
        # Extract multi-scale features
        multi_scale_features = self._extract_features(pixel_values)
        
        result = {
            'multi_scale_features': multi_scale_features,
            'feature_dims': self.feature_dims,
            'feature_shapes': [(f.shape[2], f.shape[3]) for f in multi_scale_features]
        }
        
        # Add individual scale features
        if len(multi_scale_features) >= 3:
            result['features_p3'] = multi_scale_features[0]  # Highest resolution
            result['features_p4'] = multi_scale_features[1]  # Medium resolution
            result['features_p5'] = multi_scale_features[2]  # Lowest resolution
        
        return result

    def preprocess_images(self, images, return_tensors="pt"):
        """
        Preprocess images for YOLO.

        Args:
            images: PIL images or numpy arrays
            return_tensors: Format to return ('pt' for PyTorch)

        Returns:
            Preprocessed pixel values ready for model
        """
        import numpy as np
        from PIL import Image
        
        if not isinstance(images, list):
            images = [images]
        
        processed = []
        for img in images:
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            # Resize to YOLO input size (640x640)
            if img.shape[:2] != (640, 640):
                img = Image.fromarray(img).resize((640, 640))
                img = np.array(img)
            
            # Convert to CHW format
            if len(img.shape) == 2:  # Grayscale
                img = np.stack([img] * 3, axis=0)
            elif img.shape[2] == 3:  # HWC to CHW
                img = img.transpose(2, 0, 1)
            
            processed.append(img)
        
        batch = np.stack(processed, axis=0).astype(np.float32)
        
        if return_tensors == "pt":
            return torch.from_numpy(batch)
        return batch

    def get_feature_dim(self) -> int:
        """Return the total output feature dimension (sum of all scales)."""
        return self.total_channels
    
    def get_feature_dims_per_scale(self) -> List[int]:
        """Return feature dimensions for each scale."""
        return self.feature_dims


class YOLOFeatureExtractorWithProjection(nn.Module):
    """
    YOLO Feature Extractor with fusion and projection layer.
    Combines multi-scale features and projects to a target dimension.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        projection_dim: Optional[int] = None,
        fusion_method: str = "adaptive_pool"  # 'adaptive_pool', 'concat', 'attention'
    ):
        """
        Initialize YOLO feature extractor with projection.

        Args:
            model_name: Ultralytics model name
            pretrained: Whether to load pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
            projection_dim: Output dimension after projection (None = no projection)
            fusion_method: How to fuse multi-scale features
                - 'adaptive_pool': Pool each scale to same size, then concatenate
                - 'concat': Flatten and concatenate all features
                - 'attention': Use attention-based fusion
        """
        super().__init__()

        self.yolo = YOLOFeatureExtractor(
            model_name=model_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

        self.fusion_method = fusion_method
        self.projection_dim = projection_dim
        
        # Calculate fusion output dimension
        if fusion_method == "adaptive_pool":
            # Pool to 8x8 for each scale
            self.pool_size = (8, 8)
            self.fusion_dim = sum(self.yolo.feature_dims) * self.pool_size[0] * self.pool_size[1]
        elif fusion_method == "concat":
            # Flatten all features
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 640, 640)
                outputs = self.yolo(dummy)
                total_elements = sum([f.numel() // f.shape[0] for f in outputs['multi_scale_features']])
                self.fusion_dim = total_elements
        elif fusion_method == "attention":
            # Use learnable attention weights
            self.scale_attention = nn.Parameter(torch.ones(len(self.yolo.feature_dims)))
            # Pool and concatenate
            self.pool_size = (8, 8)
            self.fusion_dim = sum(self.yolo.feature_dims) * self.pool_size[0] * self.pool_size[1]
        
        # Optional projection layer
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.fusion_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU()
            )
        else:
            self.projection = None

    def _fuse_features(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-scale features into a single tensor.
        
        Args:
            multi_scale_features: List of feature tensors from different scales
            
        Returns:
            Fused feature tensor [B, fusion_dim]
        """
        batch_size = multi_scale_features[0].shape[0]
        
        if self.fusion_method == "adaptive_pool":
            # Adaptive pooling to same size for each scale
            pooled_features = []
            for features in multi_scale_features:
                # [B, C, H, W] -> [B, C, pool_h, pool_w]
                pooled = nn.functional.adaptive_avg_pool2d(features, self.pool_size)
                # [B, C, pool_h, pool_w] -> [B, C * pool_h * pool_w]
                pooled = pooled.flatten(1)
                pooled_features.append(pooled)
            
            # Concatenate all scales
            fused = torch.cat(pooled_features, dim=1)
            
        elif self.fusion_method == "concat":
            # Flatten and concatenate
            flattened = [f.flatten(1) for f in multi_scale_features]
            fused = torch.cat(flattened, dim=1)
            
        elif self.fusion_method == "attention":
            # Attention-weighted fusion
            pooled_features = []
            for features in multi_scale_features:
                pooled = nn.functional.adaptive_avg_pool2d(features, self.pool_size)
                pooled = pooled.flatten(1)
                pooled_features.append(pooled)
            
            # Apply attention weights
            attention_weights = torch.softmax(self.scale_attention, dim=0)
            weighted_features = []
            for i, pooled in enumerate(pooled_features):
                weighted_features.append(pooled * attention_weights[i])
            
            fused = torch.cat(weighted_features, dim=1)
        
        return fused

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale fusion and projection.

        Args:
            pixel_values: Input images [batch_size, channels, height, width]

        Returns:
            Feature tensor [batch_size, projection_dim or fusion_dim]
        """
        outputs = self.yolo(pixel_values)
        
        # Fuse multi-scale features
        fused_features = self._fuse_features(outputs['multi_scale_features'])
        
        # Apply projection if exists
        if self.projection is not None:
            fused_features = self.projection(fused_features)
        
        return fused_features

    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        if self.projection_dim is not None:
            return self.projection_dim
        return self.fusion_dim


class MultiScaleFeatureAdapter(nn.Module):
    """
    Adapter to convert YOLO multi-scale features to sequence format
    compatible with attention layers and transformers.
    """
    
    def __init__(
        self,
        yolo_extractor: YOLOFeatureExtractor,
        target_dim: int = 768,
        num_tokens: int = 197  # Compatible with ViT (14x14 + 1 CLS)
    ):
        """
        Initialize adapter.
        
        Args:
            yolo_extractor: YOLO feature extractor instance
            target_dim: Target dimension for each token
            num_tokens: Number of output tokens (including CLS token)
        """
        super().__init__()
        
        self.yolo = yolo_extractor
        self.target_dim = target_dim
        self.num_tokens = num_tokens
        
        # Create projection for each scale
        self.scale_projections = nn.ModuleList([
            nn.Conv2d(dim, target_dim, kernel_size=1)
            for dim in self.yolo.feature_dims
        ])
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, target_dim))
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, target_dim))
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass converting multi-scale features to token sequence.
        
        Args:
            pixel_values: Input images [B, C, H, W]
            
        Returns:
            Token sequence [B, num_tokens, target_dim]
        """
        outputs = self.yolo(pixel_values)
        batch_size = pixel_values.shape[0]
        
        # Project and pool each scale
        tokens_list = []
        for features, proj in zip(outputs['multi_scale_features'], self.scale_projections):
            # Project: [B, C_in, H, W] -> [B, target_dim, H, W]
            projected = proj(features)
            # Pool: [B, target_dim, H, W] -> [B, target_dim]
            pooled = nn.functional.adaptive_avg_pool2d(projected, (1, 1)).flatten(1)
            tokens_list.append(pooled)
        
        # Stack scale tokens: [B, num_scales, target_dim]
        scale_tokens = torch.stack(tokens_list, dim=1)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Combine: [B, num_scales + 1, target_dim]
        tokens = torch.cat([cls_tokens, scale_tokens], dim=1)
        
        # Pad or truncate to num_tokens if needed
        if tokens.shape[1] < self.num_tokens:
            padding = torch.zeros(
                batch_size, self.num_tokens - tokens.shape[1], self.target_dim,
                device=tokens.device, dtype=tokens.dtype
            )
            tokens = torch.cat([tokens, padding], dim=1)
        elif tokens.shape[1] > self.num_tokens:
            tokens = tokens[:, :self.num_tokens]
        
        # Add position embeddings
        tokens = tokens + self.pos_embed[:, :tokens.shape[1]]
        
        return tokens
