"""
Image preprocessing transforms for DICOM images (V1 only).
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T


class DICOMToTensor:
    """Convert DICOM pixel array to tensor with normalization."""
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
    
    def __call__(self, pixel_array: np.ndarray) -> torch.Tensor:
        """
        Args:
            pixel_array: (H, W) numpy array
        
        Returns:
            Tensor (1, H, W) or (3, H, W) if normalized
        """
        tensor = torch.from_numpy(pixel_array).float()
        
        # Handle different input shapes
        if tensor.dim() == 2:
            # (H, W) -> (1, H, W)
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 3:
            # (C, H, W) or (H, W, C) - handle both cases
            if tensor.shape[0] < tensor.shape[2]:
                # Likely (H, W, C), transpose to (C, H, W)
                tensor = tensor.permute(2, 0, 1)
            # If already (C, H, W), take first channel if C > 3
            if tensor.shape[0] > 3:
                # Take first channel and treat as grayscale
                tensor = tensor[0:1]
        elif tensor.dim() > 3:
            # Flatten extra dimensions
            while tensor.dim() > 3:
                tensor = tensor.squeeze(0)
            # Ensure 2D then add channel
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
        
        # Ensure we have (C, H, W) format with C <= 3
        if tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor after processing, got {tensor.dim()}D with shape {tensor.shape}")
        
        # If more than 3 channels, take first 3
        if tensor.shape[0] > 3:
            tensor = tensor[:3]
        
        # Normalize to [0, 1] if requested
        if self.normalize:
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            if tensor_max > tensor_min:
                tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        
        # Convert to 3-channel if needed (for pretrained models)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.shape[0] == 2:
            # If 2 channels, repeat last channel
            tensor = torch.cat([tensor, tensor[-1:]], dim=0)
        
        return tensor


class ResizeDICOM:
    """Resize DICOM image to target size."""
    
    def __init__(self, size: int = 224):
        self.size = size
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: (C, H, W)
        
        Returns:
            Resized tensor (C, size, size)
        """
        if tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {tensor.dim()}D")
        
        # Use bilinear interpolation
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self.size, self.size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return tensor


class DICOMTransform:
    """Complete transform pipeline for DICOM images."""
    
    def __init__(
        self,
        size: int = 224,
        normalize: bool = True,
        augment: bool = False
    ):
        self.size = size
        self.normalize = normalize
        self.augment = augment
        
        # Base transforms
        self.to_tensor = DICOMToTensor(normalize=normalize)
        self.resize = ResizeDICOM(size=size)
        
        # Optional augmentation (conservative)
        if augment:
            self.augment_transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=5),
            ])
        else:
            self.augment_transform = None
    
    def __call__(self, pixel_array: np.ndarray) -> torch.Tensor:
        """
        Args:
            pixel_array: (H, W) numpy array
        
        Returns:
            Tensor (3, size, size)
        """
        tensor = self.to_tensor(pixel_array)
        tensor = self.resize(tensor)
        
        if self.augment_transform is not None:
            tensor = self.augment_transform(tensor)
        
        return tensor


def get_dicom_transforms(
    size: int = 224,
    normalize: bool = True,
    augment: bool = False
) -> DICOMTransform:
    """Get DICOM transform pipeline."""
    return DICOMTransform(size=size, normalize=normalize, augment=augment)


