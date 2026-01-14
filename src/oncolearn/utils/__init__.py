"""
Utilities module.
"""

from .config import Config, load_config, save_config

# Conditionally import torch-dependent modules
try:
    from .data_loader import MedicalImageDataset, create_data_loaders
    from .metrics import MetricsTracker, compute_metrics
    from .visualization import plot_training_curves, visualize_attention
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Always available
__all__ = [
    'Config',
    'load_config',
    'save_config',
]

# Add torch-dependent exports if available
if _TORCH_AVAILABLE:
    __all__.extend([
        'MedicalImageDataset',
        'create_data_loaders',
        'compute_metrics',
        'MetricsTracker',
        'plot_training_curves',
        'visualize_attention'
    ])
