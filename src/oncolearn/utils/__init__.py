"""
Utilities module.
"""

# Conditionally import torch-dependent modules
try:
    from .data_loader import MedicalImageDataset, create_data_loaders
    from .metrics import MetricsTracker, compute_metrics
    from .visualization import plot_training_curves, visualize_attention
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

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
