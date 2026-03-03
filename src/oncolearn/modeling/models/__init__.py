"""Registered OncoLearn model classes (pl.LightningModule).

Importing this package triggers all @register_model decorators.
"""
from .gated_late_fusion import GatedLateFusionClassifier

__all__ = ["GatedLateFusionClassifier"]
