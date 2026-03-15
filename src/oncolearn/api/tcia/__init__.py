"""
TCIA (The Cancer Imaging Archive) data module.

This module provides utilities for downloading TCIA manifest files and imaging data
for TCGA cohorts using a configuration-driven approach.
"""

from .builder import TCIACohortBuilder
from .tcia_dataset import TCIADataset

__all__ = ["TCIACohortBuilder", "TCIADataset"]
