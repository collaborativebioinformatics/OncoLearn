from .gene import GeneDataModule          # noqa: F401 — triggers @register_modality("gene")
from .clinical import ClinicalDataModule  # noqa: F401 — triggers @register_modality("clinical")
from .base import TabularDataset

__all__ = ["GeneDataModule", "ClinicalDataModule", "TabularDataset"]
