"""
OncoLearn data loading pipeline DSL.

Provides a SQL-inspired, stack-based DSL for declaring multi-source, multi-modal
data loading pipelines in plain Python files.

Example pipeline file::

    from oncolearn.data.pipeline import DataSource, Load, Join, Sequence, Modality, Dataset
    from oncolearn.data.pipeline.transforms import map_ajcc_stage

    src = DataSource(config="data/configs/sources/cbioportal/brca_tcga.yaml", base_dir="data/sources/cbioportal")

    dataset = Dataset(
        modalities=[
            Modality(
                name="oncolearn.modality.clinical",
                pipeline=Load("clinical_patient", source=src),
                label_col="AJCC_PATHOLOGIC_TUMOR_STAGE",
                label_transform=map_ajcc_stage,
            ),
        ]
    )
"""
from .nodes import (
    BaseModality,
    DataSource,
    Dataset,
    ImageModality,
    Join,
    Load,
    Log2Normalization,
    Sequence,
    TabularModality,
)
from .executor import run

__all__ = [
    "BaseModality",
    "DataSource",
    "Dataset",
    "ImageModality",
    "Join",
    "Load",
    "Log2Normalization",
    "Sequence",
    "TabularModality",
    "run",
]
