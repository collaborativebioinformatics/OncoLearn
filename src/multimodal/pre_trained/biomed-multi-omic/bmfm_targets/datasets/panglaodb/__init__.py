"""
The single_cell_rna package consists of modules for processing single-cell
RNA-seq data and converting them to pytorch datasets for use in training neural
networks.

The package is organized into the following modules:
    - panglaodb_dataset: a module for creating pytorch datasets from the PanglaoDB dataset
    - streaming_panglaodb_dataset: a module for creating litdata based streaming pytorch datasets from the PanglaoDB dataset
    - panglaodb_data_module: a module for creating pytorch lightning data modules from the PanglaoDB dataset
    - panglaodb_converter: a module for converting rdata files from the PanglaoDB dataset to h5ad files
    - panglao_metadata_util: a module for creating metadata files for the PanglaoDB dataset
"""

from .panglaodb_dataset import PanglaoDBDataset
from .panglaodb_data_module import PanglaoDBDataModule
from .streaming_panglaodb_dataset import (
    StreamingPanglaoDBDataset,
    StreamingPanglaoDBDataModule,
)

__all__ = [
    "PanglaoDBDataset",
    "PanglaoDBDataModule",
    "StreamingPanglaoDBDataset",
    "StreamingPanglaoDBDataModule",
]
