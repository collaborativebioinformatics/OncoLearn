import logging

from bmfm_targets.datasets.base_rna_dataset import BaseRNAExpressionDataset
from bmfm_targets.datasets.dataset_transformer import DatasetTransformer
from bmfm_targets.training.data_module import DataModule

logging.basicConfig(
    level=logging.INFO,
    filename="bulk_rna_dataset.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BulkRnaDatasetLabeled(BaseRNAExpressionDataset):
    """A PyTorch Dataset for bulk RNA h5ad files."""

    DATASET_NAME = "bulk_rna"
    source_h5ad_file_name = "bulk_rna.h5ad"


class BulkRnaDatasetUnlabeled(BaseRNAExpressionDataset):
    """A PyTorch Dataset for bulk RNA h5ad files."""

    DATASET_NAME = "bulk_rna"
    source_h5ad_file_name = "bulk_rna.h5ad"

    def get_sample_metadata(self, idx):
        cell_name = str(self.cell_names[idx])
        metadata = {
            "cell_name": cell_name,
        }

        return metadata


class BulkRnaDataModuleUnlabeled(DataModule):
    """PyTorch Lightning DataModule for BulkRna dataset."""

    DATASET_FACTORY = BulkRnaDatasetUnlabeled
    DATASET_TRANSFORMER_FACTORY = DatasetTransformer


class BulkRnaDataModuleLabeled(DataModule):
    """PyTorch Lightning DataModule for BulkRna dataset."""

    DATASET_FACTORY = BulkRnaDatasetLabeled
    DATASET_TRANSFORMER_FACTORY = DatasetTransformer
