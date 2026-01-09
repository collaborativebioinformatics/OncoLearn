import logging

from torch.utils.data import DataLoader, Dataset

from bmfm_targets.datasets.panglaodb import PanglaoDBDataset
from bmfm_targets.training.data_module import DataModule

logging.basicConfig(
    level=logging.INFO,
    filename="panglaodb_dataset.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PanglaoDBDataModule(DataModule):
    """PyTorch Lightning DataModule for PanglaoDB dataset."""

    DATASET_FACTORY: Dataset = PanglaoDBDataset

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def prepare_data(self) -> None:
        return

    def _prepare_dataset_kwargs(self):
        if self.limit_genes is not None:
            if self.dataset_kwargs is None:
                self.dataset_kwargs = {}
            self.dataset_kwargs["limit_genes"] = self._get_limited_gene_list(
                self.limit_genes
            )
        return self.dataset_kwargs
