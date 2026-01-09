import logging
from pathlib import Path

from torch.utils.data import Dataset

from bmfm_targets.datasets.base_dna_dataset import BaseDNASeqDataset
from bmfm_targets.training.data_module import DNASeqDataModule

logging.basicConfig(
    level=logging.INFO,
    filename="snpdb_dataset.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DNASeqMPRADataset(BaseDNASeqDataset):
    """
    A PyTorch Dataset for SNPdb RData files.

    Attributes
    ----------
        raw_data_dir (str | Path): Path to the directory containing the data.
        split (str): Split to use. Must be one of train, dev, test.
        num_workers (int): Number of workers to use for parallel processing.

    """

    DATASET_NAME = "mpra"
    default_label_dict_path = (
        Path(__file__).parent / f"{DATASET_NAME.lower()}_all_labels.json"
    )


class DNASeqMPRADataModule(DNASeqDataModule):
    DATASET_FACTORY: Dataset = DNASeqMPRADataset
