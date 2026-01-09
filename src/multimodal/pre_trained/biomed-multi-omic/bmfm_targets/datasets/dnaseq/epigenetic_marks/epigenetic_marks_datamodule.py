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


class DNASeqEpigeneticMarksDataset(BaseDNASeqDataset):
    DATASET_NAME = "Epigenetic_Marks"
    default_label_dict_path = (
        Path(__file__).parent / f"{DATASET_NAME.lower()}_all_labels.json"
    )


class DNASeqEpigeneticMarksDataModule(DNASeqDataModule):
    DATASET_FACTORY: Dataset = DNASeqEpigeneticMarksDataset
