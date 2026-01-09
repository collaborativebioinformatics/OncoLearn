import logging
from functools import partial
from pathlib import Path

import cellxgene_census as cc
import cellxgene_census.experimental.ml as census_ml
import numpy as np
import tiledbsoma as soma
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from bmfm_targets.datasets.cellxgene.cellxgene_soma_utils import get_split_value_filter
from bmfm_targets.tokenization import MultiFieldInstance
from bmfm_targets.training.data_module import DataModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("cellxgene_soma_dataset.log", mode="w"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class CellXGeneSOMADataset:
    """
    Dataset object built on top of CellXGene's SOMA TileDB database.

    For details see https://chanzuckerberg.github.io/cellxgene-census/index.html
    """

    def __init__(
        self,
        uri: (
            str | Path
        ) = "/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/cellxgene/soma-2023-12-15",
        value_filter="is_primary_data == True",
        split: str | None = "train",
        batch_size: int = 64,
        label_columns: list[str] | None = None,
        soma_chunk_size: int = 20_000,
        custom_split_file: str | None = None,
        limit_samples: int | None = None,
        limit_samples_shuffle: bool = False,
        shuffle: bool = True,
    ):
        """
        CellxGeneSOMADataset.

        Args:
        ----
        uri (str  |  Path, optional): Path to soma database. If `None`, will access the hosted
            version on AWS, which may be slow but can run from . Defaults to the downloaded copy
            on CCC at "/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/cellxgene/soma-2023-12-15".
        value_filter (str, optional): clause to filter the samples by, supports the columns
            in cellxgene standardized obs tables including cell_type, tissue_type, dataset_id etc.
            Defaults to "is_primary_data == True".
        split (str | None, optional): Split to use, must match a split in the split file,
            otherwise the complete dataset is used. Defaults to "train".
        batch_size (int, optional): Batch size. The batch size is set at the dataset level,
            not the dataloader level because of the underlying iterable data access.
            Defaults to 64.
        label_columns (list(str), optional): Label columns for MultiFieldInstance metadata.
            Defaults to ["cell_type"].
        soma_chunk_size (int, optional): Chunk size for reading ahead, each worker
            will use this much. Defaults to 20_000.
        custom_split_file (str | None, optional): Path to custom split file.
            If None, will use the default splits included in the package. Defaults to None.
        shuffle (bool): Flag designed for pytest proporses only to prevent shuffling during comparison
            of output from CellXGeneSOMADataset and CellXGeneNexusDataset.
        """
        value_filter += get_split_value_filter(split, custom_split_file)
        self.uri = uri
        self.census = cc.open_soma(uri=self.uri, census_version="2023-12-15")
        self.value_filter = value_filter
        self.label_columns = label_columns
        if self.label_columns is None:
            self.label_columns = ["cell_type"]
        self.soma_experiment = self.census["census_data"]["homo_sapiens"]

        obs_column_names = [*self.label_columns]
        if not "soma_joinid" in obs_column_names:
            obs_column_names.insert(0, "soma_joinid")
        self.experiment_datapipe = census_ml.ExperimentDataPipe(
            self.soma_experiment,
            measurement_name="RNA",
            X_name="normalized",
            obs_query=soma.AxisQuery(value_filter=value_filter),
            obs_column_names=obs_column_names,
            batch_size=batch_size,
            shuffle=shuffle,
            return_sparse_X=False,  # sparse breaks multiprocessing
            soma_chunk_size=soma_chunk_size,
        )
        self.all_genes = (
            self.soma_experiment.ms["RNA"]
            .var.read(column_names=["feature_name"])
            .concat()
            .to_pandas()
            .squeeze()
            .values
        )
        # copy property to variable
        obs_encoders: dict[str, LabelEncoder] = self.experiment_datapipe.obs_encoders

        label_encoders = {l: obs_encoders[l].classes_ for l in obs_column_names}
        self.label_dict = {
            label_name: {l: i for i, l in enumerate(labels)}
            for label_name, labels in label_encoders.items()
        }

        self.name_mapper = {
            label_name: dict(enumerate(labels))
            for label_name, labels in label_encoders.items()
        }

        convert_partial = partial(
            convert_to_multifield,
            all_genes=self.all_genes,
            name_mapper=self.name_mapper,
            label_columns=self.label_columns,
        )
        self.output_datapipe = self.experiment_datapipe.map(convert_partial)

        if limit_samples:
            logger.warning(
                "limit_samples is not recommended for CellXGeneSomaDataset, consider using CellXGeneNexusDataset"
            )
            self.output_datapipe = self.experiment_datapipe.header(limit_samples).map(
                convert_partial
            )
        if limit_samples_shuffle:
            logger.warning(
                "limit_samples_shuffle is not supported for CellXGeneSomaDataset"
            )


def convert_to_multifield(
    xy: tuple[torch.Tensor, torch.Tensor],
    all_genes: np.ndarray,
    name_mapper: dict[str, dict[int, str]],
    label_columns: list[str],
) -> list[MultiFieldInstance]:
    """
    Convert reads and labels tensor to list of MultiFieldInstances.

    This translates the data from the torch tensors supplied by census_ml to
    the MultiFieldInstances used by bmfm_targets.
    This function applies transforms similar to those in bmfm_targets.transforms

    Args:
    ----
        xy (tuple[torch.TensorType, torch.TensorType]): reads and labels. The first column
            of labels is the soma_id and the remaining columns are the requested labels as
            integers, which can be decoded by the experiment_datapipe.
        all_genes (np.ndarray): list of all genes to lookup the names for the nonzero reads
        name_mapper (dict[int, dict]): a dictionary mapping the label values to their string values

    Returns:
    -------
        list[MultiFieldInstance]: list of MultiFieldInstances for the batch
    """
    x, y = xy
    mfi_list = []
    for sample, labels in zip(x, y):
        if x.is_sparse:
            sample = sample.coalesce()
            genes = all_genes[sample.indices()].squeeze()
            expressions = sample.values().squeeze().numpy()
        else:
            nz = sample.nonzero()
            genes = all_genes[nz].squeeze()
            expressions = sample[nz].squeeze().numpy()
        mfi_data = {"genes": [*genes], "expressions": [*expressions]}

        mfi_metadata = {}
        if "soma_joinid" not in label_columns:
            label_columns = ["soma_joinid"] + label_columns
        for idx, lc in enumerate(label_columns):
            mfi_metadata[lc] = name_mapper[lc][labels[idx].item()]

        mfi_metadata["cell_name"] = mfi_metadata.pop("soma_joinid")
        mfi_list.append(MultiFieldInstance(data=mfi_data, metadata=mfi_metadata))
    return mfi_list


class CellXGeneSOMADataModule(DataModule):
    DATASET_FACTORY = CellXGeneSOMADataset

    def __post_init__(self):
        if self.dataset_kwargs is None:
            self.dataset_kwargs = {}
        self.dataset_kwargs["batch_size"] = self.batch_size
        if self.num_workers > 0:
            _init_multiprocessing()
        return

    def prepare_data(self) -> None:
        return

    def _prepare_dataset_kwargs(self):
        final_dataset_kwargs = {**self.dataset_kwargs}
        if self.label_columns:
            final_dataset_kwargs["label_columns"] = [
                label.label_column_name
                for label in self.label_columns
                if not label.is_regression_label
            ]
        return final_dataset_kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset.output_datapipe,
            num_workers=self.num_workers,
            batch_size=None,  # batching is handled by our ExperimentDataPipe
            collate_fn=self.collate_fn,
            shuffle=False,  # shuffling is handled by our ExperimentDataPipe
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dev_dataset.output_datapipe,
            num_workers=self.num_workers,
            batch_size=None,  # batching is handled by our ExperimentDataPipe
            collate_fn=self.collate_fn,
            shuffle=False,  # shuffling is handled by our ExperimentDataPipe
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset.output_datapipe,
            num_workers=self.num_workers,
            batch_size=None,  # batching is handled by our ExperimentDataPipe
            collate_fn=self.collate_fn,
            shuffle=False,  # shuffling is handled by our ExperimentDataPipe
        )


def _init_multiprocessing() -> None:
    """
    Ensures use of "spawn" for starting child processes with multiprocessing.
    Forked processes are known to be problematic:
      https://pytorch.org/docs/stable/notes/multiprocessing.html#avoiding-and-fighting-deadlocks
    Also, CUDA does not support forked child processes:
      https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing.

    This function is based on the function from census_ml.
    """
    torch.multiprocessing.set_start_method("fork", force=True)
    orig_start_method = torch.multiprocessing.get_start_method()
    if orig_start_method != "spawn":
        if orig_start_method:
            logger.warning(
                "switching torch multiprocessing start method from "
                f'"{torch.multiprocessing.get_start_method()}" to "spawn"'
            )
        torch.multiprocessing.set_start_method("spawn", force=True)
