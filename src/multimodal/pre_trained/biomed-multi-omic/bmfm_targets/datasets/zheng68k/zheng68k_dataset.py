import logging
from pathlib import Path

import pandas as pd

from bmfm_targets.datasets.base_rna_dataset import BaseRNAExpressionDataset
from bmfm_targets.datasets.dataset_transformer import DatasetTransformer
from bmfm_targets.training.data_module import DataModule

logging.basicConfig(
    level=logging.INFO,
    filename="Zheng68k_dataset.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from bmfm_targets.datasets.datasets_utils import (
    load_extra_metadata_mapping,
)


class Zheng68kDataset(BaseRNAExpressionDataset):
    """
    A PyTorch Dataset for Zheng68k h5ad files.

    Attributes
    ----------
    data_dir (str | Path): Path to the directory containing the data.
            split (str): Split to use. Must be one of train, dev, test.
            split_params (dict, optional): _description_. Defaults to default_split.
            transforms (list[dict] | None, optional): List of transforms to be applied to the datasets.
    """

    DATASET_NAME = "zheng68k"
    source_h5ad_file_name = "zheng68k.h5ad"
    default_label_dict_path = Path(__file__).parent / "zheng68k_all_labels.json"
    DEFAULT_TRANSFORMS = [
        {
            "transform_name": "RenameGenesTransform",
            "transform_args": {
                "gene_map": None,
            },
        },
        {
            "transform_name": "KeepGenesTransform",
            "transform_args": {"genes_to_keep": None},
        },
        {
            "transform_name": "FilterCellsTransform",
            "transform_args": {"min_counts": 700},
        },
        {
            "transform_name": "FilterCellsTransform",
            "transform_args": {"max_counts": 2500},
        },
        {
            "transform_name": "FilterCellsTransform",
            "transform_args": {"min_genes": 300},
        },
        {
            "transform_name": "FilterCellsTransform",
            "transform_args": {"max_genes": 1000},
        },
        {
            "transform_name": "QcMetricsTransform",
            "transform_args": {"pct_counts_mt": 7},
        },
        {
            "transform_name": "NormalizeTotalTransform",
            "transform_args": {
                "exclude_highly_expressed": False,
                "max_fraction": 0.05,
                "target_sum": 10000.0,
            },
        },
        {
            "transform_name": "LogTransform",
            "transform_args": {"base": 2, "chunk_size": None, "chunked": None},
        },
        {
            "transform_name": "BinTransform",
            "transform_args": {"num_bins": 10, "binning_method": "int_cast"},
        },
    ]


def add_zheng_celltype_ontology(metadata_df: pd.DataFrame):
    """
    Add the cellxgene celltype ontology to zheng68k obs.

    Args:
    ----
        metadata_df (pd.DataFrame): _description_

    Returns:
    -------
        metadata_df: pd.DataFrame with additional column mapped
    """
    label_column_name = "celltype"
    mapping_variable = "cell_type_ontology_term_id"
    label_map = load_extra_metadata_mapping(
        dataset_name="zheng68k",
        mapping_key=label_column_name,
        mapping_value=mapping_variable,
    )
    mapped_label_col = metadata_df[label_column_name].map(label_map)
    named_column_dict = {mapping_variable: mapped_label_col}
    return metadata_df.assign(**named_column_dict)


class Zheng68kDataModule(DataModule):
    """PyTorch Lightning DataModule for Zheng68k dataset."""

    DATASET_FACTORY = Zheng68kDataset
    DATASET_TRANSFORMER_FACTORY = DatasetTransformer
