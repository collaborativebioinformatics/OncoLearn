import logging
import warnings
from pathlib import Path

from bmfm_targets.datasets.base_rna_dataset import BaseRNAExpressionDataset
from bmfm_targets.datasets.dataset_transformer import DatasetTransformer
from bmfm_targets.training.data_module import DataModule

logging.basicConfig(
    level=logging.INFO,
    filename="cellxgene_dataset.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CellXGeneDataset(BaseRNAExpressionDataset):
    """
    A PyTorch Dataset for cellXgene h5ad files.

    For more information on the cellXgene resources see:
    [https://github.ibm.com/BiomedSciAI-Innersource/bmfm-targets/tree/main/benchmarks/cell_type_prediction/datasets/CellXGene]

    The cellXgene metadata includes the following information (all fields are detailed in: [https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#cell_type_ontology_term_id](data_schema)):
    - soma_joinid: ID from joining multiple data sources
    - dataset_id
    - assay: [https://cellxgene.cziscience.com/docs/04__Analyze%20Public%20Data/4_2__Gene%20Expression%20Documentation/4_2_3__Gene%20Expression%20Data%20Processing](sequencing assay)
    - assay_ontology_term_id: [https://cellxgene.cziscience.com/docs/04__Analyze%20Public%20Data/4_2__Gene%20Expression%20Documentation/4_2_3__Gene%20Expression%20Data%20Processing](sequencing assay EFO ontology )
    - cell_type: [https://cellxgene.cziscience.com/docs/04__Analyze%20Public%20Data/4_2__Gene%20Expression%20Documentation/4_2_2__Cell%20Type%20and%20Gene%20Ordering](cell type by the original data contributers)
    - cell_type_ontology_term_id: [https://cellxgene.cziscience.com/docs/04__Analyze%20Public%20Data/4_2__Gene%20Expression%20Documentation/4_2_2__Cell%20Type%20and%20Gene%20Ordering](the closest sell type ntolgy term)
    - development_stage: Cells developmental stage
    - development_stage_ontology_term_id:
    - disease
    - disease_ontology_term_id
    - donor_id
    - is_primary_data: True if this is the canonical instance of this cellular observation and False if not
    - self_reported_ethnicity
    - self_reported_ethnicity_ontology_term_id
    - sex
    - sex_ontology_term_id
    - suspension_type
    - tissue
    - tissue_ontology_term_id
    - tissue_general: broader category of the tissue
    - tissue_general_ontology_term_id
    """

    DATASET_NAME = "cellxgene"
    source_h5ad_file_name = "cellxgene.h5ad"
    default_label_dict_path = Path(__file__).parent / f"{DATASET_NAME}_all_labels.json"

    def __post_init__(self):
        warnings.warn(
            "This object is no longer supported and will be removed. Please use CellXGeneSOMADataset",
            DeprecationWarning,
        )


class CellXGeneDataModule(DataModule):
    """PyTorch Lightning DataModule for cellXgene dataset."""

    DATASET_FACTORY = CellXGeneDataset
    DATASET_TRANSFORMER_FACTORY = DatasetTransformer
