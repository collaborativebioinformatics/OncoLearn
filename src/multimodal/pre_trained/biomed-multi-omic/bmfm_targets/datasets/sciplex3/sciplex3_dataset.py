import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
from anndata import AnnData, read_h5ad
from scipy.sparse import csr_matrix

from bmfm_targets.datasets.base_rna_dataset import (
    BaseRNAExpressionDataset,
    multifield_instance_wrapper,
)
from bmfm_targets.datasets.dataset_transformer import DatasetTransformer
from bmfm_targets.datasets.datasets_utils import random_subsampling
from bmfm_targets.tokenization.resources import get_protein_coding_genes
from bmfm_targets.training.data_module import DataModule
from bmfm_targets.transforms.compose import Compose
from bmfm_targets.transforms.sc_transforms import make_transform

logging.basicConfig(
    level=logging.INFO,
    filename="sciplex3_dataset.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

default_transforms = [
    {"transform_name": "ScaleCountsTransform", "transform_args": {"scale_factor": 10}},
    {
        "transform_name": "BinTransform",
        "transform_args": {"binning_method": "int_cast"},
    },
    {"transform_name": "FilterCellsTransform", "transform_args": {"min_counts": 2}},
    {"transform_name": "FilterGenesTransform", "transform_args": {"min_counts": 1}},
]


class SciPlex3Dataset(BaseRNAExpressionDataset):
    """
    A PyTorch Dataset for the sciplex3 drug response dataset.

    Source paper:
    Sanjay R. Srivatsan et al., Massively multiplex chemical transcriptomics at
    single-cell resolution. Science 367, 45-51(2020).
    https://www.science.org/doi/10.1126/science.aax6234
    SI https://www.science.org/doi/suppl/10.1126/science.aax6234/suppl_file/aax6234-srivatsan-sm.pdf
    (necessary for understanding the meaning of the fields)
    GEO https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944

    h5ad file prepared by
    Hetzel, L. et al. Predicting cellular responses to novel drug perturbations at a
    single-cell resolution. In Proceedings of 36th Conference on
    Neural Information Processing Systems (eds Koyejo, S. et al.)
    26711-26722 (Curran Associates, 2023).
    and downloaded from
    https://f003.backblazeb2.com/file/chemCPA-datasets/sciplex_complete_middle_subset.h5ad

    Note that this file limits the genes to 2000 based on HVG/DEG considerations, see
    the paper.

    Contains the following labels:

      - cell_type (str) cell line name
         counts: {'MCF7': 177730, 'K562': 89199, 'A549': 87711}

    Proliferation measures (derived from expression levels):
      - g1s_score (float) aggregated normalized expression score of G1/S marker genes.
         count when rounded to ints: {0: 59040, 1: 110835, 2: 131593, 3: 50871, 4: 2298, 5: 3}
      - g2m_score (float) aggregated normalized expression score of G2/M marker genes.
        count when rounded to ints: {0: 39626, 1: 92889, 2: 143724, 3: 72046, 4: 6346, 5: 9}
      - proliferation_index (float) log sum of g1s_score and g2m_score
        count when rounded to ints: {0: 17667, 1: 48872, 2: 126576, 3: 133503, 4: 28000, 5: 22}


    Drug identifier labels:
      - product_name (str) one of 187 drug names + control
          counts: ~750-~2,500 of each drug and 13,000 control
      - SMILES (str) SMILES string of molecule or 'CS(C)=O' if control
      - condition (str) exactly the same as product name except here 'JQ1' is
          replaced by '(+)-JQ1' in all samples where it occurs
      - control, vehicle (int) 1 if control, 0 if treated
      - dose, dose_character (float, str) dosage as float or str
         counts: {'10000': 107323, '1000': 99276, '100': 73830, '10': 61207, '0': 13004}
      - dose_pattern (str) UNKNOWN
         counts: {'1': 110484, '2': 102571, '3': 77117, '4': 64468}
      - dose_val (float) dosage rescaled between 0 and 1, with control set to 1 also

    Drug biology labels:
      - target (str) the target of the drug
        counts: {'HDAC': 41385, 'JAK': 31399, 'Aurora Kinase': 18659, 'DNA/RNA Synthesis': 15401,
        'Histone Methyltransferase': 14788, 'PARP': 12783, 'Sirtuin': 10234, 'NA': 9717,
        'DNA alkylator': 6473, 'Epigenetic Reader Domain': 5859, 'MEK': 5763, 'HIF': 5735,
        'Others': 5726, 'DNA Methyltransferase': 5609, 'Topoisomerase': 5437, 'Bcl-2': 5203,
        'CDK': 4602, 'Histone Demethylase': 4410, 'Glucocorticoid Receptor': 4320,
        'Beta Amyloid,Gamma-secretase': 4080, 'EGFR': 3859, 'Estrogen/progestogen Receptor': 3807,
        'EGFR,HER2': 3694, 'VEGFR': 3433, 'HSP (e.g. HSP90)': 3403, 'Aurora Kinase,FLT3,VEGFR': 3290,
        'Vehicle': 3287, 'PKC': 3082, 'MAO': 2262, 'NF-κB,HDAC,Histone Acetyltransferase,Nrf2': 2246,
        'Histone Acetyltransferase': 2175, 'EGFR,JAK': 2170, 'Autophagy,ROCK': 2155,
        'PKA,EGFR,PKC': 2153, 'GABA Receptor,HDAC,Autophagy': 2151, 'AMPK': 2138,
        'Lipoxygenase': 2126, 'Androgen Receptor': 2121, 'STAT': 2106, 'Aromatase': 2100,
        'MT Receptor': 2090, 'Telomerase': 2068, 'VEGFR,PDGFR,c-Kit': 2065,
        'E3 Ligase ,p53': 2037, 'CCR': 2033, 'PI3K': 2031, 'LPA Receptor': 2030,
        'Src,Sirtuin,PKC,PI3K': 2005, 'Dopamine Receptor': 2003, 'E3 Ligase ,TNF-alpha': 2002,
        'Tie-2': 1991, 'c-Kit,PDGFR,VEGFR': 1990, 'FGFR,VEGFR': 1970, 'Dehydrogenase': 1917,
        'CSF-1R,PDGFR,VEGFR': 1916, 'TGF-beta/Smad': 1911, 'TNF-alpha': 1904,
        'Glucocorticoid Receptor,Immunology & Inflammation related': 1897, 'COX': 1879,
        'EGFR,HDAC,HER2': 1860, 'Bcr-Abl': 1855, 'c-Met,Tie-2,VEGFR': 1852,
        'Autophagy,Sirtuin': 1844, 'Histamine Receptor': 1814, 'Pim': 1798,
        'Aurora Kinase,CDK': 1764, 'PDGFR,Raf,VEGFR': 1761, 'FAAH': 1747,
        'Aurora Kinase,VEGFR': 1733, 'FGFR,PDGFR,VEGFR': 1725, 'Aurora Kinase,Bcr-Abl,FLT3': 1724,
        'IGF-1R': 1724, 'Wnt/beta-catenin': 1704, 'Microtubule Associated': 1691,
        'c-RET,FLT3,JAK': 1677, 'ALK,c-Met': 1607, 'c-Met,IGF-1R,Trk receptor': 1561,
        'c-RET,VEGFR': 1548, 'FAK': 1528, 'IDH1': 1502, 'Bcr-Abl,c-Kit,Src': 1481, 'Src': 1409,
        'Aurora Kinase,Bcr-Abl,c-RET,FGFR': 1403, 'HDAC,PI3K': 1375, 'mTOR': 1365,
        'Aurora Kinase,Bcr-Abl,JAK': 1072, 'PLK': 1012, 'Survivin': 424}
      - pathway (str)
         counts: {'Epigenetics': 73642, 'DNA Damage': 52477, 'JAK/STAT': 37965,
         'Cell Cycle': 31328, 'Protein Tyrosine Kinase': 28965, 'Angiogenesis': 18992,
         'Cytoskeletal Signaling': 17770, 'Endocrinology & Hormones': 12051,
         'NA': 9717, 'Apoptosis': 9568, 'Metabolism': 9486, 'Neuronal Signaling': 7847,
         'Others': 7538, 'MAPK': 7524, 'PI3K/Akt/mTOR': 5534, 'TGF-beta/Smad': 4993,
         'GPCR & G Protein': 4120, 'Proteases': 4080, 'Ubiquitin': 4019,
         'Vehicle': 3287, 'Microbiology': 2033, 'Stem Cells &  Wnt': 1704}
      - pathway_level_1 (str)
         counts: {'Epigenetic regulation': 89486, 'Tyrosine kinase signaling': 50480,
         'JAK/STAT signaling': 42290, 'DNA damage & DNA repair': 36244,
         'Cell cycle regulation': 31998, 'Vehicle': 13004, 'Other': 12195,
         'Nuclear receptor signaling': 12145, 'Protein folding & Protein degradation': 11389,
         'Metabolic regulation': 10974, 'Neuronal signaling': 8559, 'Antioxidant': 8188,
         'Apoptotic regulation': 7664, 'HIF signaling': 5735, 'TGF/BMP signaling': 5386,
         'PKC signaling': 5220, 'Focal adhesion signaling': 3683}
      - pathway_level_2 (str)
         counts: {'Histone deacetylation': 58849, 'RTK activity': 36306,
         'JAK kinase activity': 34187, 'Aurora kinase activity': 23941,
         'Histone methylation': 14965, 'ADP-rybosilation': 14487, 'Nucleotide analog': 13491,
         'Vehicle': 13004, 'Nuclear receptor activity': 12145, 'Antioxidant': 8188,
         'MAPK activity': 7524, 'DNA methylation': 7517, 'Alkylating agent': 6379,
         'CDK activity': 5645, 'Abl/Src activity': 4745, 'PI3K-AKT-MTOR activity': 4408,
         'Gamma secretase activity': 4080, 'HIF prolyl-hydroxylation': 4065,
         'Mitochondria-mediated apoptosis': 3966, 'E3 ubiquitin ligase activity': 3906,
         'Bromodomain': 3815, 'HSP90 activity': 3403, 'Tyrosine kinase activity': 3224,
         'PKC activitiy': 3082, 'Oxidative deamination activity': 2262,
         'Anti-inflammatory': 2246, 'Catecholamine degradation': 2204,
         'Bacterial topoisomerase activity': 2193, 'Histone acetylation': 2175,
         'Histone demethylase': 2165, 'ROCK activity': 2155, 'AMPK activity': 2138,
         'Lypoxigenase activity': 2126, 'STAT activity': 2106, 'Aromatase activity': 2100,
         'MT receptor activity': 2090, 'Oxidizing agent': 2067, 'TP53 degradation': 2037,
         'Chemokine Receptor activity': 2033, 'LPA receptor activity': 2030,
         'Dopamine receptor activity': 2003, 'Aldehyde dehydrogenase activity': 1917,
         'Activin receptor type activity': 1911, 'Crosslinking agent': 1910,
         'Cyclooxigenase activity': 1879, 'Histamine receptor activity': 1814,
         'PIM1 activity': 1798, 'Fatty acid amide hydrolase activity': 1747,
         'GPCR activity': 1670, 'Mitochondrial pathway of apoptosis': 1661,
         'Dimethylallyl trans transferase activity': 1582,
         'Isocitrate dehydrogenase activity': 1502, 'Cell cycle regulation': 1473,
         'Toposiomerase activity': 1385, 'Spindle formation': 939}

    Experiment labels:
      - batch (int)
        counts: {'0': 71213, '1': 71190, '3': 70878, '2': 70730, '4': 70629}
      - n_counts (float/int) number of UMI counts per cell between 500 and 15,000
      - size_factor (float) Size factors were calculated as the log UMI counts observed in a single
        cell divided by the geometric mean of log UMI counts from all measured cells.
        count when rounded to ints: {0: 56767, 1: 201888, 2: 56835, 3: 20081, 4: 8940, 5: 4327,
        6: 2271, 7: 1347, 8: 755, 9: 511, 10: 273, 11: 188, 12: 131, 13: 82, 14: 58, 15: 46,
        16: 28, 17: 24, 18: 18, 19: 16, 20: 13, 21: 10, 22: 8, 23: 5, 24: 2, 25: 7, 26: 2,
        27: 1, 31: 1, 33: 1, 35: 2, 39: 1, 52: 1}
      - replicate
        counts: {'rep2': 192654, 'rep1': 161986}

    Split columns:
      - split_cellcycle_ood (str)
         counts: {'train': 345032, 'test': 5559, 'odd': 4049}
      - split_epigenetic_ood (str)
         counts: {'train': 330572, 'test': 16351, 'odd': 7717}
      - split_ho_epigenetic (str)
         counts: {'train': 290889, 'test': 50772, 'ood': 12979}
      - split_ho_epigenetic_all (str)
         counts: {'train': 275955, 'test': 49694, 'ood': 28991}
      - split_ho_pathway (str)
         counts: {'train': 292283, 'test': 51798, 'ood': 10559}
      - split_ood_finetuning (str)
         counts: {'train': 312405, 'test': 28784, 'ood': 13451}
      - split_random (str)
         counts: {'train': 248202, 'test': 53460, 'ood': 52978}
      - split_tyrosine_ood (str)
         counts: {'train': 338824, 'test': 8644, 'odd': 7172}

    Composite columns (concatenations of other columns):
        drug_dose_name                                               Enzastaurin_0.1
        cov_drug_dose_name                                      A549_Enzastaurin_0.1
        cov_drug                                                    A549_Enzastaurin
        product_dose                                     Enzastaurin (LY317615)_1000
    """

    DATASET_NAME = "sciplex3"
    source_h5ad_file_name = "sciplex3.h5ad"

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
            "transform_args": {"min_counts": 50},
        },
        {
            "transform_name": "FilterCellsTransform",
            "transform_args": {"max_counts": 150},
        },
        {
            "transform_name": "FilterCellsTransform",
            "transform_args": {"min_genes": 50},
        },
        {
            "transform_name": "FilterCellsTransform",
            "transform_args": {"max_genes": 700},
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
    default_label_dict_path = Path(__file__).parent / f"{DATASET_NAME}_all_labels.json"

    def __init__(
        self,
        data_dir: str | Path | None = None,
        split: str | None = None,
        source_h5ad_file_name: str | None = None,
        split_column: str | None = None,
        transforms: list[dict] | None = None,
        label_dict_path: str | None = None,
        label_columns: list[str] | None = None,
        regression_label_columns: list[str] | None = None,
        limit_samples: int | None = None,
        limit_samples_shuffle: bool = True,
        expose_zeros: str | None = None,
        output_wrapper: Callable | None = None,
        filter_query: str | None = None,
        limit_genes: list[str] | None = None,
    ) -> None:
        """
        Initializes the dataset.

        Args:
        ----
            data_dir (str | Path): Path to the directory containing the data.
                Either `data_dir` or `source_h5ad_file_name` must be provided.
            split (str): Split to use. Must be one of train, dev, test or None to get all splits.
            source_h5ad_file_name (str): h5ad file name if using one other than the default
            split_column (str): split column name eg "split_random"
            transforms (list[dict] | None, optional): List of transforms to be applied to the datasets.
            label_dict_path: str | None = None
            label_columns: str | None = None
            limit_samples: int limit dataset to this many samples
            limit_samples_shuffle: bool shuffle dataset while limiting

        Raises:
        ------
            ValueError: If the split is not one of train, dev, test or None.
            FileNotFoundError: If the input data file (in h5ad format) does not exist.
            ValueError: If neither `data_dir` nor `source_h5ad_file_name` are provided.
        """
        if source_h5ad_file_name is not None:
            self.h5ad_data_path = source_h5ad_file_name
        elif data_dir is not None:
            data_dir = Path(data_dir)
            self.h5ad_data_path = data_dir / "h5ad" / self.source_h5ad_file_name
        else:
            raise ValueError(
                "Either `data_dir` or `source_h5ad_file_name` must be provided."
            )
        self.split_renamer = {"train": "train", "dev": "test", "test": "ood"}
        self.split_column_name = split_column
        self.split = None
        if split is not None:
            self.split = self.split_renamer[split]
        self.expose_zeros = expose_zeros
        self.limit_genes = limit_genes
        self.filter_query = filter_query
        self.label_dict_path = label_dict_path
        if self.label_dict_path is None:
            self.label_dict_path = self.default_label_dict_path
        self.label_dict = {}
        self.label_columns = label_columns
        self.regression_label_columns = regression_label_columns
        transforms = transforms if transforms is not None else default_transforms
        self.transforms = (
            Compose([make_transform(**d) for d in transforms]) if transforms else None
        )
        if expose_zeros not in ["all", None]:
            raise NotImplementedError("Unsupported option for exposing zeros")
        if split not in ["train", "dev", "test"] and split is not None:
            raise ValueError("Split must be one of train, dev, test")

        if not os.path.exists(self.h5ad_data_path):
            raise FileNotFoundError(
                str(self.h5ad_data_path) + " input data file does not exist",
            )
        self.processed_data = self.process_datasets()
        self.label_dict = self.get_label_dict(self.label_dict_path)
        self.processed_data = self.filter_data(self.processed_data)

        self.metadata = self.processed_data.obs
        self.binned_data = self.processed_data.X
        self.all_genes = self.processed_data.var_names
        self.cell_names = np.array(self.processed_data.obs_names)

        if limit_samples is not None:
            self.processed_data = random_subsampling(
                adata=self.processed_data,
                n_samples=limit_samples,
                shuffle=limit_samples_shuffle,
            )

        if output_wrapper is None:
            self.output_wrapper = multifield_instance_wrapper
        else:
            self.output_wrapper = output_wrapper

    def process_datasets(self) -> AnnData:
        """
        Processes the datasets by applying the pre-transforms and
        concatenating the datasets.

        Returns
        -------
            AnnData: Processed data.
        """
        raw_data = read_h5ad(self.h5ad_data_path)

        if self.transforms is not None:
            processed_data = self.transforms(adata=raw_data)["adata"]
        else:
            processed_data = raw_data

        if not isinstance(processed_data.X, csr_matrix):
            if isinstance(processed_data.X, np.ndarray):
                processed_data.X = csr_matrix(processed_data.X)
            else:
                processed_data.X = processed_data.X.tocsr()

        return processed_data


class SciPlex3DataModule(DataModule):
    """PyTorch Lightning DataModule for Human Cell Atlas dataset."""

    DATASET_FACTORY = SciPlex3Dataset
    DATASET_TRANSFORMER_FACTORY = DatasetTransformer

    def prepare_data(self) -> None:
        return

    def _prepare_dataset_kwargs(self):
        final_dataset_kwargs = {}
        if self.dataset_kwargs:
            final_dataset_kwargs = {**self.dataset_kwargs}
        final_dataset_kwargs["limit_samples_shuffle"] = self.shuffle
        if self.label_columns:
            final_dataset_kwargs["label_columns"] = [
                label.label_column_name
                for label in self.label_columns
                if not label.is_regression_label
            ]
            final_dataset_kwargs["regression_label_columns"] = [
                label.label_column_name
                for label in self.label_columns
                if label.is_regression_label
            ]
        if self.limit_genes is not None:
            if self.limit_genes == "tokenizer":
                final_dataset_kwargs["limit_genes"] = [
                    *self.tokenizer.get_field_vocab("genes")
                ]
            elif self.limit_genes == "protein_coding":
                pc_genes = get_protein_coding_genes()
                final_dataset_kwargs["limit_genes"] = pc_genes
            else:
                raise ValueError("Unsupported option passed for limit_genes")
        return final_dataset_kwargs
