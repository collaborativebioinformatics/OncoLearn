import datetime
import logging
import mmap
import multiprocessing
import os
import shutil
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
from anndata import AnnData, concat, read_h5ad

from bmfm_targets.datasets.base_rna_dataset import (
    BaseRNAExpressionDataset,
    multifield_instance_wrapper,
)
from bmfm_targets.datasets.dataset_transformer import default_transforms
from bmfm_targets.datasets.datasets_utils import random_subsampling
from bmfm_targets.datasets.panglaodb.panglaodb_converter import (
    convert_all_rdatas_to_h5ad,
    create_sra_splits,
)
from bmfm_targets.tokenization import get_gene2vec_tokenizer
from bmfm_targets.transforms.compose import Compose
from bmfm_targets.transforms.sc_transforms import make_transform

logging.basicConfig(
    level=logging.INFO,
    filename="panglaodb_dataset.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

tokenizer = get_gene2vec_tokenizer()
genes_to_keep = tokenizer.get_field_vocab("genes")


class PanglaoDBDataset(BaseRNAExpressionDataset):
    """
    A PyTorch Dataset for PanglaoDB RData files.

    Attributes
    ----------
        data_dir (str | Path): Path to the directory containing the data.
        data_info_path (str | Path): Path to the metadata file containing details about datasets.
        split (str): Split to use. Must be one of train, dev, test.
        convert_rdata_to_h5ad (bool): Whether to convert the RData files to h5ad files.
        transform_datasets (bool): Whether to apply the transforms to the datasets.
        filter_query (str): Query to filter the datasets.
        pre_transforms (list[dict] | None): List of transforms to be applied before merging the datasets.
        post_transforms (list[dict] | None): List of transforms to be applied after merging the datasets.
        num_workers (int): Number of workers to use for parallel processing.
    """

    URL = "https://panglaodb.se/bulk.html"

    DATASET_NAME = "PanglaoDB"

    def __init__(
        self,
        data_dir: str | Path,
        data_info_path: str | Path,
        split: str,
        processed_dir_name: str = "processed",
        convert_rdata_to_h5ad: bool = False,
        transform_datasets: bool = False,
        filter_query: str | None = None,
        pre_transforms: list[dict] | None = None,
        post_transforms: list[dict] | None = None,
        num_workers: int = 0,
        limit_samples: int | None = None,
        limit_samples_shuffle: bool | None = None,
        limit_genes: list[str] | None = None,
        sort_genes_var: str | None = None,
        expose_zeros: Literal["all"] | None = None,
        output_wrapper: Callable | None = None,
    ) -> None:
        """
        Initializes the dataset.

        Args:
        ----
        data_dir (str | Path): Path to the directory containing the data.
        data_info_path (str | Path): Path to the metadata file containing details about datasets.
        split (str): Split to use. Must be one of train, dev, test.
        convert_rdata_to_h5ad (bool): Whether to convert the RData files to h5ad files.
        transform_datasets (bool): Whether to apply the transforms to the datasets.
        filter_query (str): Query to filter the datasets.
        pre_transforms (list[dict] | None): List of transforms to be applied before merging the datasets.
        post_transforms (list[dict] | None): List of transforms to be applied after merging the datasets.
        num_workers (int): Number of workers to use for parallel processing.  0 for work in the main thread
        limit_samples: (int | None) limit the number of samples, None to load all
        limit_samples_shuffle (bool | None) : shuffle the limited samples (random sub sampling)


        Raises:
        ------
            FileNotFoundError: If the RData directory does not exist.
            FileNotFoundError: If the data info file does not exist.
            FileNotFoundError: If the processed file does not exist.
            ValueError: If the split is not one of train, dev, test.
        """
        data_dir = Path(data_dir)
        self.rdata_dir = data_dir / "rdata"
        self.h5ad_dir = data_dir / "h5ad"
        self.processed_dir = data_dir / processed_dir_name
        self.data_info_path = Path(data_info_path)
        self.filter_query = filter_query
        self.expose_zeros = expose_zeros
        self.split = split
        self.label_columns = None
        self.regression_label_columns = None
        self.sort_genes_var = sort_genes_var
        self.limit_genes = limit_genes
        if pre_transforms is None:
            pre_transforms = default_transforms
        self.pre_transforms = (
            Compose([make_transform(**d) for d in pre_transforms])
            if pre_transforms
            else None
        )
        self.post_transforms = (
            Compose([make_transform(**d) for d in post_transforms])
            if post_transforms
            else None
        )
        self.num_workers = num_workers
        if expose_zeros not in ["all", None]:
            raise NotImplementedError(
                f"Unsupported option for exposing zeros: {expose_zeros}"
            )

        if not self.rdata_dir.exists():
            raise FileNotFoundError(
                "RData directory does not exist. Please download the RData file from {} manually.",
                format(self.URL),
            )

        if split not in ["train", "dev", "test"]:
            raise ValueError("Split must be one of train, dev, test")

        if convert_rdata_to_h5ad:
            if self.data_info_path.exists():
                sra_df_dict = create_sra_splits(
                    self.data_info_path, self.rdata_dir, self.filter_query
                )
                convert_all_rdatas_to_h5ad(
                    self.h5ad_dir, sra_df_dict, num_workers=self.num_workers
                )
            else:
                raise FileNotFoundError(
                    "Data info file is needed for converting the raw data to h5ad",
                    format(self.URL),
                )

        if transform_datasets:
            if not os.path.exists(self.processed_dir):
                os.mkdir(self.processed_dir)
            self.processed_data = self._process_datasets()

        else:
            processed_file = Path(self.processed_dir).joinpath(
                self.split, "processed_final.h5ad"
            )
            if not os.path.exists(processed_file):
                raise FileNotFoundError(
                    "Processed file does not exist. Please run the dataset with transform_datasets=True",
                )
            else:
                self.processed_data = read_h5ad(processed_file)

        if self.limit_genes is not None:
            self.processed_data = self.limit_data_to_gene_list(
                self.processed_data, self.limit_genes
            )
        self.binned_data = self.processed_data.X
        self.all_genes = np.array(self.processed_data.var_names)
        self.cell_names = np.array(self.processed_data.obs_names)
        self.metadata = self.processed_data.obs
        if output_wrapper is None:
            self.output_wrapper = multifield_instance_wrapper
        else:
            self.output_wrapper = output_wrapper
        if limit_samples:
            self.processed_data = random_subsampling(
                adata=self.processed_data,
                n_samples=limit_samples,
                shuffle=limit_samples_shuffle,
            )

    def get_vocab_for_field(self, field):
        if field == "expressions":
            return np.arange(0, np.max(self.binned_data.data) + 1)
        if field == "genes":
            return self.all_genes
        raise ValueError(f"Unknown field f{field}")

    def _process_datasets(self) -> AnnData:
        """
        Processes the datasets by applying the pre-transforms and
        concatenating the datasets.

        Returns
        -------
            AnnData: Processed data.
        """
        processed_file = self.processed_dir / self.split / "processed_final.h5ad"
        if not os.path.exists(Path(self.processed_dir) / self.split):
            os.mkdir(Path(self.processed_dir) / self.split)
        self._execute_pre_transforms_in_parallel(self.pre_transforms)
        processed_data = self._concatenate_h5ad_datasets()
        processed_data.write_h5ad(processed_file)
        return processed_data

    def _execute_pre_transforms_in_parallel(self, pre_transforms: Compose | None):
        """
        Executes the pre-transforms in parallel.

        Args:
        ----
            pre_transforms (Compose): Compose object containing transforms to be applied before merging the datasets.
        """
        h5ad_fnames = os.listdir(self.h5ad_dir / self.split)
        if len(h5ad_fnames) == 0:
            raise ValueError(
                f"Requested transform of non-existent h5ad data at {self.h5ad_dir / self.split}"
            )
        output_locations = [
            self.processed_dir / self.split / h5ad_fname for h5ad_fname in h5ad_fnames
        ]
        h5ad_locations = [
            self.h5ad_dir / self.split / h5ad_file_name
            for h5ad_file_name in h5ad_fnames
        ]
        partial_process_dataset = partial(
            self._process_dataset, pre_transforms=pre_transforms
        )
        if os.path.exists(self.processed_dir / self.split):
            shutil.rmtree(self.processed_dir / self.split)
        os.makedirs(self.processed_dir / self.split, exist_ok=True)
        if self.num_workers == 0:
            for h5ad_loc, output_loc in zip(h5ad_locations, output_locations):
                self._process_dataset(h5ad_loc, output_loc, pre_transforms)
        else:
            pool = multiprocessing.Pool(processes=self.num_workers)

            pool.starmap(partial_process_dataset, zip(h5ad_locations, output_locations))
            pool.close()
            pool.join()

    def _process_dataset(
        self,
        h5ad_location: str | Path,
        output_location: str | Path,
        pre_transforms: Compose | None,
    ):
        """
        Processes a single dataset by applying the pre-transforms.

        Args:
        ----
            h5ad_location (str): Path to the h5ad file.
            output_location (str): Path to the output file.
            pre_transforms (Compose): Compose object containing transforms to be applied before merging the datasets.
        """
        study_data = self._load_h5ad_mmap(h5ad_location)
        if pre_transforms is not None:
            study_data = pre_transforms(adata=study_data)["adata"]
        logger.info(f"shape of the dataset after pre-transforms {study_data.shape}")
        logger.info(
            "processed file and ending time and pretransforms: {} {} ".format(
                h5ad_location, datetime.datetime.now().strftime("%H:%M:%S")
            )
        )
        study_data.write_h5ad(output_location)

    def _load_h5ad_mmap(self, file_path: str | Path) -> AnnData:
        """
        Loads an AnnData file using memory mapping.

        Args:
        ----
            file_path (str): Path to the AnnData file.

        Returns:
        -------
            AnnData: Loaded AnnData file.
        """
        with open(file_path) as file:
            mmapped_file = mmap.mmap(file.fileno(), 0, prot=mmap.PROT_READ)
            return read_h5ad(mmapped_file)

    def _concatenate_h5ad_datasets(self) -> AnnData:
        """
        Concatenates multiple AnnData files into a single AnnData object.

        Returns
        -------
            AnnData: Concatenated AnnData files.
        """
        processed_files = [
            *filter(
                lambda f: f.name != "processed_final.h5ad",
                (self.processed_dir / self.split).glob("*"),
            )
        ]
        if self.num_workers == 0:
            ann_datas = [self._load_h5ad_mmap(h5) for h5 in processed_files]
        else:
            pool = multiprocessing.Pool()
            ann_datas = pool.map(self._load_h5ad_mmap, processed_files)
            pool.close()
            pool.join()
        concatenated = concat(ann_datas, join="outer", merge="unique")
        if self.post_transforms is not None:
            return self.post_transforms(adata=concatenated)["adata"]
        return concatenated
