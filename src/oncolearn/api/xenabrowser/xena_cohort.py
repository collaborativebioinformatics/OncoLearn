"""
XenaCohort class that provides convenient data loading methods for Xena Browser cohorts.
"""

from functools import reduce
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from ..cohort import Cohort
from ..dataset import DataCategory, Dataset


class XenaCohort(Cohort):
    """
    Xena Browser cohort with convenient data loading methods.

    This class extends the base Cohort class to provide methods for loading
    and merging datasets by category (e.g., clinical(), mrna_seq(), etc.).
    """

    def __init__(self, name: str, description: str, code: str,
                 datasets: Optional[List[Dataset]] = None,
                 base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize a XenaCohort.

        Args:
            name: The name of the cohort (e.g., "TCGA-BRCA")
            description: A description of the cohort
            code: Cohort code (e.g., "BRCA")
            datasets: Optional list of Dataset objects belonging to this cohort
            base_dir: Base directory where data is stored (defaults to data/xenabrowser/{name})
        """
        super().__init__(name, description, datasets)
        self.code = code

        if base_dir is None:
            self.base_dir = Path("data/xenabrowser") / name
        else:
            self.base_dir = Path(base_dir)

    def _load_datasets_by_category(
        self,
        category: DataCategory,
        merge_on: Optional[str] = "sample",
        how: str = "outer"
    ) -> Optional[pd.DataFrame]:
        """
        Load and optionally merge all datasets for a specific category.

        Args:
            category: Data category to load
            merge_on: Column name to merge datasets on (default: 'sample'). 
                     If None, concatenates datasets vertically.
            how: Type of merge to perform ('inner', 'outer', 'left', 'right'). Default is 'outer'.

        Returns:
            Merged DataFrame or None if no datasets found
        """
        datasets = self.get_datasets_by_category(category)

        if not datasets:
            print(f"No datasets found for category: {category.value}")
            return None

        dfs = []
        for dataset in datasets:
            # Determine file path
            file_path = self.base_dir / dataset.filename

            # Handle .gz extension
            if not file_path.exists() and file_path.with_suffix('').suffix == '.tsv':
                # Try without .gz
                file_path = file_path.with_suffix('')

            if not file_path.exists():
                print(
                    f"Warning: File not found for {dataset.name}: {file_path}")
                continue

            try:
                # Load TSV file
                df = pd.read_csv(file_path, sep='\t', low_memory=False)

                # Deduplicate: first drop exact duplicates
                initial_rows = len(df)
                df = df.drop_duplicates()

                # Then deduplicate on merge key if it exists
                if merge_on and merge_on in df.columns:
                    df = df.drop_duplicates(subset=[merge_on], keep='first')
                    final_rows = len(df)
                    if initial_rows > final_rows:
                        print(
                            f"Deduplicated {dataset.name}: {initial_rows} -> {final_rows} rows")

                dfs.append(df)
            except Exception as e:
                print(f"Error loading {dataset.name} from {file_path}: {e}")
                continue

        if not dfs:
            print(f"No data loaded for category: {category.value}")
            return None

        # Merge or concatenate
        result = None
        if merge_on and len(dfs) > 1:
            # Merge datasets on specified column
            result = reduce(lambda left, right: pd.merge(
                left, right, on=merge_on, how=how), dfs)
        else:
            # Concatenate vertically
            result = pd.concat(dfs, ignore_index=True)

        return result

    def clinical(self, merge_on: Optional[str] = "sample", how: str = "outer") -> Optional[pd.DataFrame]:
        """
        Load all clinical datasets.

        Args:
            merge_on: Column to merge on (default: 'sample'). 
                     If None, concatenates datasets.
            how: Type of merge to perform ('inner', 'outer', 'left', 'right'). Default is 'outer'.

        Returns:
            DataFrame with clinical data or None if not available
        """
        return self._load_datasets_by_category(
            DataCategory.CLINICAL, merge_on, how)

    def mrna_seq(self, merge_on: Optional[str] = "sample", how: str = "outer") -> Optional[pd.DataFrame]:
        """
        Load all mRNA sequencing datasets.

        Args:
            merge_on: Column to merge on (default: 'sample'). If None, concatenates datasets.
            how: Type of merge to perform ('inner', 'outer', 'left', 'right'). Default is 'outer'.

        Returns:
            DataFrame with mRNA-seq data or None if not available
        """
        return self._load_datasets_by_category(DataCategory.MRNA_SEQ, merge_on, how)

    def protein(self, merge_on: Optional[str] = "sample", how: str = "outer") -> Optional[pd.DataFrame]:
        """
        Load all protein expression datasets.

        Args:
            merge_on: Column to merge on (default: 'sample'). If None, concatenates datasets.
            how: Type of merge to perform ('inner', 'outer', 'left', 'right'). Default is 'outer'.

        Returns:
            DataFrame with protein data or None if not available
        """
        return self._load_datasets_by_category(DataCategory.PROTEIN, merge_on, how)

    def methylation(self, merge_on: Optional[str] = "sample", how: str = "outer") -> Optional[pd.DataFrame]:
        """
        Load all DNA methylation datasets.

        Args:
            merge_on: Column to merge on (default: 'sample'). If None, concatenates datasets.
            how: Type of merge to perform ('inner', 'outer', 'left', 'right'). Default is 'outer'.

        Returns:
            DataFrame with methylation data or None if not available
        """
        return self._load_datasets_by_category(DataCategory.METHYLATION, merge_on, how)

    def cnv(self, merge_on: Optional[str] = "sample", how: str = "outer") -> Optional[pd.DataFrame]:
        """
        Load all copy number variation datasets.

        Args:
            merge_on: Column to merge on (default: 'sample'). If None, concatenates datasets.
            how: Type of merge to perform ('inner', 'outer', 'left', 'right'). Default is 'outer'.

        Returns:
            DataFrame with CNV data or None if not available
        """
        return self._load_datasets_by_category(DataCategory.CNV, merge_on, how)

    def mutation(self, merge_on: Optional[str] = "sample", how: str = "outer") -> Optional[pd.DataFrame]:
        """
        Load all somatic mutation datasets.

        Args:
            merge_on: Column to merge on (default: 'sample'). If None, concatenates datasets.
            how: Type of merge to perform ('inner', 'outer', 'left', 'right'). Default is 'outer'.

        Returns:
            DataFrame with mutation data or None if not available
        """
        return self._load_datasets_by_category(DataCategory.MUTATION, merge_on, how)

    def mirna_seq(self, merge_on: Optional[str] = "sample", how: str = "outer") -> Optional[pd.DataFrame]:
        """
        Load all microRNA sequencing datasets.

        Args:
            merge_on: Column to merge on (default: 'sample'). If None, concatenates datasets.
            how: Type of merge to perform ('inner', 'outer', 'left', 'right'). Default is 'outer'.

        Returns:
            DataFrame with miRNA-seq data or None if not available
        """
        return self._load_datasets_by_category(DataCategory.MIRNA_SEQ, merge_on, how)

    def genomics(self, merge_on: Optional[str] = "sample", how: str = "outer") -> Optional[pd.DataFrame]:
        """
        Load all general genomics datasets (e.g., ATAC-seq).

        Args:
            merge_on: Column to merge on (default: 'sample'). If None, concatenates datasets.
            how: Type of merge to perform ('inner', 'outer', 'left', 'right'). Default is 'outer'.

        Returns:
            DataFrame with genomics data or None if not available
        """
        return self._load_datasets_by_category(DataCategory.GENOMICS, merge_on, how)

    def download(self, output_dir=None, download_all=True, extract=True,
                 download_mapping=False, download_raw=False):
        """
        Download all datasets in this cohort.

        Args:
            output_dir: Directory to save downloads (defaults to self.base_dir)
            download_all: Whether to download all datasets
            extract: Whether to extract gzipped files
            download_mapping: Whether to download gene mapping files
            download_raw: Whether to download raw data files
        """
        if output_dir is None:
            output_dir = str(self.base_dir)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {self.code} cohort to {output_path}")

        if download_all:
            # Calculate size for each dataset and build file details list
            from oncolearn.utils.download import (
                confirm_cohort_download,
                get_file_size_from_url,
            )

            file_details = []
            total_size = 0

            print("Calculating total download size...")
            for dataset in self.datasets:
                size = get_file_size_from_url(dataset.url)
                if size:
                    total_size += size
                file_details.append((dataset.filename, size if size else 0))

            # Show single confirmation for entire cohort if we have size info
            if total_size > 0:
                if not confirm_cohort_download(
                    cohort_name=self.code,
                    total_size_bytes=total_size,
                    file_details=file_details,
                    verbose=True
                ):
                    print("Cohort download cancelled.")
                    return

            # Download all datasets without individual confirmations
            for dataset in self.datasets:
                try:
                    dataset.download(str(output_path), extract=extract, confirm=False,
                                     download_mapping=download_mapping,
                                     download_raw=download_raw)
                except Exception as e:
                    print(f"Error downloading {dataset.name}: {e}")

    def __repr__(self) -> str:
        """String representation of the cohort."""
        return (f"XenaCohort(code='{self.code}', name='{self.name}', "
                f"datasets={len(self.datasets)}, base_dir='{self.base_dir}')")
