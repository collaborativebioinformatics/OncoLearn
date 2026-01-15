"""
Generic dataset implementation that can be configured from YAML files.
"""

from typing import Optional

from oncolearn.utils.download import download_and_extract_gzip

from ..dataset import DataCategory, Dataset


class XenaDataset(Dataset):
    """
    A generic Xena Browser dataset that can be configured with metadata.
    
    This class is instantiated from YAML configuration files rather than
    being subclassed for each specific dataset.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        category: DataCategory,
        url: str,
        filename: str,
        default_subdir: str,
        gene_mapping_url: Optional[str] = None,
        raw_data_url: Optional[str] = None
    ):
        """
        Initialize a Xena dataset.
        
        Args:
            name: Dataset name
            description: Dataset description
            category: Data category (from DataCategory enum)
            url: Download URL
            filename: Filename to save as
            default_subdir: Default subdirectory within cohort folder
            gene_mapping_url: Optional gene mapping download URL
            raw_data_url: Optional raw data download URL
        """
        super().__init__(name, description)
        self.DATA_CATEGORY = category
        self.url = url
        self.filename = filename
        self.default_subdir = default_subdir
        self.gene_mapping_url = gene_mapping_url
        self.raw_data_url = raw_data_url
    
    def download(self, output_dir: Optional[str] = None, extract: bool = True, 
                confirm: bool = True, download_mapping: bool = False, 
                download_raw: bool = False) -> None:
        """Download the dataset.
        
        Args:
            output_dir: Optional directory to save the downloaded data
            extract: Whether to extract gzipped files after download
            confirm: Whether to ask for confirmation before downloading
            download_mapping: Whether to download gene mapping file
            download_raw: Whether to download raw data file
        """
        if output_dir is None:
            output_dir = f"data/xenabrowser/{self.default_subdir}"
        
        # Download main data file
        download_and_extract_gzip(
            url=self.url,
            output_dir=output_dir,
            filename=self.filename,
            dataset_name=self.name,
            extract=extract,
            verbose=True,
            confirm=confirm
        )
        
        # Download gene mapping if requested and available
        if download_mapping and self.gene_mapping_url:
            mapping_filename = self.gene_mapping_url.split('/')[-1].replace('%2F', '_')
            download_and_extract_gzip(
                url=self.gene_mapping_url,
                output_dir=output_dir,
                filename=mapping_filename,
                dataset_name=f"{self.name} (mapping)",
                extract=extract,
                verbose=True,
                confirm=False
            )
        
        # Download raw data if requested and available
        if download_raw and self.raw_data_url:
            # Note: raw_data_url might be a documentation link, not a download
            # Only download if it looks like a downloadable file
            if not self.raw_data_url.startswith('https://docs.gdc.cancer.gov') and \
               not self.raw_data_url.startswith('https://gdc.cancer.gov/about-data'):
                raw_filename = self.raw_data_url.split('/')[-1].replace('%2F', '_')
                download_and_extract_gzip(
                    url=self.raw_data_url,
                    output_dir=output_dir,
                    filename=raw_filename,
                    dataset_name=f"{self.name} (raw)",
                    extract=extract,
                    verbose=True,
                    confirm=False
                )
