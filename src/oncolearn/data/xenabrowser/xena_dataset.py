"""
Generic dataset implementation that can be configured from YAML files.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from ..dataset import Dataset, DataCategory
from .utils import download_and_extract_gzip


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
        default_subdir: str
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
        """
        super().__init__(name, description)
        self.DATA_CATEGORY = category
        self.url = url
        self.filename = filename
        self.default_subdir = default_subdir
    
    def download(self, output_dir: Optional[str] = None) -> None:
        """Download the dataset."""
        if output_dir is None:
            output_dir = f"data/xenabrowser/{self.default_subdir}"
        
        download_and_extract_gzip(
            url=self.url,
            output_dir=output_dir,
            filename=self.filename,
            dataset_name=self.name
        )
