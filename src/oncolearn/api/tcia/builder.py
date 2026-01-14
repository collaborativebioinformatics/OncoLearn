"""
Cohort builder that constructs TCIA cohorts from YAML configuration files.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from oncolearn.utils.download import confirm_cohort_download

from ..cohort import Cohort
from ..cohort_builder import CohortBuilder as BaseCohortBuilder
from ..dataset import DataCategory
from .tcia_dataset import TCIADataset


class TCIACohortBuilder(BaseCohortBuilder):
    """
    Builder class that constructs TCIA Cohort objects from YAML configuration files.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the TCIA cohort builder.
        
        Args:
            config_dir: Directory containing YAML configuration files.
                       Defaults to 'data/tcia/configs' in the project root.
        """
        if config_dir is None:
            # Default to data/tcia/configs in project root
            # Navigate from src/oncolearn/data/tcia to project root
            project_root = Path(__file__).parent.parent.parent.parent.parent
            config_dir = project_root / "data" / "tcia" / "configs"
        
        super().__init__(config_dir)
        self.config_dir = Path(config_dir)
    
    def _parse_category(self, category_str: str) -> DataCategory:
        """
        Parse a category string to DataCategory enum.
        
        Args:
            category_str: String representation of category (e.g., "manifest")
            
        Returns:
            DataCategory enum value
        """
        category_map = {
            "image": DataCategory.IMAGE,
            "clinical": DataCategory.CLINICAL,
            "mrna_seq": DataCategory.MRNA_SEQ,
            "dna_seq": DataCategory.DNA_SEQ,
            "mirna_seq": DataCategory.MIRNA_SEQ,
            "protein": DataCategory.PROTEIN,
            "methylation": DataCategory.METHYLATION,
            "cnv": DataCategory.CNV,
            "mutation": DataCategory.MUTATION,
            "snp": DataCategory.SNP,
            "transcriptome": DataCategory.TRANSCRIPTOME,
            "metabolomics": DataCategory.METABOLOMICS,
            "proteomics": DataCategory.PROTEOMICS,
            "genomics": DataCategory.GENOMICS,
            "manifest": DataCategory.MANIFEST,
            "multimodal": DataCategory.MULTIMODAL,
        }
        
        return category_map.get(category_str.lower(), DataCategory.MANIFEST)
    
    def _build_dataset(self, dataset_config: Dict[str, Any], cohort_code: str) -> TCIADataset:
        """
        Build a single TCIA dataset from configuration.
        
        Args:
            dataset_config: Dictionary containing dataset configuration
            cohort_code: Cohort code (e.g., "BRCA")
            
        Returns:
            Configured TCIADataset instance
        """
        # Parse category if specified, otherwise let TCIADataset determine it
        category = None
        if "category" in dataset_config:
            category = self._parse_category(dataset_config["category"])
        
        return TCIADataset(
            name=dataset_config["name"],
            description=dataset_config["description"],
            url=dataset_config["url"],
            filename=dataset_config["filename"],
            default_subdir=dataset_config.get("default_subdir", f"TCGA-{cohort_code}"),
            file_type=dataset_config.get("file_type", "manifest"),
            category=category
        )
    
    def build_from_file(self, yaml_file: Path) -> Cohort:
        """
        Build a cohort from a YAML configuration file.
        
        Args:
            yaml_file: Path to YAML configuration file
            
        Returns:
            Configured Cohort instance
        """
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        
        cohort_info = config["cohort"]
        datasets_config = config.get("datasets", [])
        
        # Build all datasets
        datasets = []
        for dataset_config in datasets_config:
            dataset = self._build_dataset(dataset_config, cohort_info["code"])
            datasets.append(dataset)
        
        # Create a dynamic cohort class
        class ConfiguredCohort(Cohort):
            def __init__(self):
                super().__init__(
                    name=cohort_info["name"],
                    description=cohort_info["description"],
                    datasets=datasets
                )
            
            def download(self, output_dir=None, download_all=True, download_images=False, extract=True, verbose=True):
                """
                Download all datasets in the cohort.
                
                Args:
                    output_dir: Base output directory
                    download_all: If True, download all datasets
                    download_images: If True, run nbia-data-retriever for .tcia files
                    extract: Whether to extract gzipped files after download
                    verbose: Print progress messages
                """
                if output_dir is None:
                    output_dir = "data/tcia/manifests"
                
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                if verbose:
                    print(f"Downloading {cohort_info['code']} cohort to {output_path}")
                
                if download_all:
                    # Calculate size for each dataset and build file details list
                    from oncolearn.utils.download import get_file_size_from_url
                    
                    file_details = []
                    total_size = 0
                    
                    if verbose:
                        print("Calculating total download size...")
                    
                    for dataset in self.datasets:
                        size = get_file_size_from_url(dataset.url)
                        if size:
                            total_size += size
                        file_details.append((dataset.filename, size if size else 0))
                    
                    # Show single confirmation for entire cohort if we have size info
                    if total_size > 0:
                        if not confirm_cohort_download(
                            cohort_name=cohort_info['code'],
                            total_size_bytes=total_size,
                            file_details=file_details,
                            verbose=verbose
                        ):
                            if verbose:
                                print("Cohort download cancelled.")
                            return
                    
                    # Download all datasets without individual confirmations
                    for dataset in self.datasets:
                        try:
                            dataset.download(str(output_path), download_images=download_images, extract=extract, verbose=verbose, confirm=False)
                        except Exception as e:
                            if verbose:
                                print(f"    [ERROR] Error downloading {dataset.name}: {e}")
        
        return ConfiguredCohort()
    
    def build_cohort(self, cohort_code: str) -> Cohort:
        """
        Build a cohort by code (e.g., "BRCA").
        
        Args:
            cohort_code: Cohort code
            
        Returns:
            Configured Cohort instance
        """
        yaml_file = self.config_dir / f"{cohort_code.lower()}.yaml"
        
        if not yaml_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_file}")
        
        return self.build_from_file(yaml_file)
    
    def list_available_cohorts(self) -> List[str]:
        """
        List all available cohort codes.
        
        Returns:
            List of cohort codes
        """
        if not self.config_dir.exists():
            return []
        
        return [f.stem.upper() for f in self.config_dir.glob("*.yaml")]
