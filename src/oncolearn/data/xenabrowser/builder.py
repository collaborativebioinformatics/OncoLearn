"""
Cohort builder that constructs cohorts from YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..cohort import Cohort
from ..cohort_builder import CohortBuilder as BaseCohortBuilder
from ..dataset import DataCategory
from .xena_dataset import XenaDataset


class XenaCohortBuilder(BaseCohortBuilder):
    """
    Builder class that constructs Cohort objects from YAML configuration files.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the cohort builder.
        
        Args:
            config_dir: Directory containing YAML configuration files.
                       Defaults to 'data/xenabrowser/configs' in the project root.
        """
        if config_dir is None:
            # Default to data/xenabrowser/configs in project root
            # Navigate from src/oncolearn/data/xenabrowser to project root
            project_root = Path(__file__).parent.parent.parent.parent.parent
            config_dir = project_root / "data" / "xenabrowser" / "configs"
        
        super().__init__(config_dir)
        self.config_dir = Path(config_dir)
    
    def _parse_category(self, category_str: str) -> DataCategory:
        """
        Parse a category string to DataCategory enum.
        
        Args:
            category_str: String representation of category (e.g., "mrna_seq")
            
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
        
        return category_map.get(category_str.lower(), DataCategory.CLINICAL)
    
    def _build_dataset(self, dataset_config: Dict[str, Any], cohort_code: str) -> XenaDataset:
        """
        Build a single dataset from configuration.
        
        Args:
            dataset_config: Dictionary containing dataset configuration
            cohort_code: Cohort code (e.g., "BRCA")
            
        Returns:
            Configured XenaDataset instance
        """
        category = self._parse_category(dataset_config["category"])
        
        return XenaDataset(
            name=dataset_config["name"],
            description=dataset_config["description"],
            category=category,
            url=dataset_config["url"],
            filename=dataset_config["filename"],
            default_subdir=dataset_config.get("default_subdir", f"TCGA-{cohort_code}")
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
        datasets_config = config["datasets"]
        
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
            
            def download(self, output_dir=None, download_all=True):
                if output_dir is None:
                    output_dir = f"data/xenabrowser/{cohort_info['name']}"
                
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                print(f"Downloading {cohort_info['code']} cohort to {output_path}")
                
                if download_all:
                    for dataset in self.datasets:
                        try:
                            dataset.download(str(output_path))
                        except Exception as e:
                            print(f"Error downloading {dataset.name}: {e}")
        
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

        return cohorts
