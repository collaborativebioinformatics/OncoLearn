"""
Cohort builder that constructs cohorts from YAML configuration files.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from oncolearn.utils.download import confirm_cohort_download

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
            category_str: String representation of category (e.g., "mrna_seq", "ATAC-seq", "gene expression RNAseq")
            
        Returns:
            DataCategory enum value
        """
        category_map = {
            "image": DataCategory.IMAGE,
            "clinical": DataCategory.CLINICAL,
            "phenotype": DataCategory.CLINICAL,
            "mrna_seq": DataCategory.MRNA_SEQ,
            "mrna": DataCategory.MRNA_SEQ,
            "gene expression rnaseq": DataCategory.MRNA_SEQ,
            "gene_expression_rnaseq": DataCategory.MRNA_SEQ,
            "dna_seq": DataCategory.DNA_SEQ,
            "dna": DataCategory.DNA_SEQ,
            "mirna_seq": DataCategory.MIRNA_SEQ,
            "mirna": DataCategory.MIRNA_SEQ,
            "stem loop expression": DataCategory.MIRNA_SEQ,
            "stem_loop_expression": DataCategory.MIRNA_SEQ,
            "protein": DataCategory.PROTEIN,
            "protein expression": DataCategory.PROTEIN,
            "protein_expression": DataCategory.PROTEIN,
            "methylation": DataCategory.METHYLATION,
            "dna methylation": DataCategory.METHYLATION,
            "dna_methylation": DataCategory.METHYLATION,
            "cnv": DataCategory.CNV,
            "copy number": DataCategory.CNV,
            "copy_number": DataCategory.CNV,
            "copy number (gene-level)": DataCategory.CNV,
            "copy_number_gene_level": DataCategory.CNV,
            "mutation": DataCategory.MUTATION,
            "somatic mutation": DataCategory.MUTATION,
            "somatic_mutation": DataCategory.MUTATION,
            "somatic mutation (snps and small indels)": DataCategory.MUTATION,
            "snp": DataCategory.SNP,
            "atac-seq": DataCategory.GENOMICS,
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
        
        Supports both old format (name, url, category, filename) and 
        new format (dataset_id, download, data_type, gene_mapping, raw_data).
        
        Args:
            dataset_config: Dictionary containing dataset configuration
            cohort_code: Cohort code (e.g., "BRCA")
            
        Returns:
            Configured XenaDataset instance
        """
        # Check if this is the new format (has dataset_id and download)
        if "dataset_id" in dataset_config and "download" in dataset_config:
            # New format
            dataset_id = dataset_config["dataset_id"]
            url = dataset_config["download"]
            
            # Extract filename from download URL or dataset_id
            if url.endswith('.gz'):
                filename = url.split('/')[-1].replace('%2F', '_')
            else:
                # Use dataset_id as filename base
                filename = dataset_id.split('/')[-1]
                if not filename.endswith('.tsv'):
                    filename = f"{filename}.tsv"
            
            # Parse category from data_type field
            data_type = dataset_config.get("data_type", "clinical")
            category = self._parse_category(data_type)
            
            # Use dataset_id as name
            name = dataset_id
            
            # Use wrangling or data_type as description
            description = dataset_config.get("wrangling", data_type)
            if isinstance(description, str) and len(description) > 200:
                description = description[:200] + "..."
            
            return XenaDataset(
                name=name,
                description=description,
                category=category,
                url=url,
                filename=filename,
                default_subdir=f"TCGA-{cohort_code}",
                gene_mapping_url=dataset_config.get("gene_mapping"),
                raw_data_url=dataset_config.get("raw_data")
            )
        else:
            # Old format (backwards compatibility)
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
            
            def download(self, output_dir=None, download_all=True, extract=True, 
                       download_mapping=False, download_raw=False):
                if output_dir is None:
                    output_dir = f"data/xenabrowser/{cohort_info['name']}"
                
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                print(f"Downloading {cohort_info['code']} cohort to {output_path}")
                
                if download_all:
                    # Calculate size for each dataset and build file details list
                    from oncolearn.utils.download import get_file_size_from_url
                    
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
                            cohort_name=cohort_info['code'],
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
