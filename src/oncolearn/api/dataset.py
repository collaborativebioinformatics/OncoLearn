"""
Dataset module for OncoLearn.

This module defines the base Dataset class that represents individual datasets
with download capabilities and metadata.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional


class DataCategory(Enum):
    """
    Enumeration of valid data categories for cancer ML tasks.
    """
    IMAGE = "image"  # Medical imaging data (CT, MRI, histopathology, etc.)
    CLINICAL = "clinical"  # Clinical and demographic data
    MRNA_SEQ = "mrna_seq"  # mRNA sequencing data
    DNA_SEQ = "dna_seq"  # DNA sequencing data
    MIRNA_SEQ = "mirna_seq"  # microRNA sequencing data
    PROTEIN = "protein"  # Protein expression data
    METHYLATION = "methylation"  # DNA methylation data
    CNV = "cnv"  # Copy number variation data
    MUTATION = "mutation"  # Somatic mutation data
    SNP = "snp"  # Single nucleotide polymorphism data
    TRANSCRIPTOME = "transcriptome"  # Transcriptome data
    METABOLOMICS = "metabolomics"  # Metabolomics data
    PROTEOMICS = "proteomics"  # Proteomics data
    GENOMICS = "genomics"  # General genomics data
    MANIFEST = "manifest"  # Data manifest files (e.g., TCIA .tcia files)
    MULTIMODAL = "multimodal"  # Combined multiple data types


class Dataset(ABC):
    """
    Abstract base class for datasets.
    
    Each dataset represents a single data source with a name, description,
    and data category. Subclasses must implement the download method.
    """
    
    # Static class variable for data category - must be a DataCategory enum value
    DATA_CATEGORY: DataCategory = None
    
    def __init__(self, name: str, description: str):
        """
        Initialize a Dataset.
        
        Args:
            name: The name of the dataset
            description: A description of the dataset
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def download(self, output_dir: Optional[str] = None) -> None:
        """
        Abstract method to download the dataset.
        
        Args:
            output_dir: Optional directory to save the downloaded data.
                       If None, uses a default location.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the dataset."""
        return f"Dataset(name='{self.name}', category='{self.DATA_CATEGORY}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the dataset."""
        return (f"Dataset(name='{self.name}', description='{self.description}', "
                f"category='{self.DATA_CATEGORY}')")
