"""
Cohort module for OncoLearn.

This module defines the Cohort class that represents a collection of related
datasets with download and management capabilities.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from .dataset import Dataset


class Cohort(ABC):
    """
    Abstract base class for cohorts.
    
    A cohort represents a collection of related datasets (e.g., all datasets
    for a specific cancer type). Each cohort has a name, description, and
    contains multiple datasets. Subclasses must implement the download method.
    """
    
    def __init__(self, name: str, description: str, datasets: Optional[List[Dataset]] = None):
        """
        Initialize a Cohort.
        
        Args:
            name: The name of the cohort
            description: A description of the cohort
            datasets: Optional list of Dataset objects belonging to this cohort.
                     If None, initializes with an empty list.
        """
        self.name = name
        self.description = description
        self.datasets = datasets if datasets is not None else []
    
    def add_dataset(self, dataset: Dataset) -> None:
        """
        Add a dataset to this cohort.
        
        Args:
            dataset: The Dataset object to add
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected Dataset object, got {type(dataset)}")
        self.datasets.append(dataset)
    
    def remove_dataset(self, dataset_name: str) -> bool:
        """
        Remove a dataset from this cohort by name.
        
        Args:
            dataset_name: The name of the dataset to remove
            
        Returns:
            True if the dataset was found and removed, False otherwise
        """
        for i, dataset in enumerate(self.datasets):
            if dataset.name == dataset_name:
                self.datasets.pop(i)
                return True
        return False
    
    def list_datasets(self) -> List[str]:
        """
        List all dataset names in this cohort.
        
        Returns:
            A list of dataset names
        """
        return [dataset.name for dataset in self.datasets]
    
    def get_dataset(self, dataset_name: str) -> Optional[Dataset]:
        """
        Get a specific dataset by name.
        
        Args:
            dataset_name: The name of the dataset to retrieve
            
        Returns:
            The Dataset object if found, None otherwise
        """
        for dataset in self.datasets:
            if dataset.name == dataset_name:
                return dataset
        return None
    
    def get_datasets_by_category(self, category: str) -> List[Dataset]:
        """
        Get all datasets in a specific category.
        
        Args:
            category: The data category to filter by
            
        Returns:
            A list of Dataset objects matching the category
        """
        return [dataset for dataset in self.datasets 
                if dataset.DATA_CATEGORY == category]
    
    @abstractmethod
    def download(self, output_dir: Optional[str] = None, 
                 download_all: bool = True) -> None:
        """
        Abstract method to download the cohort data.
        
        Args:
            output_dir: Optional directory to save the downloaded data.
                       If None, uses a default location.
            download_all: If True, downloads all datasets in the cohort.
                         If False, may download only cohort-level metadata.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the cohort."""
        return f"Cohort(name='{self.name}', datasets={len(self.datasets)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the cohort."""
        dataset_names = ', '.join([f"'{d.name}'" for d in self.datasets])
        return (f"Cohort(name='{self.name}', description='{self.description}', "
                f"datasets=[{dataset_names}])")
    
    def __len__(self) -> int:
        """Return the number of datasets in this cohort."""
        return len(self.datasets)
