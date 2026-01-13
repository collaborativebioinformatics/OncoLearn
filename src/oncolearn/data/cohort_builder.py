"""
Abstract base class for cohort builders.

This module defines the interface that all cohort builders should implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional

from .cohort import Cohort


class CohortBuilder(ABC):
    """
    Abstract base class for building cohorts from configuration sources.
    
    Cohort builders are responsible for constructing Cohort objects from
    various sources (YAML files, databases, APIs, etc.). Each implementation
    should provide methods to build individual cohorts, list available cohorts,
    and optionally build all cohorts at once.
    """
    
    def __init__(self, config_source: Optional[Path] = None):
        """
        Initialize the cohort builder.
        
        Args:
            config_source: Source of configuration data (directory, file, URL, etc.)
        """
        self.config_source = config_source
    
    @abstractmethod
    def build_cohort(self, cohort_id: str) -> Cohort:
        """
        Build a single cohort by its identifier.
        
        Args:
            cohort_id: Unique identifier for the cohort (e.g., "BRCA", "LUAD")
            
        Returns:
            Configured Cohort instance
            
        Raises:
            ValueError: If cohort_id is not found or invalid
        """
        pass
    
    @abstractmethod
    def list_available_cohorts(self) -> List[str]:
        """
        List all available cohort identifiers.
        
        Returns:
            List of cohort identifiers
        """
        pass
    
    def build_all_cohorts(self) -> Dict[str, Cohort]:
        """
        Build all available cohorts.
        
        Returns:
            Dictionary mapping cohort identifiers to Cohort instances
        """
        cohorts = {}
        available = self.list_available_cohorts()
        
        for cohort_id in available:
            try:
                cohorts[cohort_id] = self.build_cohort(cohort_id)
            except Exception as e:
                print(f"Warning: Failed to build cohort {cohort_id}: {e}")
        
        return cohorts
