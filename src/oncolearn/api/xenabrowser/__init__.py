"""
UCSC Xena Browser data module.

This module provides cohort and dataset classes for downloading TCGA data
from the UCSC Xena Browser. Cohorts are configured using YAML files and 
built dynamically using the XenaCohortBuilder.

Usage:
    from oncolearn.api.xenabrowser import XenaCohortBuilder
    
    builder = XenaCohortBuilder()
    brca_cohort = builder.build_cohort("BRCA")
    brca_cohort.download()
    
    # List available cohorts
    cohorts = builder.list_available_cohorts()
"""

from .builder import XenaCohortBuilder
from .xena_dataset import XenaDataset

__all__ = [
    'XenaCohortBuilder',
    'XenaDataset',
]
