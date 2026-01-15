"""
UCSC Xena Browser Data Downloader Utilities

Provides utility functions to download TCGA genomics data from UCSC Xena Browser
using the XenaCohortBuilder.
"""

from typing import Dict

from oncolearn.api.dataset import DataCategory
from oncolearn.api.xenabrowser import XenaCohortBuilder


def get_available_cohorts() -> list[str]:
    """
    Get list of all available cohorts from Xena Browser.
    
    Returns:
        List of cohort codes
    """
    builder = XenaCohortBuilder()
    return builder.list_available_cohorts()


def download_cohort(
    cohort_code: str,
    output_dir: str = None,
    category: DataCategory = None,
    unzip: bool = True,
    download_mapping: bool = False,
    download_raw: bool = False,
    dataset_ids: list[str] = None,
    verbose: bool = True
) -> bool:
    """
    Download a single cohort from Xena Browser.
    
    Args:
        cohort_code: Cohort code (e.g., 'BRCA')
        output_dir: Optional output directory
        category: Optional category filter (e.g., DataCategory.MUTATION)
        unzip: Whether to extract gzipped files after download
        download_mapping: Whether to download gene mapping files
        download_raw: Whether to download raw data files
        verbose: Print progress messages
        
    Returns:
        True if successful, False otherwise
    """
    builder = XenaCohortBuilder()
    
    try:
        cohort = builder.build_cohort(cohort_code)
        
        if verbose:
            if category:
                print(f"Downloading {cohort_code} ({category.value})...")
            elif dataset_ids:
                print(f"Downloading {cohort_code} (specific datasets)...")
            else:
                print(f"Downloading {cohort_code}...")
        
        # Filter by dataset IDs if specified
        if dataset_ids:
            datasets = [d for d in cohort.datasets if d.name in dataset_ids]
            
            if not datasets:
                if verbose:
                    print(f"⚠ {cohort_code}: No datasets found with IDs: {', '.join(dataset_ids)}")
                return False
            
            # Download filtered datasets
            for dataset in datasets:
                try:
                    dataset.download(output_dir, extract=unzip, confirm=False,
                                   download_mapping=download_mapping,
                                   download_raw=download_raw)
                except Exception as e:
                    if verbose:
                        print(f"[ERROR] {dataset.name}: {e}")
        # Filter by category if specified
        elif category:
            datasets = cohort.get_datasets_by_category(category)
            
            if not datasets:
                if verbose:
                    print(f"⚠ {cohort_code}: No datasets found for category '{category.value}'")
                return False
            
            # Download filtered datasets
            for dataset in datasets:
                try:
                    dataset.download(output_dir, extract=unzip, 
                                   download_mapping=download_mapping,
                                   download_raw=download_raw)
                except Exception as e:
                    if verbose:
                        print(f"[ERROR] {dataset.name}: {e}")
        else:
            cohort.download(output_dir=output_dir, download_all=True, extract=unzip,
                          download_mapping=download_mapping, download_raw=download_raw)
        
        if verbose:
            print(f"Finished downloading {cohort_code}!")
        return True
        
    except FileNotFoundError as e:
        if verbose:
            print(f"[ERROR] {cohort_code}: Configuration not found")
            print(f"Reason: {e}")
        return False
    except Exception as e:
        if verbose:
            print(f"[ERROR] {cohort_code}: Failed")
            print(f"Reason: {e}")
        return False


def download_cohorts(
    cohorts: list[str],
    output_dir: str = None,
    category: DataCategory = None,
    unzip: bool = True,
    download_mapping: bool = False,
    download_raw: bool = False,
    dataset_ids: list[str] = None,
    verbose: bool = True
) -> Dict[str, bool]:
    """
    Download multiple cohorts from Xena Browser.
    
    Args:
        cohorts: List of cohort codes
        output_dir: Optional base output directory
        category: Optional category filter
        unzip: Whether to extract gzipped files after download
        download_mapping: Whether to download gene mapping files
        download_raw: Whether to download raw data files
        verbose: Print progress messages
        
    Returns:
        Dictionary mapping cohort codes to success status
    """
    results = {}
    
    for cohort_code in cohorts:
        cohort_upper = cohort_code.upper()
        success = download_cohort(cohort_upper, output_dir, category, unzip,
                                 download_mapping, download_raw, dataset_ids, verbose)
        results[cohort_upper] = success
    
    return results


def download_all(
    output_dir: str = None,
    category: DataCategory = None,
    unzip: bool = True,
    download_mapping: bool = False,
    download_raw: bool = False,
    verbose: bool = True
) -> Dict[str, bool]:
    """
    Download all available cohorts from Xena Browser.
    
    Args:
        output_dir: Optional base output directory
        category: Optional category filter
        unzip: Whether to extract gzipped files after download
        download_mapping: Whether to download gene mapping files
        download_raw: Whether to download raw data files
        verbose: Print progress messages
        
    Returns:
        Dictionary mapping cohort codes to success status
    """
    available_cohorts = get_available_cohorts()
    
    if verbose:
        print(f"Downloading all {len(available_cohorts)} cohorts...")
        print()
    
    return download_cohorts(available_cohorts, output_dir, category, unzip,
                          download_mapping, download_raw, verbose)
