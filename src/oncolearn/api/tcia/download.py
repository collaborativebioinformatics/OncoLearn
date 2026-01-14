"""
TCIA (The Cancer Imaging Archive) Data Downloader Utilities

Provides utility functions to download TCIA imaging data manifest files for TCGA cohorts
using the TCIACohortBuilder.
"""

from typing import Dict

from oncolearn.api.tcia import TCIACohortBuilder


def get_available_cohorts() -> list[str]:
    """
    Get list of all available cohorts from TCIA.
    
    Returns:
        List of cohort codes
    """
    builder = TCIACohortBuilder()
    return builder.list_available_cohorts()


def download_cohort(
    cohort_code: str,
    output_dir: str = None,
    download_images: bool = False,
    unzip: bool = True,
    verbose: bool = True
) -> bool:
    """
    Download a single cohort from TCIA.
    
    Args:
        cohort_code: Cohort code (e.g., 'BRCA')
        output_dir: Optional output directory
        download_images: If True, run nbia-data-retriever to download images
        unzip: Whether to extract gzipped files after download
        verbose: Print progress messages
        
    Returns:
        True if successful, False otherwise
    """
    builder = TCIACohortBuilder()
    
    try:
        cohort = builder.build_cohort(cohort_code)
        
        if verbose:
            print(f"Downloading {cohort_code}...")
        
        # Check if cohort has any datasets
        if not cohort.datasets:
            if verbose:
                print(f"  ⚠ {cohort_code}: No manifests available")
            return False
        
        # Download all datasets for this cohort
        cohort.download(
            output_dir=output_dir,
            download_all=True,
            download_images=download_images,
            extract=unzip,
            verbose=verbose
        )
        
        if verbose:
            print(f"  [OK] {cohort_code}: Complete")
        return True
        
    except FileNotFoundError as e:
        if verbose:
            print(f"  [ERROR] {cohort_code}: Configuration not found")
            print(f"    Reason: {e}")
        return False
    except Exception as e:
        if verbose:
            print(f"  [ERROR] {cohort_code}: Failed")
            print(f"    Reason: {e}")
        return False


def download_cohorts(
    cohorts: list[str],
    output_dir: str = None,
    download_images: bool = False,
    unzip: bool = True,
    verbose: bool = True
) -> Dict[str, bool]:
    """
    Download multiple cohorts from TCIA.
    
    Args:
        cohorts: List of cohort codes
        output_dir: Optional base output directory
        download_images: If True, run nbia-data-retriever to download images
        unzip: Whether to extract gzipped files after download
        verbose: Print progress messages
        
    Returns:
        Dictionary mapping cohort codes to success status
    """
    results = {}
    
    for cohort_code in cohorts:
        cohort_upper = cohort_code.upper()
        success = download_cohort(cohort_upper, output_dir, download_images, unzip, verbose)
        results[cohort_upper] = success
    
    return results


def download_all(
    output_dir: str = None,
    download_images: bool = False,
    unzip: bool = True,
    verbose: bool = True
) -> Dict[str, bool]:
    """
    Download all available TCIA cohorts.
    
    Args:
        output_dir: Optional base output directory
        download_images: If True, run nbia-data-retriever to download images
        unzip: Whether to extract gzipped files after download
        verbose: Print progress messages
        
    Returns:
        Dictionary mapping cohort codes to success status
    """
    available_cohorts = get_available_cohorts()
    
    if verbose:
        print(f"Downloading all {len(available_cohorts)} cohorts...")
        print()
    
    return download_cohorts(available_cohorts, output_dir, download_images, unzip, verbose)
