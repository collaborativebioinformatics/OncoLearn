#!/usr/bin/env python3
"""
Unified Download Script for OncoLearn

Downloads cancer data from various sources (UCSC Xena Browser, TCIA, etc.)
"""

import argparse
import sys
from pathlib import Path

# Add scripts/data to path to import utility modules
sys.path.insert(0, str(Path(__file__).parent / "data"))

from download_xena import download_cohort as download_xena_cohort
from download_xena import download_cohorts as download_xena_cohorts
from download_xena import download_all as download_xena_all
from download_xena import get_available_cohorts as get_xena_cohorts
from download_tcia import download_cohort as download_tcia_cohort
from download_tcia import download_cohorts as download_tcia_cohorts
from download_tcia import download_all as download_tcia_all
from download_tcia import get_available_cohorts as get_tcia_cohorts

# Add src to path for DataCategory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from oncolearn.data.dataset import DataCategory


def download_xena(
    cohorts: list[str],
    output_dir: str = None,
    category: str = None,
    download_all_flag: bool = False
) -> dict[str, bool]:
    """
    Download cohorts from Xena Browser.
    
    Args:
        cohorts: List of cohort codes
        output_dir: Optional output directory
        category: Optional category filter string
        download_all_flag: Download all available cohorts
        
    Returns:
        Dictionary mapping cohort codes to success status
    """
    # Parse category if specified
    category_enum = None
    if category:
        category_enum = parse_category(category)
    
    # Download all or specific cohorts
    if download_all_flag:
        return download_xena_all(output_dir, category_enum, verbose=True)
    else:
        return download_xena_cohorts(cohorts, output_dir, category_enum, verbose=True)


def download_tcia(
    cohorts: list[str],
    output_dir: str = None,
    download_all_flag: bool = False,
    download_images: bool = False
) -> dict[str, bool]:
    """
    Download TCIA manifests for cohorts.
    
    Args:
        cohorts: List of cohort codes
        output_dir: Optional output directory
        download_all_flag: Download all available cohorts
        download_images: If True, run nbia-data-retriever to download images
        
    Returns:
        Dictionary mapping cohort codes to success status
    """
    if download_all_flag:
        return download_tcia_all(output_dir, download_images=download_images, verbose=True)
    else:
        return download_tcia_cohorts(cohorts, output_dir, download_images=download_images, verbose=True)



def parse_category(category_str: str) -> DataCategory:
    """Parse category string to DataCategory enum."""
    category_map = {
        "image": DataCategory.IMAGE,
        "clinical": DataCategory.CLINICAL,
        "mrna_seq": DataCategory.MRNA_SEQ,
        "mrna": DataCategory.MRNA_SEQ,
        "dna_seq": DataCategory.DNA_SEQ,
        "dna": DataCategory.DNA_SEQ,
        "mirna_seq": DataCategory.MIRNA_SEQ,
        "mirna": DataCategory.MIRNA_SEQ,
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
    
    cat_lower = category_str.lower()
    if cat_lower not in category_map:
        raise ValueError(f"Unknown category: {category_str}. Available: {', '.join(category_map.keys())}")
    
    return category_map[cat_lower]


def list_cohorts(source: str) -> None:
    """List available cohorts for a source."""
    if source == "xena":
        cohorts = get_xena_cohorts()
        print("Available Xena Browser Cohorts:")
    elif source == "tcia":
        cohorts = get_tcia_cohorts()
        print("Available TCIA Cohorts:")
    else:
        print(f"Unknown source: {source}")
        return
    
    print("=" * 80)
    for cohort in sorted(cohorts):
        print(f"  {cohort}")
    print("=" * 80)
    print(f"Total: {len(cohorts)} cohorts")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download cancer data from various sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download BRCA from Xena Browser
  uv run download --xena --cohorts BRCA
  
  # Download only mutation data
  uv run download --xena --cohorts BRCA --category mutation
  
  # Download multiple cohorts
  uv run download --xena --cohorts BRCA,LUAD,ACC
  
  # Download TCIA manifest only
  uv run download --tcia --cohorts BRCA
  
  # Download TCIA manifest and images
  uv run download --tcia --cohorts BRCA --download-images
  
  # Download all Xena cohorts
  uv run download --xena --all
  
  # List available cohorts
  uv run download --xena --list
  uv run download --tcia --list
        """
    )
    
    # Source selection (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--xena", action="store_true", help="Download from UCSC Xena Browser")
    source_group.add_argument("--tcia", action="store_true", help="Download TCIA imaging manifests")
    
    # Action selection (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--cohorts", type=str, help="Cohort code(s), comma-separated (e.g., BRCA,LUAD)")
    action_group.add_argument("--all", action="store_true", help="Download all available cohorts")
    action_group.add_argument("--list", action="store_true", help="List available cohorts and exit")
    
    # Optional arguments
    parser.add_argument("--category", type=str, help="Filter datasets by category (Xena only)")
    parser.add_argument("--output", type=str, help="Custom output directory")
    parser.add_argument("--download-images", action="store_true", help="Download actual images using nbia-data-retriever (TCIA only)")
    
    args = parser.parse_args()
    
    # Determine source
    source = "xena" if args.xena else "tcia"
    
    # Handle list action
    if args.list:
        list_cohorts(source)
        return
    
    # Category filtering only works with Xena
    if args.category and args.tcia:
        print("ERROR: --category can only be used with --xena")
        sys.exit(1)
    
    # Download images only works with TCIA
    if hasattr(args, 'download_images') and args.download_images and args.xena:
        print("ERROR: --download-images can only be used with --tcia")
        sys.exit(1)
    
    # Parse cohorts
    if args.all:
        if source == "xena":
            cohort_list = get_xena_cohorts()
        else:  # tcia
            cohort_list = get_tcia_cohorts()
    else:
        cohort_list = [c.strip().upper() for c in args.cohorts.split(',')]
    
    # Download cohorts
    if source == "xena":
        results = download_xena(cohort_list, args.output, args.category, args.all)
    else:  # tcia
        download_images = hasattr(args, 'download_images') and args.download_images
        results = download_tcia(cohort_list, args.output, args.all, download_images)
    
    # Summary
    successful = sum(results.values())
    total = len(results)
    failed = total - successful
    
    print()
    print("=" * 80)
    print(f"Summary: {successful}/{total} cohorts downloaded successfully")
    if failed > 0:
        print(f"Failed: {failed}")
        print("Failed cohorts:", ", ".join([k for k, v in results.items() if not v]))
    print("=" * 80)
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
