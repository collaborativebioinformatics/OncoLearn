#!/usr/bin/env python3
"""
Unified Download Script for OncoLearn

Downloads cancer data from various sources (UCSC Xena Browser, TCIA, etc.)
"""

import argparse
import sys

from oncolearn.api.cbioportal.download import (
    download_all as download_cbioportal_all,
)
from oncolearn.api.cbioportal.download import (
    download_cohorts as download_cbioportal_cohorts,
)
from oncolearn.api.cbioportal.download import (
    get_available_cohorts as get_cbioportal_cohorts,
)
from oncolearn.api.cbioportal.download import list_studies as cbioportal_list_studies
from oncolearn.api.dataset import DataCategory
from oncolearn.api.tcia.download import (
    download_all as download_tcia_all,
)
from oncolearn.api.tcia.download import (
    download_cohorts as download_tcia_cohorts,
)
from oncolearn.api.tcia.download import (
    get_available_cohorts as get_tcia_cohorts,
)
from oncolearn.api.xenabrowser.download import (
    download_all as download_xena_all,
)
from oncolearn.api.xenabrowser.download import (
    download_cohorts as download_xena_cohorts,
)
from oncolearn.api.xenabrowser.download import (
    get_available_cohorts as get_xena_cohorts,
)


def download_xena(
    cohorts: list[str],
    output_dir: str = None,
    category: str = None,
    download_all_flag: bool = False,
    unzip: bool = True,
    download_mapping: bool = False,
    download_raw: bool = False,
    dataset_ids: list[str] = None
) -> dict[str, bool]:
    """
    Download cohorts from Xena Browser.

    Args:
        cohorts: List of cohort codes
        output_dir: Optional output directory
        category: Optional category filter string
        download_all_flag: Download all available cohorts
        unzip: Whether to extract gzipped files after download
        download_mapping: Whether to download gene mapping files
        download_raw: Whether to download raw data files

    Returns:
        Dictionary mapping cohort codes to success status
    """
    # Parse category if specified
    category_enum = None
    if category:
        category_enum = parse_category(category)

    # Download all or specific cohorts
    if download_all_flag:
        return download_xena_all(output_dir, category_enum, unzip=unzip,
                                 download_mapping=download_mapping,
                                 download_raw=download_raw, verbose=True)
    else:
        return download_xena_cohorts(cohorts, output_dir, category_enum, unzip=unzip,
                                     download_mapping=download_mapping,
                                     download_raw=download_raw,
                                     dataset_ids=dataset_ids, verbose=True)


def download_cbioportal(
    cohorts: list[str],
    output_dir: str = None,
    download_all_flag: bool = False,
    confirm: bool = True,
    verbose: bool = True,
) -> dict[str, bool]:
    """
    Download cohorts from cBioPortal via the REST API.

    Args:
        cohorts: List of cohort codes (must match YAML configs in data/cbioportal/configs/)
        output_dir: Optional output directory
        download_all_flag: Download all configured cohorts
        confirm: Ask for confirmation before downloading
        verbose: Print progress

    Returns:
        Dictionary mapping cohort codes to success status
    """
    if download_all_flag:
        return download_cbioportal_all(output_dir=output_dir, verbose=verbose, confirm=confirm)
    return download_cbioportal_cohorts(cohorts, output_dir=output_dir, verbose=verbose, confirm=confirm)


def download_tcia(
    cohorts: list[str],
    output_dir: str = None,
    download_all_flag: bool = False,
    download_images: bool = False,
    manifest_only: bool = False,
    manifest_path: str = None,
    unzip: bool = True,
    confirm: bool = True
) -> dict[str, bool]:
    """
    Download TCIA manifests and/or images for cohorts.

    Args:
        cohorts: List of cohort codes
        output_dir: Optional output directory
        download_all_flag: Download all available cohorts
        download_images: If True, run nbia-data-retriever to download images
        manifest_only: If True, download only manifests (not images)
        manifest_path: Path to existing manifest file to use for image download
        unzip: Whether to extract gzipped files after download
        confirm: If True, ask for confirmation before downloading

    Returns:
        Dictionary mapping cohort codes to success status
    """
    # Determine what to download based on flags
    # --manifest-only: download_images=False (only manifest)
    # default (no flags): download_images=True (manifest + images)
    # --manifest <path>: use existing manifest, download_images=True
    if manifest_only:
        actual_download_images = False
    elif manifest_path:
        # Using existing manifest, only download images
        actual_download_images = True
    else:
        # Default: download manifest and images
        actual_download_images = True

    if download_all_flag:
        return download_tcia_all(output_dir, download_images=actual_download_images, manifest_path=manifest_path, unzip=unzip, verbose=True, confirm=confirm)
    else:
        return download_tcia_cohorts(cohorts, output_dir, download_images=actual_download_images, manifest_path=manifest_path, unzip=unzip, verbose=True, confirm=confirm)


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
        raise ValueError(
            f"Unknown category: {category_str}. Available: {', '.join(category_map.keys())}")

    return category_map[cat_lower]


def list_cohorts(source: str, search: str = None, cancer_type: str = None) -> None:
    """List available cohorts for a source."""
    if source == "xena":
        cohorts = get_xena_cohorts()
        print("Available Xena Browser Cohorts:")
        print("=" * 80)
        for cohort in sorted(cohorts):
            print(f"  {cohort}")
        print("=" * 80)
        print(f"Total: {len(cohorts)} cohorts")
    elif source == "tcia":
        cohorts = get_tcia_cohorts()
        print("Available TCIA Cohorts:")
        print("=" * 80)
        for cohort in sorted(cohorts):
            print(f"  {cohort}")
        print("=" * 80)
        print(f"Total: {len(cohorts)} cohorts")
    elif source == "cbioportal":
        if search or cancer_type:
            # Live API search
            print(f"Searching cBioPortal studies"
                  + (f" for '{search}'" if search else "")
                  + (f" (cancer type: {cancer_type})" if cancer_type else "") + "…")
            studies = cbioportal_list_studies(keyword=search, cancer_type_id=cancer_type)
            print("=" * 80)
            print(f"  {'Study ID':<35} {'Cancer Type':<12}  Name")
            print(f"  {'-'*35} {'-'*12}  {'-'*30}")
            for s in studies:
                print(f"  {s['studyId']:<35} {s.get('cancerTypeId',''):<12}  {s.get('name','')}")
            print("=" * 80)
            print(f"Total: {len(studies)} studies")
        else:
            # Config-file-based cohorts
            cohorts = get_cbioportal_cohorts()
            print("cBioPortal cohorts with local configs (data/cbioportal/configs/):")
            print("  (use --search or --cancer-type to search the live cBioPortal API)")
            print("=" * 80)
            for cohort in sorted(cohorts):
                print(f"  {cohort}")
            print("=" * 80)
            print(f"Total: {len(cohorts)} configured cohorts")
    else:
        print(f"Unknown source: {source}")


def register_subcommand(subparsers):
    """Register the download subcommand."""
    parser = subparsers.add_parser(
        "download",
        description="Download cancer data from various sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Download data from UCSC Xena Browser, TCIA, or cBioPortal",
        epilog="""
Examples:
  # Download BRCA from Xena Browser
  oncolearn download --xena --cohorts BRCA

  # Download and extract gzipped files
  oncolearn download --xena --cohorts BRCA --unzip

  # Download only mutation data
  oncolearn download --xena --cohorts BRCA --category mutation

  # Download TCIA manifest only (no images)
  oncolearn download --tcia --cohorts BRCA --manifest-only

  # Download cBioPortal BRCA cohort (all configured datasets)
  oncolearn download --cbioportal --cohorts BRCA

  # List configured cBioPortal cohorts
  oncolearn download --cbioportal --list

  # Search the live cBioPortal API for breast cancer studies
  oncolearn download --cbioportal --list --search breast --cancer-type brca

  # Download all configured cBioPortal cohorts without confirmation
  oncolearn download --cbioportal --all --yes
        """
    )

    # Source selection (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--xena", action="store_true", help="Download from UCSC Xena Browser")
    source_group.add_argument(
        "--tcia", action="store_true", help="Download TCIA imaging manifests")
    source_group.add_argument(
        "--cbioportal", action="store_true", help="Download from cBioPortal via REST API")

    # Action selection (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--cohorts", type=str, help="Cohort code(s), comma-separated (e.g., BRCA,LUAD)")
    action_group.add_argument(
        "--all", action="store_true", help="Download all available cohorts")
    action_group.add_argument(
        "--list", action="store_true", help="List available cohorts and exit")

    # Optional arguments
    parser.add_argument("--category", type=str,
                        help="Filter datasets by category (Xena only)")
    parser.add_argument(
        "--ids", type=str, help="Specific dataset ID(s) to download, comma-separated (Xena only)")
    parser.add_argument("--output", type=str, help="Custom output directory")
    parser.add_argument("--download-images", action="store_true",
                        help="Download actual images using nbia-data-retriever (TCIA only) - DEPRECATED: images are downloaded by default now")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Download only manifest files, not images (TCIA only)")
    parser.add_argument("--manifest", type=str,
                        help="Path to existing manifest file to use for downloading images (TCIA only)")
    parser.add_argument("--unzip", action="store_true",
                        default=False, help="Extract gzipped files after download")
    parser.add_argument("--mapping", action="store_true", default=False,
                        help="Download gene mapping files (Xena only)")
    parser.add_argument("--raw", action="store_true", default=False,
                        help="Download raw data files (Xena only)")
    parser.add_argument("--yes", "-y", action="store_true", default=False,
                        help="Skip confirmation prompts and proceed with download")
    # cBioPortal-specific
    parser.add_argument("--search", type=str, default=None,
                        help="Search keyword for cBioPortal study discovery (used with --list)")
    parser.add_argument("--cancer-type", type=str, default=None,
                        help="Filter cBioPortal studies by cancer type ID (e.g. 'brca') (used with --list)")

    # Set the function to call when this subcommand is used
    parser.set_defaults(func=execute)


def execute(args):
    """Execute the download command."""

    # Determine source
    if args.xena:
        source = "xena"
    elif args.tcia:
        source = "tcia"
    else:
        source = "cbioportal"

    # Handle list action
    if args.list:
        search = getattr(args, 'search', None)
        cancer_type = getattr(args, 'cancer_type', None)
        list_cohorts(source, search=search, cancer_type=cancer_type)
        return

    # Category filtering only works with Xena
    if args.category and source != "xena":
        print("ERROR: --category can only be used with --xena")
        sys.exit(1)

    # TCIA-specific flags guard
    if source != "tcia":
        if hasattr(args, 'download_images') and args.download_images:
            print("ERROR: --download-images can only be used with --tcia")
            sys.exit(1)
        if hasattr(args, 'manifest_only') and args.manifest_only:
            print("ERROR: --manifest-only can only be used with --tcia")
            sys.exit(1)
        if hasattr(args, 'manifest') and args.manifest:
            print("ERROR: --manifest can only be used with --tcia")
            sys.exit(1)

    # Validate TCIA flag combinations
    if source == "tcia":
        manifest_only = hasattr(args, 'manifest_only') and args.manifest_only
        manifest_path = getattr(args, 'manifest', None)
        if manifest_only and manifest_path:
            print("ERROR: Cannot use both --manifest-only and --manifest together")
            sys.exit(1)

    # Parse cohorts
    if args.all:
        if source == "xena":
            cohort_list = get_xena_cohorts()
        elif source == "tcia":
            cohort_list = get_tcia_cohorts()
        else:
            cohort_list = get_cbioportal_cohorts()
    else:
        cohort_list = [c.strip().upper() for c in args.cohorts.split(',') if c.strip()]

    # Download cohorts
    unzip = hasattr(args, 'unzip') and args.unzip
    confirm = not (hasattr(args, 'yes') and args.yes)

    if source == "xena":
        download_mapping = hasattr(args, 'mapping') and args.mapping
        download_raw = hasattr(args, 'raw') and args.raw
        dataset_ids = None
        if hasattr(args, 'ids') and args.ids:
            dataset_ids = [d.strip() for d in args.ids.split(',')]
        results = download_xena(cohort_list, args.output, args.category, args.all, unzip,
                                download_mapping, download_raw, dataset_ids)
    elif source == "tcia":
        download_images = hasattr(args, 'download_images') and args.download_images
        manifest_only = hasattr(args, 'manifest_only') and args.manifest_only
        manifest_path = getattr(args, 'manifest', None)
        results = download_tcia(cohort_list, args.output,
                                args.all, download_images, manifest_only, manifest_path, unzip, confirm)
    else:  # cbioportal
        results = download_cbioportal(cohort_list, args.output, args.all, confirm=confirm)

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


def main():
    """Direct entry point for backwards compatibility."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download cancer data from various sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download BRCA from Xena Browser
  download --xena --cohorts BRCA
  
  # Download and extract gzipped files
  download --xena --cohorts BRCA --unzip
  
  # Download only mutation data
  download --xena --cohorts BRCA --category mutation
  
  # Download multiple cohorts
  download --xena --cohorts BRCA,LUAD,ACC
  
  # Download TCIA manifest only
  download --tcia --cohorts BRCA
  
  # Download TCIA manifest and images
  download --tcia --cohorts BRCA --download-images
  
  # Download all Xena cohorts
  download --xena --all
  
  # List available cohorts
  download --xena --list
  download --tcia --list
        """
    )

    # Source selection (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--xena", action="store_true", help="Download from UCSC Xena Browser")
    source_group.add_argument(
        "--tcia", action="store_true", help="Download TCIA imaging manifests")

    # Action selection (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--cohorts", type=str, help="Cohort code(s), comma-separated (e.g., BRCA,LUAD)")
    action_group.add_argument(
        "--all", action="store_true", help="Download all available cohorts")
    action_group.add_argument(
        "--list", action="store_true", help="List available cohorts and exit")

    # Optional arguments
    parser.add_argument("--category", type=str,
                        help="Filter datasets by category (Xena only)")
    parser.add_argument("--output", type=str, help="Custom output directory")
    parser.add_argument("--download-images", action="store_true",
                        help="Download actual images using nbia-data-retriever (TCIA only)")
    parser.add_argument("--unzip", action="store_true",
                        default=False, help="Extract gzipped files after download")
    parser.add_argument("--yes", "-y", action="store_true", default=False,
                        help="Skip confirmation prompts and proceed with download")

    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
