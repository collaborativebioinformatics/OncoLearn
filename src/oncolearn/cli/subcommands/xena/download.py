"""Xena Browser download logic."""

import sys

from oncolearn.api.dataset import DataCategory
from oncolearn.api.xenabrowser.download import (
    download_all as download_xena_all,
    download_cohorts as download_xena_cohorts,
    get_available_cohorts as get_xena_cohorts,
)


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
            f"Unknown category: {category_str}. Available: {', '.join(category_map.keys())}"
        )
    return category_map[cat_lower]


def list_cohorts() -> None:
    """Print available Xena Browser cohorts."""
    cohorts = get_xena_cohorts()
    print("Available Xena Browser Cohorts:")
    print("=" * 80)
    for cohort in sorted(cohorts):
        print(f"  {cohort}")
    print("=" * 80)
    print(f"Total: {len(cohorts)} cohorts")


def download(args) -> None:
    """Execute the Xena download action."""
    if args.list:
        list_cohorts()
        sys.exit(0)

    category_enum = None
    if args.category:
        category_enum = parse_category(args.category)

    dataset_ids = None
    if args.ids:
        dataset_ids = [d.strip() for d in args.ids.split(",")]

    if args.all:
        cohort_list = get_xena_cohorts()
        results = download_xena_all(
            args.output, category_enum,
            unzip=args.unzip,
            download_mapping=args.mapping,
            download_raw=args.raw,
            verbose=True,
        )
    else:
        cohort_list = [c.strip().upper() for c in args.cohorts.split(",") if c.strip()]
        results = download_xena_cohorts(
            cohort_list, args.output, category_enum,
            unzip=args.unzip,
            download_mapping=args.mapping,
            download_raw=args.raw,
            dataset_ids=dataset_ids,
            verbose=True,
        )

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
