"""TCIA download logic."""

import sys

from oncolearn.api.tcia.download import (
    download_all as download_tcia_all,
    download_cohorts as download_tcia_cohorts,
    get_available_cohorts as get_tcia_cohorts,
)


def list_cohorts() -> None:
    """Print available TCIA cohorts."""
    cohorts = get_tcia_cohorts()
    print("Available TCIA Cohorts:")
    print("=" * 80)
    for cohort in sorted(cohorts):
        print(f"  {cohort}")
    print("=" * 80)
    print(f"Total: {len(cohorts)} cohorts")


def download(args) -> None:
    """Execute the TCIA download action."""
    if args.list:
        list_cohorts()
        sys.exit(0)

    # Validate flag combinations
    manifest_only = getattr(args, "manifest_only", False)
    manifest_path = getattr(args, "manifest", None)
    if manifest_only and manifest_path:
        print("ERROR: Cannot use both --manifest-only and --manifest together")
        sys.exit(1)

    unzip = getattr(args, "unzip", False)
    confirm = not getattr(args, "yes", False)

    if getattr(args, "all", False):
        cohort_list = get_tcia_cohorts()
    else:
        cohort_list = [c.strip().upper() for c in args.cohorts.split(",") if c.strip()]

    if manifest_only:
        actual_download_images = False
    else:
        actual_download_images = True

    if getattr(args, "all", False):
        results = download_tcia_all(
            args.output,
            download_images=actual_download_images,
            manifest_path=manifest_path,
            unzip=unzip,
            verbose=True,
            confirm=confirm,
        )
    else:
        results = download_tcia_cohorts(
            cohort_list,
            args.output,
            download_images=actual_download_images,
            manifest_path=manifest_path,
            unzip=unzip,
            verbose=True,
            confirm=confirm,
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
