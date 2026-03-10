"""cBioPortal download logic."""

import sys

from oncolearn.api.cbioportal.download import (
    download_all as download_cbioportal_all,
    download_cohorts as download_cbioportal_cohorts,
    get_available_cohorts as get_cbioportal_cohorts,
)
from oncolearn.api.cbioportal.download import list_studies as cbioportal_list_studies


def list_cohorts(search: str = None, cancer_type: str = None) -> None:
    """Print available cBioPortal cohorts."""
    if search or cancer_type:
        print(
            "Searching cBioPortal studies"
            + (f" for '{search}'" if search else "")
            + (f" (cancer type: {cancer_type})" if cancer_type else "")
            + "…"
        )
        studies = cbioportal_list_studies(keyword=search, cancer_type_id=cancer_type)
        print("=" * 80)
        print(f"  {'Study ID':<35} {'Cancer Type':<12}  Name")
        print(f"  {'-'*35} {'-'*12}  {'-'*30}")
        for s in studies:
            print(f"  {s['studyId']:<35} {s.get('cancerTypeId',''):<12}  {s.get('name','')}")
        print("=" * 80)
        print(f"Total: {len(studies)} studies")
    else:
        cohorts = get_cbioportal_cohorts()
        print("cBioPortal cohorts with local configs (data/cbioportal/configs/):")
        print("  (use --search or --cancer-type to search the live cBioPortal API)")
        print("=" * 80)
        for cohort in sorted(cohorts):
            print(f"  {cohort}")
        print("=" * 80)
        print(f"Total: {len(cohorts)} configured cohorts")


def download(args) -> None:
    """Execute the cBioPortal download action."""
    if args.list:
        list_cohorts(
            search=getattr(args, "search", None),
            cancer_type=getattr(args, "cancer_type", None),
        )
        sys.exit(0)

    confirm = not getattr(args, "yes", False)

    if args.all:
        results = download_cbioportal_all(
            output_dir=args.output, verbose=True, confirm=confirm
        )
    else:
        cohort_list = [c.strip().upper() for c in args.cohorts.split(",") if c.strip()]
        results = download_cbioportal_cohorts(
            cohort_list, output_dir=args.output, verbose=True, confirm=confirm
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
