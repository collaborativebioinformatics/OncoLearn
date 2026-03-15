"""
Utilities for reading/writing patient-ID split files and generating K-fold splits.

Split files contain one patient_id per line, e.g.::

    TCGA-A1-A0SD
    TCGA-A2-A0CM
    ...
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Set


def read_id_file(path: Path) -> Optional[Set[str]]:
    """Read one patient_id per line from *path*.

    Returns:
        Set of patient IDs, or ``None`` if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return None
    ids = set()
    with path.open("r") as f:
        for line in f:
            pid = line.strip()
            if pid:
                ids.add(pid)
    return ids


def write_id_file(path: Path, ids: Iterable[str]) -> None:
    """Write patient IDs one per line to *path*, sorted for reproducibility.

    Parent directories are created as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for pid in sorted(set(ids)):
            f.write(pid + "\n")


def generate_kfold_splits(
    patient_ids: List[str],
    labels: List[int],
    output_dir: Path,
    n_splits: int = 5,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> List[Path]:
    """Generate stratified K-fold splits and save them to disk.

    For each fold *N*:

    * **test** — the held-out fold
    * **val** — *val_fraction* of the training fold (stratified)
    * **train** — the remainder of the training fold

    Files are written to ``output_dir/fold_N/{train,test,validation}.txt``,
    one patient_id per line.

    Args:
        patient_ids: List of patient identifiers (same length as *labels*).
        labels: Integer class label for each patient.
        output_dir: Root directory under which ``fold_0/``, ``fold_1/``, … are created.
        n_splits: Number of folds (default 5).
        val_fraction: Fraction of the training set to reserve for validation (default 0.1).
        seed: Random seed for reproducibility (default 42).

    Returns:
        List of ``Path`` objects, one per fold directory.
    """
    try:
        from sklearn.model_selection import StratifiedKFold, train_test_split
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for split generation: pip install scikit-learn"
        ) from exc

    output_dir = Path(output_dir)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Deduplicate while preserving label alignment
    seen: set = set()
    ids_arr: List[str] = []
    lbls_arr: List[int] = []
    for pid, lbl in zip(patient_ids, labels):
        if pid not in seen:
            seen.add(pid)
            ids_arr.append(pid)
            lbls_arr.append(lbl)

    fold_dirs: List[Path] = []

    for fold_idx, (train_val_indices, test_indices) in enumerate(
        skf.split(ids_arr, lbls_arr)
    ):
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        test_ids = [ids_arr[i] for i in test_indices]
        train_val_ids = [ids_arr[i] for i in train_val_indices]
        train_val_labels = [lbls_arr[i] for i in train_val_indices]

        # Stratified val split from the train+val portion
        _use_stratified = val_fraction > 0 and len(set(train_val_labels)) > 1
        if _use_stratified:
            try:
                train_ids, val_ids, _, _ = train_test_split(
                    train_val_ids,
                    train_val_labels,
                    test_size=val_fraction,
                    random_state=seed,
                    stratify=train_val_labels,
                )
            except ValueError:
                _use_stratified = False  # fall through to non-stratified below
        if not _use_stratified:
            # Fallback: non-stratified split (single-class, rare class, or val_fraction=0)
            n_val = max(1, int(len(train_val_ids) * val_fraction))
            val_ids = train_val_ids[:n_val]
            train_ids = train_val_ids[n_val:]

        write_id_file(fold_dir / "train.txt", train_ids)
        write_id_file(fold_dir / "test.txt", test_ids)
        write_id_file(fold_dir / "validation.txt", val_ids)

        # Sanity-check: no patient should appear in more than one split
        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)
        tv_overlap = train_set & val_set
        te_overlap = train_set & test_set
        ve_overlap = val_set & test_set
        if tv_overlap or te_overlap or ve_overlap:
            raise AssertionError(
                f"fold_{fold_idx} has overlapping patients: "
                f"train∩val={tv_overlap}, train∩test={te_overlap}, val∩test={ve_overlap}"
            )

        label_map = dict(zip(ids_arr, lbls_arr))
        train_counts = Counter(label_map[pid] for pid in train_ids)
        val_counts = Counter(label_map[pid] for pid in val_ids)
        test_counts = Counter(label_map[pid] for pid in test_ids)

        print(
            f"  fold_{fold_idx}: "
            f"train={len(train_ids)} {dict(sorted(train_counts.items()))}  "
            f"val={len(val_ids)} {dict(sorted(val_counts.items()))}  "
            f"test={len(test_ids)} {dict(sorted(test_counts.items()))}"
        )

        fold_dirs.append(fold_dir)

    return fold_dirs
