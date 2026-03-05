from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from oncolearn.registry.modalities import get_modality
import oncolearn.data.modalities  # noqa: F401 — triggers @register_modality decorators
from oncolearn.data.split_utils import read_id_file


def _load_split_ids(splits_dir: str) -> Dict[str, Optional[Set[str]]]:
    """Load patient IDs from split files in *splits_dir*.

    Expects ``train.txt``, ``test.txt``, ``validation.txt`` (one ID per line).
    Falls back: missing validation.txt → use test IDs; missing test.txt → use val IDs.
    """
    p = Path(splits_dir)
    train_ids = read_id_file(p / "train.txt")
    test_ids = read_id_file(p / "test.txt")
    val_ids = read_id_file(p / "validation.txt")
    if val_ids is None:
        val_ids = test_ids
    if test_ids is None:
        test_ids = val_ids
    return {"train": train_ids, "val": val_ids, "test": test_ids}


class MultimodalDataset(Dataset):
    """
    A Dataset that wraps multiple uni-modal datasets and aligns them based on a common key.
    """
    def __init__(
        self,
        datasets: Dict[str, Dataset],
        join_on: str = "patient_id",
        strategy: str = "inner",
        allowed_ids: Optional[Set[str]] = None,
    ):
        """
        Args:
            datasets: Dictionary mapping modality name to its PyTorch Dataset.
            join_on: The dictionary key yielded by underlying datasets to align on.
            strategy: Join strategy ('inner' only supported for now).
            allowed_ids: If set, only patients whose ID is in this set are included.
        """
        super().__init__()
        self.datasets = datasets
        self.join_on = join_on
        self.strategy = strategy
        self.allowed_ids = allowed_ids
        self._indices_map = None

        self._build_index()

    def _build_index(self):
        """Build the cross-modality alignment index."""
        modality_to_key_map = {}
        for mod_name, ds in self.datasets.items():
            if not hasattr(ds, "get_keys"):
                raise AttributeError(
                    f"Dataset for modality {mod_name} must implement `get_keys()` "
                    f"returning a list of '{self.join_on}' identifiers."
                )
            keys = ds.get_keys()
            modality_to_key_map[mod_name] = {k: idx for idx, k in enumerate(keys)}

        if self.strategy != "inner":
            raise NotImplementedError(f"Strategy {self.strategy} not yet supported.")

        common_keys = None
        for key_map in modality_to_key_map.values():
            if common_keys is None:
                common_keys = set(key_map.keys())
            else:
                common_keys = common_keys.intersection(key_map.keys())

        # Filter to allowed_ids when provided (external split)
        if self.allowed_ids is not None:
            common_keys = common_keys.intersection(self.allowed_ids)

        common_keys = sorted(list(common_keys))

        self._indices_map = []
        for key in common_keys:
            idx_record = {"_patient_id": key}
            for mod_name, key_map in modality_to_key_map.items():
                idx_record[mod_name] = key_map[key]
            self._indices_map.append(idx_record)

        n_filtered = f" (filtered from allowed_ids)" if self.allowed_ids is not None else ""
        print(
            f"MultimodalDataset aligned {len(self._indices_map)} samples "
            f"via {self.strategy} join on {self.join_on}{n_filtered}"
        )

    def get_labels(self) -> List[int]:
        """Return the integer label for each sample, reading from the first labelled dataset."""
        labels = []
        for record_indices in self._indices_map:
            label = None
            for mod_name, ds in self.datasets.items():
                idx = record_indices[mod_name]
                if hasattr(ds, "labels") and ds.labels is not None:
                    label = int(ds.labels[idx])
                    break
            labels.append(label)
        return labels

    def __len__(self):
        return len(self._indices_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record_indices = self._indices_map[idx]

        combined = {}
        for mod_name, ds in self.datasets.items():
            mod_idx = record_indices[mod_name]
            mod_data = ds[mod_idx]

            combined[mod_name] = mod_data.get(mod_name, mod_data)

            if self.join_on not in combined and self.join_on in mod_data:
                combined[self.join_on] = mod_data[self.join_on]
            if "label" not in combined and "label" in mod_data:
                combined["label"] = mod_data["label"]

        return combined


class MultimodalDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning builder that accepts either string registry names
    or instantiated DataModules, and merges them for multimodal training.
    """
    def __init__(
        self,
        modalities: List[Union[str, pl.LightningDataModule]],
        join_on: str = "patient_id",
        strategy: str = "inner",
        batch_size: int = 16,
        num_workers: int = 4,
        splits_dir: Optional[str] = None,
        num_classes: Optional[int] = None,
    ):
        """
        Args:
            modalities: List of modality names or DataModules.
            join_on: Identifier key to merge patients.
            strategy: Inner join vs outer join.
            batch_size: DataLoader batch size.
            num_workers: DataLoader num workers.
            splits_dir: Path to folder with train/test/validation.txt split files.
                        If provided, split files override per-modality random splits.
        """
        super().__init__()
        self.join_on = join_on
        self.strategy = strategy
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splits_dir = splits_dir
        self.num_classes = num_classes
        self.class_weights: Optional[torch.Tensor] = None

        self.datamodules = {}
        for mod in modalities:
            if isinstance(mod, str):
                mod_cls = get_modality(mod)
                dm_instance = mod_cls()
                self.datamodules[mod] = dm_instance
            elif isinstance(mod, pl.LightningDataModule):
                name = getattr(mod, "name", mod.__class__.__name__.lower())
                self.datamodules[name] = mod
            else:
                raise ValueError(f"Unknown modality type: {type(mod)}")

    def prepare_data(self):
        for name, dm in self.datamodules.items():
            dm.prepare_data()

    def setup(self, stage: Optional[str] = None):
        if self.splits_dir:
            # External split files — build full datasets then filter by ID
            for name, dm in self.datamodules.items():
                dm.setup_full(stage=stage)

            splits = _load_split_ids(self.splits_dir)
            full_datasets = {name: dm.full_dataset for name, dm in self.datamodules.items()}

            if stage == "fit" or stage is None:
                self.train_dataset = MultimodalDataset(
                    full_datasets, self.join_on, self.strategy,
                    allowed_ids=splits["train"],
                )
                self.val_dataset = MultimodalDataset(
                    full_datasets, self.join_on, self.strategy,
                    allowed_ids=splits["val"],
                )

            if stage == "test" or stage is None:
                self.test_dataset = MultimodalDataset(
                    full_datasets, self.join_on, self.strategy,
                    allowed_ids=splits["test"],
                )
        else:
            # Default: per-modality random splits
            for name, dm in self.datamodules.items():
                dm.setup(stage=stage)

            if stage == "fit" or stage is None:
                self.train_dataset = MultimodalDataset(
                    {name: dm.train_dataset for name, dm in self.datamodules.items()},
                    self.join_on, self.strategy,
                )
                self.val_dataset = MultimodalDataset(
                    {name: dm.val_dataset for name, dm in self.datamodules.items()},
                    self.join_on, self.strategy,
                )

            if stage == "test" or stage is None:
                self.test_dataset = MultimodalDataset(
                    {name: dm.test_dataset for name, dm in self.datamodules.items()},
                    self.join_on, self.strategy,
                )

        # Compute inverse-frequency class weights from the training split
        if (stage == "fit" or stage is None) and hasattr(self, "train_dataset"):
            self._compute_class_weights()

    def _compute_class_weights(self) -> None:
        labels = self.train_dataset.get_labels()
        if not labels or any(l is None for l in labels):
            return
        counts = Counter(labels)
        n_classes = self.num_classes if self.num_classes is not None else max(counts) + 1
        total = len(labels)
        self.class_weights = torch.tensor(
            [total / (n_classes * counts.get(c, 1)) for c in range(n_classes)],
            dtype=torch.float32,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
        )
