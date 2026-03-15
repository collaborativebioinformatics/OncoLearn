"""
MultimodalDataset: aligns multiple uni-modal datasets by patient ID.
"""
import logging
from typing import Dict, List, Optional, Set, Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


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

        n_filtered = " (filtered from allowed_ids)" if self.allowed_ids is not None else ""
        logger.info(
            "MultimodalDataset aligned %d samples via %s join on %s%s",
            len(self._indices_map), self.strategy, self.join_on, n_filtered,
        )

    def get_labels(self) -> List[int]:
        """Return the integer label for each sample.

        Iterates modalities in order and returns the first valid (non-NaN)
        label found.  This allows the clinical modality to supply stage labels
        even when the gene modality's full dataset has no PAM50 annotations.
        """
        labels = []
        for record_indices in self._indices_map:
            label = None
            for mod_name, ds in self.datasets.items():
                idx = record_indices[mod_name]
                if hasattr(ds, "labels") and ds.labels is not None:
                    try:
                        label = int(ds.labels[idx])
                        break
                    except (ValueError, TypeError):
                        continue  # NaN or non-castable → try next modality
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
