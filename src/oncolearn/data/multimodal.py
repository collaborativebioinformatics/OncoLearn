from typing import List, Dict, Optional, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pytorch_lightning as pl

from oncolearn.registry.modalities import get_modality
import oncolearn.data.modalities  # noqa: F401 — triggers @register_modality decorators


class MultimodalDataset(Dataset):
    """
    A Dataset that wraps multiple uni-modal datasets and aligns them based on a common key.
    """
    def __init__(
        self,
        datasets: Dict[str, Dataset],
        join_on: str = "patient_id",
        strategy: str = "inner"
    ):
        """
        Args:
            datasets: Dictionary mapping modality name (e.g. 'image') to its PyTorch Dataset.
            join_on: The dictionary key yielded by underlying datasets to align on.
            strategy: Join strategy ('inner' only supported for now).
        """
        super().__init__()
        self.datasets = datasets
        self.join_on = join_on
        self.strategy = strategy
        self._indices_map = None  # List of dicts: [{'image': idx1, 'tabular': idx2}, ...]
        
        self._build_index()
        
    def _build_index(self):
        """
        Iterate over all underlying datasets (or their metadata) to find the common `join_on` keys
        and build an index mapping.
        
        Assumes each dataset has a `.patient_ids` attribute or similar fast indexing. 
        If not, we would have to exhaustively query `dataset[i][join_on]`, which is slow.
        For now, we expect each Dataset to implement `get_keys()` returning a list of the join_on values.
        """
        # Dictionary mapping modality name to a mapping of join_key -> index
        modality_to_key_map = {}
        for mod_name, ds in self.datasets.items():
            if not hasattr(ds, "get_keys"):
                raise AttributeError(
                    f"Dataset for modality {mod_name} must implement `get_keys()` "
                    f"returning a list of '{self.join_on}' identifiers."
                )
            
            keys = ds.get_keys()
            modality_to_key_map[mod_name] = {k: idx for idx, k in enumerate(keys)}
            
        # Find intersecting keys (inner join)
        if self.strategy != "inner":
            raise NotImplementedError(f"Strategy {self.strategy} not yet supported.")
            
        common_keys = None
        for key_map in modality_to_key_map.values():
            if common_keys is None:
                common_keys = set(key_map.keys())
            else:
                common_keys = common_keys.intersection(key_map.keys())
                
        common_keys = sorted(list(common_keys))
        
        # Build strict index paths
        self._indices_map = []
        for key in common_keys:
            idx_record = {}
            for mod_name, key_map in modality_to_key_map.items():
                idx_record[mod_name] = key_map[key]
            self._indices_map.append(idx_record)
            
        print(f"MultimodalDataset aligned {len(self._indices_map)} samples via {self.strategy} join on {self.join_on}")

    def __len__(self):
        return len(self._indices_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary grouping the separate modalities.
        e.g. {'image': tensor, 'tabular': tensor, 'patient_id': 'TCGA-...'}
        """
        record_indices = self._indices_map[idx]
        
        # We will merge the outputs. We assume labels match across modalities.
        combined = {}
        for mod_name, ds in self.datasets.items():
            mod_idx = record_indices[mod_name]
            mod_data = ds[mod_idx]
            
            # Put the actual tensor inside the combined dict under the modality name
            combined[mod_name] = mod_data.get(mod_name, mod_data)
            
            # Carry over the join key and label (assuming first dataset defines standard label)
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
        num_workers: int = 4
    ):
        """
        Args:
            modalities: List of modality names (to fetch from registry) or DataModules.
            join_on: Identifier key to merge patients.
            strategy: Inner join vs outer join.
            batch_size: DataLoader batch size.
            num_workers: DataLoader num workers.
        """
        super().__init__()
        self.join_on = join_on
        self.strategy = strategy
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Instantiate any string modalities from registry
        self.datamodules = {}
        for mod in modalities:
            if isinstance(mod, str):
                mod_cls = get_modality(mod)
                # We assume registry classes can be instantiated neutrally here,
                # or we pass kwargs if needed later.
                dm_instance = mod_cls()
                self.datamodules[mod] = dm_instance
            elif isinstance(mod, pl.LightningDataModule):
                # We need a name. Might check attribute or registry.
                # Assuming the module has a .name attribute for brevity.
                name = getattr(mod, "name", mod.__class__.__name__.lower())
                self.datamodules[name] = mod
            else:
                raise ValueError(f"Unknown modality type: {type(mod)}")

    def prepare_data(self):
        """Called only on 1 GPU for downloading."""
        for name, dm in self.datamodules.items():
            dm.prepare_data()
            
    def setup(self, stage: Optional[str] = None):
        """Called on every GPU. Dispatches to underlying datamodules, then builds unified PyTorch Datasets."""
        for name, dm in self.datamodules.items():
            dm.setup(stage=stage)
            
        if stage == 'fit' or stage is None:
            train_datasets = {name: dm.train_dataset for name, dm in self.datamodules.items()}
            val_datasets = {name: dm.val_dataset for name, dm in self.datamodules.items()}
            
            self.train_dataset = MultimodalDataset(train_datasets, self.join_on, self.strategy)
            self.val_dataset = MultimodalDataset(val_datasets, self.join_on, self.strategy)
            
        if stage == 'test' or stage is None:
            test_datasets = {name: dm.test_dataset for name, dm in self.datamodules.items()}
            self.test_dataset = MultimodalDataset(test_datasets, self.join_on, self.strategy)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
