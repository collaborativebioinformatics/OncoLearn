import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from oncolearn.data.modalities.loaders import DEFAULT_LOADERS

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """
    Internal PyTorch Dataset for loading sequence of images per patient.
    Matches the prior functionality of yielding `(N, 3, 224, 224)` for hierarchical encoders.
    """
    def __init__(
        self,
        patient_to_files: Dict[str, List[Path]],
        patient_ids: List[str],
        transform: Optional[Any] = None,
        n_slices: int = 5,
        batch_key: str = "image",
    ):
        self.patient_to_files = patient_to_files
        self.patient_ids = patient_ids
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.n_slices = n_slices
        self.batch_key = batch_key

    def get_keys(self) -> List[str]:
        """Method required by MultimodalDataset to align records."""
        return self.patient_ids

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        patient_id = self.patient_ids[idx]
        files = self.patient_to_files.get(patient_id, [])

        # Sort files so sampling is deterministic across the volume
        files = sorted(files)

        # Uniform sampling of N slices
        n_total = len(files)
        if n_total == 0:
            # Fallback to zero tensor (N_slices, 3, 224, 224)
            # In practical setups, we drop patients without images beforehand
            return {
                self.batch_key: torch.zeros((self.n_slices, 3, 224, 224), dtype=torch.float32),
                "patient_id": patient_id
            }

        indices = [int(i * (n_total / self.n_slices)) for i in range(self.n_slices)]
        sampled_files = [files[min(i, n_total - 1)] for i in indices]

        tensors = []
        for img_path in sampled_files:
            loaded_image = None
            for loader_cls in DEFAULT_LOADERS:
                if loader_cls.can_load(img_path):
                    loaded_image = loader_cls.load(img_path)
                    break

            if loaded_image is None:
                # Fallback to prevent crash on corrupted individual slices
                loaded_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            else:
                loaded_tensor = self.transform(loaded_image)
            tensors.append(loaded_tensor)

        sequence_tensor = torch.stack(tensors, dim=0)  # (N, 3, 224, 224)

        return {
            self.batch_key: sequence_tensor,
            "patient_id": patient_id
        }
