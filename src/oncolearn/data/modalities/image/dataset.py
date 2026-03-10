import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from oncolearn.registry.modalities import register_modality
from oncolearn.api.tcia.tcia_dataset import TCIADataset
from oncolearn.data.modalities.image.loaders import DEFAULT_LOADERS

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
        self.transform = transform
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
                if self.transform:
                    loaded_tensor = self.transform(loaded_image)
                else:
                    loaded_tensor = transforms.ToTensor()(loaded_image)
            tensors.append(loaded_tensor)
            
        sequence_tensor = torch.stack(tensors, dim=0) # (N, 3, 224, 224)

        return {
            self.batch_key: sequence_tensor,
            "patient_id": patient_id
        }


@register_modality("image", "oncolearn.modality.image")
class ImageDataModule(pl.LightningDataModule):
    """
    API-first LightningDataModule for Images.
    Uses TCIADataset to ensure metadata and images exist before yielding them to loaders.
    """
    def __init__(
        self,
        tcia_manifest_url: Optional[str] = None,
        cohort_code: str = "BRCA",
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 16,
        num_workers: int = 4,
        base_directory: str = "data/tcia",
        train_split: float = 0.8,
        seed: int = 42,
        n_slices: int = 5,
        prefer_mr: bool = True,
        batch_key: str = "image",
        files: Optional[List[Path]] = None,  # accepted but unused (images come from TCIA)
    ):
        super().__init__()
        self.name = "image"
        self.tcia_manifest_url = tcia_manifest_url
        self.cohort_name = cohort_code
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = Path(base_directory)
        self.train_split = train_split
        self.seed = seed
        self.n_slices = n_slices
        self.prefer_mr = prefer_mr
        self.batch_key = batch_key

        self.api_dataset = None
        if self.tcia_manifest_url:
            self.api_dataset = TCIADataset(
                name=f"TCIA_{self.cohort_name}",
                description="TCIA manifest dynamically loaded by ImageDataModule",
                url=self.tcia_manifest_url,
                filename=f"{self.cohort_name}.tcia",
                default_subdir=f"TCGA-{self.cohort_name}"
            )
            
        self.patient_to_files = {}
        self.patient_ids = []
        self._full_dataset: Optional["ImageDataset"] = None

    def prepare_data(self):
        """
        Download manifest and images via API if requested.
        Called only on 1 GPU.
        """
        if self.api_dataset is not None:
            # Trigger download
            self.api_dataset.download(
                output_dir=str(self.data_dir / "manifests"),
                download_images=True,
                confirm=False  # Auto-run in automated environments
            )

    def setup(self, stage: Optional[str] = None):
        """
        Parse local files and build patient IDs.
        """
        # Discover images in target directory
        target_dir = self.data_dir / f"TCGA-{self.cohort_name}"
        if not target_dir.exists():
            print(f"Warning: Image directory {target_dir} not found. Ensure prepare_data() succeeded.")
            self._full_dataset = ImageDataset({}, [], n_slices=self.n_slices, batch_key=self.batch_key)
            self.train_dataset = self.val_dataset = self.test_dataset = self._full_dataset
            return

        # Collect all valid image files and map to patient IDs
        self.patient_to_files = {}
        file_count = 0

        for file_path in target_dir.rglob("*"):
            if file_path.is_file():
                for loader in DEFAULT_LOADERS:
                    if loader.can_load(file_path):
                        p_id = self._extract_patient_id(file_path)
                        if p_id not in self.patient_to_files:
                            self.patient_to_files[p_id] = []
                        self.patient_to_files[p_id].append(file_path)
                        file_count += 1
                        break
                        
        # Prefer MR series over MG when both are present for a patient
        if self.prefer_mr:
            self.patient_to_files = self._filter_mr_preferred(self.patient_to_files)

        self.patient_ids = list(self.patient_to_files.keys())
        print(f"ImageDataModule mapped {file_count} valid image files across {len(self.patient_ids)} patients.")

        # Transforms matching the original pipeline: resize to 224×224,
        # flip + ±5° rotation for training augmentation, ToTensor for both.
        train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
        ])
        eval_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

        # Create base dataset (use eval_transform so slices are resized consistently)
        full_dataset = ImageDataset(
            self.patient_to_files, self.patient_ids,
            transform=eval_transform, n_slices=self.n_slices, batch_key=self.batch_key,
        )
        self._full_dataset = full_dataset

        if len(full_dataset) == 0:
            self.train_dataset = self.val_dataset = self.test_dataset = full_dataset
            return

        # Split
        total_size = len(full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = total_size - train_size

        generator = torch.Generator().manual_seed(self.seed)
        train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)

        self.train_dataset = ImageDataset(
            self.patient_to_files,
            [self.patient_ids[i] for i in train_ds.indices],
            transform=train_transform, n_slices=self.n_slices, batch_key=self.batch_key,
        )
        self.val_dataset = ImageDataset(
            self.patient_to_files,
            [self.patient_ids[i] for i in val_ds.indices],
            transform=eval_transform, n_slices=self.n_slices, batch_key=self.batch_key,
        )
        self.test_dataset = self.val_dataset

    @property
    def full_dataset(self) -> "ImageDataset":
        """Full dataset (all patients, no split) — available after setup()."""
        return self._full_dataset

    def setup_full(self, stage=None):
        """Ensure setup() has been called so full_dataset is available."""
        if self._full_dataset is None:
            self.setup(stage=stage)

    def _filter_mr_preferred(
        self, patient_to_files: Dict[str, List[Path]]
    ) -> Dict[str, List[Path]]:
        """
        For each patient, keep only MR-modality files when MR series are present.
        Falls back to all files when MR cannot be determined (e.g. non-DICOM).

        Reads one DICOM header per series directory (stop_before_pixels=True) to
        determine the Modality tag cheaply, matching the original pipeline's
        preference for MR over MG.
        """
        try:
            import pydicom
        except ImportError:
            return patient_to_files

        filtered: Dict[str, List[Path]] = {}
        for pid, files in patient_to_files.items():
            # Group files by their parent directory (= series)
            dir_to_files: Dict[Path, List[Path]] = {}
            for f in files:
                dir_to_files.setdefault(f.parent, []).append(f)

            # Read one header per series directory to get the Modality tag
            dir_modality: Dict[Path, Optional[str]] = {}
            for series_dir, series_files in dir_to_files.items():
                try:
                    ds = pydicom.dcmread(
                        str(series_files[0]), stop_before_pixels=True
                    )
                    dir_modality[series_dir] = getattr(ds, "Modality", None)
                except Exception:
                    dir_modality[series_dir] = None

            # If any MR series found, keep only MR files
            mr_dirs = {d for d, m in dir_modality.items() if m == "MR"}
            if mr_dirs:
                filtered[pid] = [f for f in files if f.parent in mr_dirs]
            else:
                filtered[pid] = files

        mr_filtered = sum(
            1 for pid in filtered
            if len(filtered[pid]) < len(patient_to_files[pid])
        )
        if mr_filtered:
            logger.info(
                "MR preference filter: removed MG files for %d patients.", mr_filtered
            )
        return filtered

    def _extract_patient_id(self, img_path: Path) -> str:
        """Helper to rip the patient ID out of the TCGA/TCIA formatted path."""
        parts = img_path.parts
        for part in parts:
            if part.startswith('TCGA-'):
                tcga_parts = part.split('-')
                if len(tcga_parts) >= 3:
                    return '-'.join(tcga_parts[:3])
        return img_path.stem

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
