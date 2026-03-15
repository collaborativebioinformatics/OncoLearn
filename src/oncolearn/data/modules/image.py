import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
from torchvision import transforms

from oncolearn.registry.modalities import register_modality
from oncolearn.api.tcia.tcia_dataset import TCIADataset
from oncolearn.data.modalities.loaders import DEFAULT_LOADERS
from oncolearn.data.modalities.image import ImageDataset
from .base import OncoDataModule

logger = logging.getLogger(__name__)


@register_modality("image", "oncolearn.modality.image")
class ImageDataModule(OncoDataModule):
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
        base_directory: str = "data/sources/tcia",
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
        self._full_dataset: Optional[ImageDataset] = None

    def prepare_data(self):
        """Download manifest and images via API if requested. Called only on 1 GPU."""
        if self.api_dataset is not None:
            self.api_dataset.download(
                output_dir=str(self.data_dir / "manifests"),
                download_images=True,
                confirm=False
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Parse local files and build patient IDs."""
        target_dir = self.data_dir / f"TCGA-{self.cohort_name}"
        if not target_dir.exists():
            logger.warning("Image directory %s not found. Ensure prepare_data() succeeded.", target_dir)
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

        if self.prefer_mr:
            self.patient_to_files = self._filter_mr_preferred(self.patient_to_files)

        self.patient_ids = list(self.patient_to_files.keys())
        logger.info("ImageDataModule mapped %d valid image files across %d patients.", file_count, len(self.patient_ids))

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

        full_dataset = ImageDataset(
            self.patient_to_files, self.patient_ids,
            transform=eval_transform, n_slices=self.n_slices, batch_key=self.batch_key,
        )
        self._full_dataset = full_dataset

        if len(full_dataset) == 0:
            self.train_dataset = self.val_dataset = self.test_dataset = full_dataset
            return

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
    def full_dataset(self) -> Optional[ImageDataset]:
        """Full dataset (all patients, no split) — available after setup()."""
        return self._full_dataset

    def setup_full(self, stage: Optional[str] = None) -> None:
        """Ensure setup() has been called so full_dataset is available."""
        if self._full_dataset is None:
            self.setup(stage=stage)

    def _filter_mr_preferred(
        self, patient_to_files: Dict[str, List[Path]]
    ) -> Dict[str, List[Path]]:
        """Keep only MR-modality files when MR series are present for a patient."""
        try:
            import pydicom
        except ImportError:
            return patient_to_files

        filtered: Dict[str, List[Path]] = {}
        for pid, files in patient_to_files.items():
            dir_to_files: Dict[Path, List[Path]] = {}
            for f in files:
                dir_to_files.setdefault(f.parent, []).append(f)

            dir_modality: Dict[Path, Optional[str]] = {}
            for series_dir, series_files in dir_to_files.items():
                try:
                    ds = pydicom.dcmread(str(series_files[0]), stop_before_pixels=True)
                    dir_modality[series_dir] = getattr(ds, "Modality", None)
                except Exception:
                    dir_modality[series_dir] = None

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
            logger.info("MR preference filter: removed MG files for %d patients.", mr_filtered)
        return filtered

    def _extract_patient_id(self, img_path: Path) -> str:
        """Extract TCGA patient ID from a TCIA-formatted path."""
        parts = img_path.parts
        for part in parts:
            if part.startswith('TCGA-'):
                tcga_parts = part.split('-')
                if len(tcga_parts) >= 3:
                    return '-'.join(tcga_parts[:3])
        return img_path.stem
