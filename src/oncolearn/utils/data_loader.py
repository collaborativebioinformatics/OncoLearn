"""
Data loading utilities for medical imaging datasets. Supports DICOM images from TCIA and other 
medical imaging formats.Integrates with genetic (gene expression) data for multimodal learning.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MedicalImageDataset(Dataset):
    """
    Dataset for medical imaging data with clinical labels and genetic data.
    Supports DICOM, PNG, JPEG, and other common image formats.
    Integrates gene expression data from processed TSV files.
    """

    def __init__(
        self,
        data_dir: str,
        clinical_file: Optional[str] = None,
        genetic_data_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[transforms.Compose] = None,
        extension: str = "*.png",
        label_column: str = "ajcc_pathologic_stage.diagnoses",
        use_genetic_data: bool = True,
        max_genes: Optional[int] = 1000,
        cancer_type: Optional[str] = None,
    ):
        """
        Initialize medical image dataset with genetic data integration.

        Args:
            data_dir: Directory containing images
            clinical_file: Path to clinical data TSV/CSV file (optional, will use genetic data file if not provided)
            genetic_data_dir: Directory containing merged genetic data TSV files (e.g., /data/processed)
            image_size: Target image size (H, W)
            transform: Optional torchvision transforms
            extension: File extension pattern (e.g., "*.png", "*.dcm")
            label_column: Column name for labels (e.g., 'ajcc_pathologic_stage.diagnoses')
            use_genetic_data: Whether to load and return genetic features
            max_genes: Maximum number of genes to use (uses top variance genes)
            cancer_type: Specific cancer type to load (e.g., 'BRCA', 'LUAD'). If None, loads all types.
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.label_column = label_column
        self.use_genetic_data = use_genetic_data
        self.max_genes = max_genes
        self.cancer_type = cancer_type

        # Load genetic data from processed directory
        self.genetic_data = None
        self.gene_features = None
        self.gene_columns = None

        if genetic_data_dir and os.path.exists(genetic_data_dir):
            self._load_genetic_data(genetic_data_dir, cancer_type)
            # Use genetic data file which already contains clinical info
            self.clinical_data = self.genetic_data
            print(f"Using genetic data file with {len(self.clinical_data)} samples")
        elif clinical_file and os.path.exists(clinical_file):
            self.clinical_data = pd.read_csv(clinical_file, sep='\t')
            print(f"Loaded clinical data: {len(self.clinical_data)} samples")

        # Create label mapping
        if self.clinical_data is not None and label_column in self.clinical_data.columns:
            unique_labels = self.clinical_data[label_column].dropna().unique()
            self.labels_map = {label: idx for idx,
                               label in enumerate(sorted(unique_labels))}
            print(
                f"Found {len(self.labels_map)} unique labels in '{label_column}':")
            for label, idx in sorted(self.labels_map.items(), key=lambda x: x[1]):
                count = (self.clinical_data[label_column] == label).sum()
                print(f"  {idx}: {label} ({count} samples)")

        # Find all images
        self.image_files = sorted(list(self.data_dir.rglob(extension)))
        print(f"Found {len(self.image_files)} images in {data_dir}")

        # Cross-reference images with genetic data
        if self.genetic_data is not None:
            self._cross_reference_samples()

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Cache for loaded images (optional, memory permitting)
        self.cache = {}

    def _load_genetic_data(self, genetic_data_dir: str, cancer_type: Optional[str] = None):
        """
        Load genetic data from processed directory.

        Args:
            genetic_data_dir: Directory containing merged TSV files
            cancer_type: Specific cancer type to load (e.g., 'BRCA'). If None, loads all.
        """
        genetic_dir = Path(genetic_data_dir)

        # Find all merged TSV files, optionally filtered by cancer type
        if cancer_type:
            pattern = f"TCGA-{cancer_type.upper()}_merged.tsv"
            tsv_files = list(genetic_dir.glob(pattern))
            if not tsv_files:
                print(
                    f"Warning: No file matching {pattern} found in {genetic_data_dir}")
                return
        else:
            tsv_files = list(genetic_dir.glob("TCGA-*_merged.tsv"))

        if not tsv_files:
            print(f"Warning: No merged TSV files found in {genetic_data_dir}")
            return

        print(f"\nLoading genetic data from {len(tsv_files)} file(s)...")

        # Load all TSV files and concatenate
        dfs = []
        for tsv_file in tsv_files:
            print(f"  Loading {tsv_file.name}...")
            df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
            dfs.append(df)

        # Concatenate all dataframes
        self.genetic_data = pd.concat(dfs, ignore_index=True)
        print(f"  Total samples loaded: {len(self.genetic_data)}")

        # Extract gene expression columns (starting with ENSG)
        gene_cols = [
            col for col in self.genetic_data.columns if col.startswith('ENSG')]
        print(f"  Found {len(gene_cols)} gene expression columns")

        if len(gene_cols) == 0:
            print("  Warning: No gene expression columns found")
            return

        # Extract gene features
        self.gene_features = self.genetic_data[['sample'] + gene_cols].copy()

        # Select top variance genes if max_genes is specified
        if self.max_genes and len(gene_cols) > self.max_genes:
            print(f"  Selecting top {self.max_genes} genes by variance...")

            # Calculate variance for each gene
            gene_data = self.gene_features[gene_cols].apply(
                pd.to_numeric, errors='coerce')
            variances = gene_data.var()
            top_genes = variances.nlargest(self.max_genes).index.tolist()

            # Keep only top variance genes
            self.gene_features = self.gene_features[['sample'] + top_genes]
            self.gene_columns = top_genes
            print(f"  Using {len(self.gene_columns)} genes")
        else:
            self.gene_columns = gene_cols

        # Convert gene expression to numeric
        for col in self.gene_columns:
            self.gene_features[col] = pd.to_numeric(
                self.gene_features[col], errors='coerce')

        # Fill NaN with 0
        self.gene_features[self.gene_columns] = self.gene_features[self.gene_columns].fillna(
            0)

        # Normalize gene expression (log2 transform + standardization)
        print("  Normalizing gene expression...")
        gene_matrix = self.gene_features[self.gene_columns].values
        # Log2(x + 1) transform
        gene_matrix = np.log2(gene_matrix + 1)
        # Standardize
        mean = gene_matrix.mean(axis=0)
        std = gene_matrix.std(axis=0) + 1e-8
        gene_matrix = (gene_matrix - mean) / std
        self.gene_features[self.gene_columns] = gene_matrix

        print("✓ Genetic data loaded successfully")

    def _cross_reference_samples(self):
        """
        Cross-reference image files with genetic data samples.
        Only keep images that have corresponding genetic data.
        """
        if self.genetic_data is None:
            return

        # Extract patient IDs from image paths
        image_patient_ids = [self._extract_patient_id(
            img_path) for img_path in self.image_files]

        # Get sample IDs from genetic data (extract patient part: TCGA-XX-XXXX)
        genetic_samples = self.genetic_data['sample'].values
        genetic_patient_ids = set()
        for sample in genetic_samples:
            # Extract patient ID (first 12 characters: TCGA-XX-XXXX)
            if pd.notna(sample) and sample.startswith('TCGA-'):
                parts = sample.split('-')
                if len(parts) >= 3:
                    patient_id = '-'.join(parts[:3])
                    genetic_patient_ids.add(patient_id)

        # Filter images to only those with genetic data
        filtered_images = []
        for img_path, patient_id in zip(self.image_files, image_patient_ids):
            if patient_id in genetic_patient_ids:
                filtered_images.append(img_path)

        original_count = len(self.image_files)
        self.image_files = filtered_images

        print("\nCross-referenced samples:")
        print(f"  Original images: {original_count}")
        print(f"  Images with genetic data: {len(self.image_files)}")
        print(f"  Samples dropped: {original_count - len(self.image_files)}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'image', 'genetic', 'label', and 'patient_id'
        """
        img_path = self.image_files[idx]

        # Load image
        if idx in self.cache:
            image = self.cache[idx]
        else:
            image = self._load_image(img_path)
            # Optionally cache (be mindful of memory)
            # self.cache[idx] = image

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get patient ID
        patient_id = self._extract_patient_id(img_path)

        # Get label
        label = self._get_label(patient_id)

        # Get genetic features
        genetic_features = self._get_genetic_features(patient_id)

        result = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'patient_id': patient_id,
            'image_path': str(img_path)
        }

        # Add genetic features if available
        if genetic_features is not None:
            result['genetic'] = torch.tensor(
                genetic_features, dtype=torch.float32)

        return result

    def _load_image(self, img_path: Path) -> Image.Image:
        """
        Load an image from various formats.

        Args:
            img_path: Path to image file

        Returns:
            PIL Image
        """
        # Handle DICOM files
        if img_path.suffix.lower() in ['.dcm', '.dicom']:
            try:
                import pydicom
                import SimpleITK as sitk

                # Try pydicom first
                try:
                    dicom = pydicom.dcmread(str(img_path))
                    pixel_array = dicom.pixel_array

                    # Normalize to 0-255
                    pixel_array = pixel_array.astype(np.float32)
                    pixel_array = (pixel_array - pixel_array.min()) / \
                        (pixel_array.max() - pixel_array.min() + 1e-8)
                    pixel_array = (pixel_array * 255).astype(np.uint8)

                    # Convert to RGB if grayscale
                    if len(pixel_array.shape) == 2:
                        pixel_array = np.stack([pixel_array] * 3, axis=-1)

                    return Image.fromarray(pixel_array)

                except Exception:
                    # Fallback to SimpleITK
                    image = sitk.ReadImage(str(img_path))
                    array = sitk.GetArrayFromImage(image)

                    # Normalize
                    array = array.astype(np.float32)
                    array = (array - array.min()) / \
                        (array.max() - array.min() + 1e-8)
                    array = (array * 255).astype(np.uint8)

                    # Convert to RGB
                    if len(array.shape) == 2:
                        array = np.stack([array] * 3, axis=-1)
                    elif len(array.shape) == 3 and array.shape[0] == 1:
                        array = np.stack([array[0]] * 3, axis=-1)

                    return Image.fromarray(array)

            except ImportError:
                raise ImportError(
                    "pydicom and SimpleITK required for DICOM files")

        # Handle standard image formats
        else:
            image = Image.open(img_path).convert('RGB')
            return image

    def _extract_patient_id(self, img_path: Path) -> str:
        """
        Extract patient ID from image path.
        Override this method for custom ID extraction logic.

        Args:
            img_path: Path to image

        Returns:
            Patient ID string
        """
        # Default: use parent directory name or file stem
        # Common TCGA pattern: TCGA-XX-XXXX-...
        parts = img_path.parts
        for part in parts:
            if part.startswith('TCGA-'):
                # Extract patient identifier (first 3 parts)
                tcga_parts = part.split('-')
                if len(tcga_parts) >= 3:
                    return '-'.join(tcga_parts[:3])

        # Fallback to file stem
        return img_path.stem

    def _get_label(self, patient_id: str) -> int:
        """
        Get label for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            Label index (int)
        """
        if self.clinical_data is None or self.label_column not in self.clinical_data.columns:
            # Return dummy label if no clinical data
            return 0

        # Find patient in clinical data by matching sample IDs
        # Genetic data uses sample IDs like TCGA-XX-XXXX-01A
        # We match on patient ID (TCGA-XX-XXXX)
        patient_data = self.clinical_data[
            self.clinical_data['sample'].str.contains(patient_id, na=False) |
            self.clinical_data.get('submitter_id', pd.Series()).str.contains(
                patient_id, na=False)
        ]

        if len(patient_data) == 0:
            # Patient not found, return default label
            return 0

        # Get label and map to index
        label_value = patient_data.iloc[0][self.label_column]
        if pd.isna(label_value):
            return 0

        return self.labels_map.get(label_value, 0)

    def _get_genetic_features(self, patient_id: str) -> Optional[np.ndarray]:
        """
        Get genetic features for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            Numpy array of gene expression values or None
        """
        if not self.use_genetic_data or self.gene_features is None:
            return None

        # Find matching sample in genetic data
        patient_samples = self.gene_features[
            self.gene_features['sample'].str.contains(patient_id, na=False)
        ]

        if len(patient_samples) == 0:
            # No genetic data for this patient, return zeros
            return np.zeros(len(self.gene_columns), dtype=np.float32)

        # Get first matching sample
        sample_data = patient_samples.iloc[0]

        # Extract gene expression values
        genetic_values = sample_data[self.gene_columns].values.astype(
            np.float32)

        return genetic_values

    def get_num_classes(self) -> int:
        """Return number of unique classes."""
        return len(self.labels_map) if self.labels_map else 1

    def get_num_genes(self) -> int:
        """Return number of gene features."""
        if self.gene_columns is None:
            return 0
        return len(self.gene_columns)


def create_data_loaders(
    data_dir: str,
    clinical_file: Optional[str] = None,
    genetic_data_dir: Optional[str] = "/workspace/data/processed",
    batch_size: int = 16,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    image_size: Tuple[int, int] = (512, 512),
    seed: int = 42,
    label_column: str = "ajcc_pathologic_stage.diagnoses",
    use_genetic_data: bool = True,
    max_genes: Optional[int] = 1000,
    cancer_type: Optional[str] = None,
    extension: str = "*.png",
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train, validation, and test data loaders with genetic data integration.

    Args:
        data_dir: Directory containing images
        clinical_file: Path to clinical data file (optional if using genetic data)
        genetic_data_dir: Directory containing merged genetic data TSV files
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        image_size: Target image size
        seed: Random seed for reproducibility
        label_column: Column name for labels (e.g., 'ajcc_pathologic_stage.diagnoses')
        use_genetic_data: Whether to load and return genetic features
        max_genes: Maximum number of genes to use (selects top variance genes)
        cancer_type: Specific cancer type to load (e.g., 'BRCA', 'LUAD'). If None, loads all types.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes)
    """
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Validation/test transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create full dataset
    full_dataset = MedicalImageDataset(
        data_dir=data_dir,
        clinical_file=clinical_file,
        genetic_data_dir=genetic_data_dir,
        image_size=image_size,
        transform=None,  # Will apply transforms per split
        label_column=label_column,
        use_genetic_data=use_genetic_data,
        max_genes=max_genes,
        cancer_type=cancer_type,
        extension=extension,
    )

    num_classes = full_dataset.get_num_classes()

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Wrap with transform
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = eval_transform
    test_dataset.dataset.transform = eval_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("\nDataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Total: {total_size} samples")
    print(f"  Num classes: {num_classes}")

    # Print genetic data info if available
    num_genes = full_dataset.get_num_genes()
    if num_genes > 0:
        print(f"  Num genes: {num_genes}")
    print()

    return train_loader, val_loader, test_loader, num_classes
