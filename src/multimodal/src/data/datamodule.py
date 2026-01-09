"""
PyTorch Dataset and DataLoader for V1 and V2 variants.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from .cohort import (
    build_cohorts,
    get_cohort_for_variant,
    load_clinical_table,
    load_gene_set_table,
)
from .dicom_io import (
    sample_dicom_series,
    sample_dicom_series_uniform,
    get_all_series_from_patient
)
from .labels import LabelManager
from .transforms import get_dicom_transforms

logger = logging.getLogger(__name__)


class TCGAV1Dataset(Dataset):
    """Dataset for V1 (imaging-present) variant with sequence-level expansion."""
    
    def __init__(
        self,
        cohort_df: pd.DataFrame,
        clinical_table: pd.DataFrame,
        gene_set_table: pd.DataFrame,
        label_manager: LabelManager,
        dicom_root: str,
        dicom_transform=None,
        n_dicom_samples: int = 5,  # Changed default to 5 for uniform sampling
        modality_dropout: float = 0.3,
        mode: str = 'train',
        expand_by_sequences: bool = True  # New: expand dataset by sequences
    ):
        self.cohort_df = cohort_df.reset_index(drop=True)
        self.clinical_table = clinical_table
        self.gene_set_table = gene_set_table
        self.label_manager = label_manager
        self.dicom_root = Path(dicom_root)
        self.dicom_transform = dicom_transform or get_dicom_transforms(size=224, augment=(mode == 'train'))
        self.n_dicom_samples = n_dicom_samples
        self.modality_dropout = modality_dropout
        self.mode = mode
        self.expand_by_sequences = expand_by_sequences
        
        # Get patient IDs
        self.patient_ids = self.cohort_df['patient_id'].tolist()
        
        # Get labels for patients
        self.patient_stage_labels = self.label_manager.get_stage_labels(self.patient_ids)
        self.patient_subtype_labels = self.label_manager.get_subtype_labels(self.patient_ids)
        
        # Build sequence-level index if expanding by sequences
        if self.expand_by_sequences:
            self.sequence_index = []  # List of (patient_idx, series_info)
            
            for patient_idx, patient_id in enumerate(self.patient_ids):
                row = self.cohort_df[self.cohort_df['patient_id'] == patient_id].iloc[0]
                dicom_series_json = row.get('dicom_series', '{}')
                
                if dicom_series_json:
                    # Get all series for this patient
                    series_list = get_all_series_from_patient(
                        dicom_series_json,
                        str(self.dicom_root),
                        modality='MR'
                    )
                    
                    # Add each series as a separate sample
                    for series_info in series_list:
                        if series_info['n_images'] >= self.n_dicom_samples:  # Only use series with enough images
                            self.sequence_index.append((patient_idx, series_info))
            
            logger.info(f"V1 Dataset: {len(self.patient_ids)} patients, {len(self.sequence_index)} sequences")
        else:
            # Original behavior: one sample per patient
            self.sequence_index = [(i, None) for i in range(len(self.patient_ids))]
            logger.info(f"V1 Dataset: {len(self.patient_ids)} patients (no sequence expansion)")
    
    def __len__(self) -> int:
        return len(self.sequence_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get patient index and series info
        patient_idx, series_info = self.sequence_index[idx]
        patient_id = self.patient_ids[patient_idx]
        row = self.cohort_df[self.cohort_df['patient_id'] == patient_id].iloc[0]
        
        # Gene features (same for all sequences from same patient)
        gene_features = None
        if patient_id in self.gene_set_table.index:
            gene_vec = self.gene_set_table.loc[patient_id].values.astype(np.float32)
            gene_features = torch.from_numpy(gene_vec)
        else:
            raise ValueError(f"Patient {patient_id} not in gene_set_table")
        
        # Clinical features (same for all sequences from same patient)
        clinical_features = None
        if patient_id in self.clinical_table.index:
            # Select numeric columns only
            clin_row = self.clinical_table.loc[patient_id]
            # If Series, convert to numeric and filter
            if isinstance(clin_row, pd.Series):
                numeric_cols = pd.to_numeric(clin_row, errors='coerce').fillna(0).values.astype(np.float32)
            else:
                # DataFrame case (shouldn't happen but handle it)
                numeric_cols = clin_row.select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
            clinical_features = torch.from_numpy(numeric_cols)
        else:
            raise ValueError(f"Patient {patient_id} not in clinical_table")
        
        # Image features (DICOM) - sequence-specific
        image_features = None
        modality_ids = None
        
        # Modality dropout (training only)
        use_image = True
        if self.mode == 'train' and np.random.random() < self.modality_dropout:
            use_image = False
        
        if use_image and series_info is not None:
            # Uniform sampling from this specific series
            series_paths = series_info['paths']
            n_total = len(series_paths)
            
            if n_total >= self.n_dicom_samples:
                # Uniform sampling: sample every n_total/n_samples-th image
                step = n_total / self.n_dicom_samples
                sampled_indices = [int(i * step) for i in range(self.n_dicom_samples)]
                dicom_paths = [series_paths[idx] for idx in sampled_indices]
            else:
                # Not enough images, use all available
                dicom_paths = series_paths
            
            if dicom_paths:
                # Load and transform images
                from .dicom_io import read_dicom_image
                images_list = []
                modality_ids_list = []
                
                for path in dicom_paths:
                    try:
                        pixel_array = read_dicom_image(path)
                        if pixel_array is not None:
                            img_tensor = self.dicom_transform(pixel_array)  # (3, H, W)
                            
                            # Validate shape: should be (3, H, W)
                            if img_tensor.dim() != 3:
                                logger.warning(f"Invalid image tensor dim: {img_tensor.dim()}, expected 3. Shape: {img_tensor.shape}")
                                continue
                            if img_tensor.shape[0] != 3:
                                logger.warning(f"Invalid image channels: {img_tensor.shape[0]}, expected 3. Shape: {img_tensor.shape}")
                                # Try to fix: if single channel, repeat to 3
                                if img_tensor.shape[0] == 1:
                                    img_tensor = img_tensor.repeat(3, 1, 1)
                                else:
                                    logger.error(f"Cannot fix image with {img_tensor.shape[0]} channels. Skipping.")
                                    continue
                            
                            images_list.append(img_tensor)
                            # Determine modality: MR=0, MG=1
                            modality = series_info.get('modality', 'MR')
                            modality_id = 0 if modality == 'MR' else 1
                            modality_ids_list.append(modality_id)
                    except Exception as e:
                        logger.warning(f"Failed to load/transform image from {path}: {e}")
                        continue
                
                if images_list:
                    # Validate all images have same shape before stacking
                    shapes = [img.shape for img in images_list]
                    if len(set(shapes)) > 1:
                        logger.warning(f"Inconsistent image shapes in batch: {shapes}")
                        # Use the most common shape
                        from collections import Counter
                        shape_counts = Counter(shapes)
                        target_shape = shape_counts.most_common(1)[0][0]
                        # Filter or fix images
                        fixed_images = []
                        for img in images_list:
                            if img.shape == target_shape:
                                fixed_images.append(img)
                            else:
                                logger.warning(f"Skipping image with shape {img.shape}, expected {target_shape}")
                        images_list = fixed_images
                    
                    if images_list:
                        # Stack into (N, 3, H, W) where N is number of images
                        image_features = torch.stack(images_list)  # (N, 3, H, W)
                        modality_ids = torch.tensor(modality_ids_list[:len(images_list)], dtype=torch.long)
        
        # Labels (same for all sequences from same patient)
        stage_label = torch.tensor(self.patient_stage_labels[patient_idx], dtype=torch.long)
        subtype_label = None
        if self.patient_subtype_labels is not None:
            subtype_label = torch.tensor(self.patient_subtype_labels[patient_idx], dtype=torch.long)
        
        # Modality dropout for gene/clinical
        use_gene = True
        use_clinical = True
        if self.mode == 'train':
            if np.random.random() < self.modality_dropout:
                use_gene = False
            if np.random.random() < self.modality_dropout:
                use_clinical = False
            
            # Ensure at least one modality
            if not use_gene and not use_clinical and not use_image:
                use_gene = True
        
        result = {
            'patient_id': patient_id,
            'gene': gene_features if use_gene else None,
            'clinical': clinical_features if use_clinical else None,
            'image': image_features if (use_image and image_features is not None) else None,
            'stage_label': stage_label,
            'subtype_label': subtype_label,
        }
        
        # Store modality IDs if images are present
        if use_image and image_features is not None:
            if 'modality_ids' in locals() and modality_ids is not None:
                result['modality_ids'] = modality_ids
            else:
                # Default: all MR (0) - should not happen if code above is correct
                result['modality_ids'] = None
        
        return result


class TCGAV2Dataset(Dataset):
    """Dataset for V2 (no-imaging) variant."""
    
    def __init__(
        self,
        cohort_df: pd.DataFrame,
        clinical_table: pd.DataFrame,
        gene_set_table: pd.DataFrame,
        label_manager: LabelManager,
        modality_dropout: float = 0.3,
        mode: str = 'train'
    ):
        self.cohort_df = cohort_df.reset_index(drop=True)
        self.clinical_table = clinical_table
        self.gene_set_table = gene_set_table
        self.label_manager = label_manager
        self.modality_dropout = modality_dropout
        self.mode = mode
        
        # Get patient IDs
        self.patient_ids = self.cohort_df['patient_id'].tolist()
        
        # Get labels
        self.stage_labels = self.label_manager.get_stage_labels(self.patient_ids)
        self.subtype_labels = self.label_manager.get_subtype_labels(self.patient_ids)
        
        logger.info(f"V2 Dataset: {len(self.patient_ids)} patients")
    
    def __len__(self) -> int:
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patient_id = self.patient_ids[idx]
        
        # Gene features
        gene_features = None
        if patient_id in self.gene_set_table.index:
            gene_vec = self.gene_set_table.loc[patient_id].values.astype(np.float32)
            gene_features = torch.from_numpy(gene_vec)
        else:
            raise ValueError(f"Patient {patient_id} not in gene_set_table")
        
        # Clinical features
        clinical_features = None
        if patient_id in self.clinical_table.index:
            clin_row = self.clinical_table.loc[patient_id]
            numeric_cols = clin_row.select_dtypes(include=[np.number])
            numeric_cols = numeric_cols.fillna(0).values.astype(np.float32)
            clinical_features = torch.from_numpy(numeric_cols)
        else:
            raise ValueError(f"Patient {patient_id} not in clinical_table")
        
        # Labels
        stage_label = torch.tensor(self.stage_labels[idx], dtype=torch.long)
        subtype_label = None
        if self.subtype_labels is not None:
            subtype_label = torch.tensor(self.subtype_labels[idx], dtype=torch.long)
        
        # Modality dropout (training only)
        use_gene = True
        use_clinical = True
        if self.mode == 'train':
            if np.random.random() < self.modality_dropout:
                use_gene = False
            if np.random.random() < self.modality_dropout:
                use_clinical = False
            
            # Ensure at least one modality
            if not use_gene and not use_clinical:
                use_gene = True
        
        return {
            'patient_id': patient_id,
            'gene': gene_features if use_gene else None,
            'clinical': clinical_features if use_clinical else None,
            'image': None,  # No image for V2
            'stage_label': stage_label,
            'subtype_label': subtype_label,
        }


def collate_fn_v1(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for V1 (with images)."""
    patient_ids = [item['patient_id'] for item in batch]
    batch_size = len(batch)
    
    # Gene features - handle None values with placeholders
    gene_batch = []
    for item in batch:
        if item['gene'] is not None:
            gene_batch.append(item['gene'])
        else:
            # Create placeholder with same shape as other genes
            # Find a non-None gene to get shape
            for other_item in batch:
                if other_item['gene'] is not None:
                    placeholder = torch.zeros_like(other_item['gene'])
                    gene_batch.append(placeholder)
                    break
            else:
                # No genes in batch, skip
                pass
    
    if gene_batch:
        gene_tensor = torch.stack(gene_batch)
        # Ensure batch size matches
        if gene_tensor.shape[0] != batch_size:
            # Pad or trim to match batch size
            if gene_tensor.shape[0] < batch_size:
                pad = torch.zeros(batch_size - gene_tensor.shape[0], *gene_tensor.shape[1:], dtype=gene_tensor.dtype)
                gene_tensor = torch.cat([gene_tensor, pad], dim=0)
            else:
                gene_tensor = gene_tensor[:batch_size]
    else:
        gene_tensor = None
    
    # Clinical features - handle None values with placeholders
    clinical_batch = []
    for item in batch:
        if item['clinical'] is not None:
            clinical_batch.append(item['clinical'])
        else:
            # Create placeholder
            for other_item in batch:
                if other_item['clinical'] is not None:
                    placeholder = torch.zeros_like(other_item['clinical'])
                    clinical_batch.append(placeholder)
                    break
    
    if clinical_batch:
        # Pad to same length
        max_len = max(c.shape[0] for c in clinical_batch)
        padded_clinical = []
        for c in clinical_batch:
            if c.shape[0] < max_len:
                pad = torch.zeros(max_len - c.shape[0], dtype=c.dtype)
                c = torch.cat([c, pad])
            padded_clinical.append(c)
        clinical_tensor = torch.stack(padded_clinical)
        # Ensure batch size matches
        if clinical_tensor.shape[0] != batch_size:
            if clinical_tensor.shape[0] < batch_size:
                pad = torch.zeros(batch_size - clinical_tensor.shape[0], *clinical_tensor.shape[1:], dtype=clinical_tensor.dtype)
                clinical_tensor = torch.cat([clinical_tensor, pad], dim=0)
            else:
                clinical_tensor = clinical_tensor[:batch_size]
    else:
        clinical_tensor = None
    
    # Image features (handle variable number of images per patient)
    # Create placeholders for None images to maintain batch size
    image_batch = []
    modality_ids_batch = []
    
    # Find a reference image to get shape
    reference_image = None
    reference_mod_ids = None
    for item in batch:
        if item['image'] is not None:
            reference_image = item['image']
            reference_mod_ids = item.get('modality_ids')
            if reference_mod_ids is None:
                N = reference_image.shape[0]
                reference_mod_ids = torch.zeros(N, dtype=torch.long)
            break
    
    for item in batch:
        if item['image'] is not None:
            # item['image'] is (N, 3, H, W) where N can vary
            image_batch.append(item['image'])
            mod_ids = item.get('modality_ids')
            if mod_ids is not None:
                modality_ids_batch.append(mod_ids)
            else:
                # Default: all MR (0)
                N = item['image'].shape[0]
                modality_ids_batch.append(torch.zeros(N, dtype=torch.long))
        else:
            # Create placeholder image
            if reference_image is not None:
                # Use same number of images as reference
                N_ref = reference_image.shape[0]
                placeholder = torch.zeros(N_ref, 3, 224, 224, dtype=torch.float32)
                image_batch.append(placeholder)
                placeholder_mod_ids = torch.zeros(N_ref, dtype=torch.long)
                modality_ids_batch.append(placeholder_mod_ids)
            else:
                # No reference, skip (shouldn't happen if at least one image exists)
                pass
    
    if image_batch:
        # Filter out invalid images (e.g., feature maps with wrong channel count)
        valid_image_batch = []
        valid_modality_ids_batch = []
        expected_C = 3  # Expected channels for RGB
        expected_H = 224  # Expected height
        expected_W = 224  # Expected width
        
        for img, mod_ids in zip(image_batch, modality_ids_batch):
            N, C, H, W = img.shape
            # Validate: should be (N, 3, 224, 224)
            if C == expected_C and H == expected_H and W == expected_W:
                valid_image_batch.append(img)
                valid_modality_ids_batch.append(mod_ids)
            else:
                logger.warning(f"Invalid image shape in batch: {img.shape}, expected (N, {expected_C}, {expected_H}, {expected_W}). Skipping.")
        
        if valid_image_batch:
            # Pad to same number of images per patient
            max_n_images = max(img.shape[0] for img in valid_image_batch)
            padded_images = []
            padded_modality_ids = []
            for img, mod_ids in zip(valid_image_batch, valid_modality_ids_batch):
                N, C, H, W = img.shape
                if N < max_n_images:
                    # Pad with zeros
                    pad = torch.zeros(max_n_images - N, C, H, W, dtype=img.dtype, device=img.device)
                    img = torch.cat([img, pad], dim=0)
                    mod_pad = torch.zeros(max_n_images - N, dtype=mod_ids.dtype, device=mod_ids.device)
                    mod_ids = torch.cat([mod_ids, mod_pad], dim=0)
                padded_images.append(img)
                padded_modality_ids.append(mod_ids)
            
            image_tensor = torch.stack(padded_images)  # (B, max_N, C, H, W)
            modality_ids_tensor = torch.stack(padded_modality_ids)  # (B, max_N)
        else:
            logger.warning("No valid images in batch after filtering. Setting to None.")
            image_tensor = None
            modality_ids_tensor = None
    else:
        image_tensor = None
        modality_ids_tensor = None
    
    # Labels
    stage_labels = torch.stack([item['stage_label'] for item in batch])
    subtype_labels = None
    if batch[0]['subtype_label'] is not None:
        subtype_labels = torch.stack([item['subtype_label'] for item in batch])
    
    result = {
        'patient_ids': patient_ids,
        'gene': gene_tensor,
        'clinical': clinical_tensor,
        'image': image_tensor,
        'stage_label': stage_labels,
        'subtype_label': subtype_labels,
    }
    
    if modality_ids_tensor is not None:
        result['modality_ids'] = modality_ids_tensor
    
    return result


def collate_fn_v2(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for V2 (no images)."""
    patient_ids = [item['patient_id'] for item in batch]
    
    # Gene features
    gene_batch = [item['gene'] for item in batch if item['gene'] is not None]
    if gene_batch:
        gene_tensor = torch.stack(gene_batch)
    else:
        gene_tensor = None
    
    # Clinical features
    clinical_batch = [item['clinical'] for item in batch if item['clinical'] is not None]
    if clinical_batch:
        max_len = max(c.shape[0] for c in clinical_batch)
        padded_clinical = []
        for c in clinical_batch:
            if c.shape[0] < max_len:
                pad = torch.zeros(max_len - c.shape[0], dtype=c.dtype)
                c = torch.cat([c, pad])
            padded_clinical.append(c)
        clinical_tensor = torch.stack(padded_clinical)
    else:
        clinical_tensor = None
    
    # Labels
    stage_labels = torch.stack([item['stage_label'] for item in batch])
    subtype_labels = None
    if batch[0]['subtype_label'] is not None:
        subtype_labels = torch.stack([item['subtype_label'] for item in batch])
    
    return {
        'patient_ids': patient_ids,
        'gene': gene_tensor,
        'clinical': clinical_tensor,
        'image': None,
        'stage_label': stage_labels,
        'subtype_label': subtype_labels,
    }


class TCGADataModule:
    """Data module for TCGA-BRCA training."""
    
    def __init__(
        self,
        cohort_index_path: str,
        clinical_table_path: str,
        gene_set_path: str,
        dicom_root: Optional[str] = None,
        variant: str = 'v1_imaging',
        n_folds: int = 5,
        fold: int = 0,
        batch_size: int = 32,
        num_workers: int = 4,
        n_dicom_samples: int = 5,  # Changed default to 5 for uniform sampling
        modality_dropout: float = 0.3,
        seed: int = 42,
        pam50_file: Optional[str] = None,
        brca_labels_file: Optional[str] = None,
        expand_by_sequences: bool = True,  # New: expand dataset by sequences
        test_patients_path: Optional[str] = None  # Path to test_patients.csv (for federated splits)
    ):
        self.cohort_index_path = cohort_index_path
        self.clinical_table_path = clinical_table_path
        self.gene_set_path = gene_set_path
        self.dicom_root = dicom_root
        self.variant = variant
        self.n_folds = n_folds
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_dicom_samples = n_dicom_samples
        self.modality_dropout = modality_dropout
        self.seed = seed
        self.pam50_file = pam50_file
        self.brca_labels_file = brca_labels_file
        self.expand_by_sequences = expand_by_sequences
        self.test_patients_path = test_patients_path
        
        # Load data
        self._load_data()
        
        # Create splits
        self._create_splits()
    
    def _load_data(self):
        """Load all data tables."""
        from .cohort import load_cohort_index, load_clinical_table, load_gene_set_table, is_imaging_present
        
        self.cohort_index = load_cohort_index(self.cohort_index_path)
        self.clinical_table = load_clinical_table(self.clinical_table_path)
        self.gene_set_table = load_gene_set_table(self.gene_set_path)
        
        # Determine imaging status
        self.cohort_index['imaging_present'] = self.cohort_index.apply(
            lambda row: is_imaging_present(row, self.dicom_root), axis=1
        )
        
        # Get cohort for variant
        self.cohort_df = get_cohort_for_variant(self.cohort_index, self.variant)
        
        # Create label manager with PAM50 and BRCA labels support
        self.label_manager = LabelManager(
            self.clinical_table,
            pam50_file=self.pam50_file,
            brca_labels_file=self.brca_labels_file
        )
    
    def _create_splits(self):
        """Create train/val/test splits.
        
        If test_patients_path is provided, load test split from file (for federated splits).
        Otherwise, use stratified K-fold for train/val only.
        """
        patient_ids = self.cohort_df['patient_id'].tolist()
        stage_labels = self.label_manager.get_stage_labels(patient_ids)
        
        # If test_patients_path is provided, load test split from file
        if self.test_patients_path and Path(self.test_patients_path).exists():
            logger.info(f"Loading test split from: {self.test_patients_path}")
            test_df = pd.read_csv(self.test_patients_path)
            test_patient_ids = set(test_df['patient_id'].tolist())
            
            # Split remaining patients into train/val
            remaining_patient_ids = [pid for pid in patient_ids if pid not in test_patient_ids]
            remaining_stage_labels = [stage_labels[i] for i, pid in enumerate(patient_ids) if pid not in test_patient_ids]
            
            if len(remaining_patient_ids) > 0:
                # Use stratified split for train/val
                from sklearn.model_selection import train_test_split
                # Check if stratification is possible (need at least 2 samples per class)
                unique_labels = set(remaining_stage_labels)
                can_stratify = len(unique_labels) > 1
                if can_stratify:
                    # Check if each class has at least 2 samples
                    from collections import Counter
                    label_counts = Counter(remaining_stage_labels)
                    can_stratify = all(count >= 2 for count in label_counts.values())
                
                train_ids, val_ids = train_test_split(
                    remaining_patient_ids,
                    test_size=1.0 / self.n_folds,
                    random_state=self.seed + self.fold,
                    stratify=remaining_stage_labels if can_stratify else None
                )
            else:
                train_ids = []
                val_ids = []
            
            self.train_patient_ids = train_ids
            self.val_patient_ids = val_ids
            self.test_patient_ids = list(test_patient_ids)
            
            logger.info(f"Loaded splits from file: Train={len(self.train_patient_ids)}, Val={len(self.val_patient_ids)}, Test={len(self.test_patient_ids)}")
        else:
            # Standard K-fold split (train/val only)
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            splits = list(skf.split(patient_ids, stage_labels))
            
            train_idx, val_idx = splits[self.fold]
            
            self.train_patient_ids = [patient_ids[i] for i in train_idx]
            self.val_patient_ids = [patient_ids[i] for i in val_idx]
            self.test_patient_ids = []  # No test split in standard K-fold
            
            logger.info(f"Fold {self.fold}: Train={len(self.train_patient_ids)}, Val={len(self.val_patient_ids)}")
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        train_cohort = self.cohort_df[self.cohort_df['patient_id'].isin(self.train_patient_ids)]
        
        if self.variant == 'v1_imaging':
            dataset = TCGAV1Dataset(
                train_cohort,
                self.clinical_table,
                self.gene_set_table,
                self.label_manager,
                self.dicom_root,
                n_dicom_samples=self.n_dicom_samples,
                modality_dropout=self.modality_dropout,
                mode='train',
                expand_by_sequences=self.expand_by_sequences
            )
            collate_fn = collate_fn_v1
        else:
            dataset = TCGAV2Dataset(
                train_cohort,
                self.clinical_table,
                self.gene_set_table,
                self.label_manager,
                modality_dropout=self.modality_dropout,
                mode='train'
            )
            collate_fn = collate_fn_v2
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        val_cohort = self.cohort_df[self.cohort_df['patient_id'].isin(self.val_patient_ids)]
        
        if self.variant == 'v1_imaging':
            dataset = TCGAV1Dataset(
                val_cohort,
                self.clinical_table,
                self.gene_set_table,
                self.label_manager,
                self.dicom_root,
                n_dicom_samples=self.n_dicom_samples,
                modality_dropout=0.0,  # No dropout in val
                mode='val',
                expand_by_sequences=self.expand_by_sequences
            )
            collate_fn = collate_fn_v1
        else:
            dataset = TCGAV2Dataset(
                val_cohort,
                self.clinical_table,
                self.gene_set_table,
                self.label_manager,
                modality_dropout=0.0,  # No dropout in val
                mode='val'
            )
            collate_fn = collate_fn_v2
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test DataLoader (if test split exists)."""
        if not hasattr(self, 'test_patient_ids') or len(self.test_patient_ids) == 0:
            logger.warning("No test split available. Returning None.")
            return None
        
        test_cohort = self.cohort_df[self.cohort_df['patient_id'].isin(self.test_patient_ids)]
        
        if self.variant == 'v1_imaging':
            dataset = TCGAV1Dataset(
                test_cohort,
                self.clinical_table,
                self.gene_set_table,
                self.label_manager,
                self.dicom_root,
                n_dicom_samples=self.n_dicom_samples,
                modality_dropout=0.0,  # No dropout in test
                mode='test',
                expand_by_sequences=self.expand_by_sequences
            )
            collate_fn = collate_fn_v1
        else:
            dataset = TCGAV2Dataset(
                test_cohort,
                self.clinical_table,
                self.gene_set_table,
                self.label_manager,
                modality_dropout=0.0,  # No dropout in test
                mode='test'
            )
            collate_fn = collate_fn_v2
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

