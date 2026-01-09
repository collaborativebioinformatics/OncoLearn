"""
Label management for stage and optional subtype classification.
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_patient_id(sample_barcode: str) -> str:
    """TCGA 샘플 바코드에서 Patient ID 추출 (첫 12자)"""
    parts = sample_barcode.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return sample_barcode


class LabelManager:
    """Manages stage and subtype labels with discovery and derivation logic."""
    
    def __init__(
        self,
        clinical_table: pd.DataFrame,
        subtype_lambda: float = 0.3,
        pam50_file: Optional[str] = None,
        brca_labels_file: Optional[str] = None
    ):
        self.clinical_table = clinical_table
        self.subtype_lambda = subtype_lambda
        self.pam50_file = pam50_file
        self.brca_labels_file = brca_labels_file
        
        # Stage labels
        self.stage_classes = ['Stage I', 'Stage II', 'Stage III', 'Stage IV', 'Unknown']
        self.stage_to_idx = {s: i for i, s in enumerate(self.stage_classes)}
        self.idx_to_stage = {i: s for s, i in self.stage_to_idx.items()}
        
        # Subtype labels (discovered/derived)
        self.subtype_classes = None
        self.subtype_to_idx = None
        self.idx_to_subtype = None
        self.has_subtype = False
        self.pam50_mapping = None  # patient_id -> PAM50 subtype
        self.brca_labels_mapping = None  # patient_id -> subtype (from BRCA-data file)
        
        # Load BRCA labels file if provided (highest priority)
        if brca_labels_file:
            self._load_brca_labels()
        
        # Load PAM50 if provided
        if pam50_file:
            self._load_pam50()
        
        self._discover_stage_column()
        self._discover_or_derive_subtype()
    
    def _discover_stage_column(self) -> Optional[str]:
        """Discover stage column in clinical table."""
        stage_keywords = [
            'ajcc_pathologic_stage',
            'pathologic_stage',
            'stage',
            'clinical_stage'
        ]
        
        for keyword in stage_keywords:
            for col in self.clinical_table.columns:
                if keyword.lower() in col.lower():
                    logger.info(f"Found stage column: {col}")
                    return col
        
        logger.warning("No stage column found in clinical table")
        return None
    
    def _normalize_stage(self, stage_val: str) -> str:
        """Normalize stage value to standard classes."""
        if pd.isna(stage_val) or stage_val == '':
            return 'Unknown'
        
        stage_str = str(stage_val).strip()
        
        # Pattern matching
        if re.search(r'stage\s*[i1]', stage_str, re.IGNORECASE):
            return 'Stage I'
        elif re.search(r'stage\s*[i1]{2,2}', stage_str, re.IGNORECASE) or re.search(r'stage\s*2', stage_str, re.IGNORECASE):
            return 'Stage II'
        elif re.search(r'stage\s*[i1]{3,3}', stage_str, re.IGNORECASE) or re.search(r'stage\s*3', stage_str, re.IGNORECASE):
            return 'Stage III'
        elif re.search(r'stage\s*[i1]{4,4}', stage_str, re.IGNORECASE) or re.search(r'stage\s*4', stage_str, re.IGNORECASE):
            return 'Stage IV'
        else:
            return 'Unknown'
    
    def _discover_subtype_column(self) -> Optional[str]:
        """Discover subtype column by keywords."""
        subtype_keywords = ['pam50', 'subtype', 'molecular', 'intrinsic']
        
        for keyword in subtype_keywords:
            for col in self.clinical_table.columns:
                if keyword.lower() in col.lower():
                    # Validate column
                    unique_vals = self.clinical_table[col].dropna().unique()
                    n_classes = len(unique_vals)
                    
                    # Check validity: 2-6 classes, no single class >90%
                    if 2 <= n_classes <= 6:
                        max_freq = (self.clinical_table[col].value_counts().max() / len(self.clinical_table)) * 100
                        if max_freq < 90:
                            logger.info(f"Found subtype column: {col} ({n_classes} classes)")
                            return col
        
        return None
    
    def _derive_subtype_from_hr_her2(self) -> Optional[pd.Series]:
        """Derive subtype from ER/PR/HER2 status."""
        # Find ER, PR, HER2 columns
        er_col = None
        pr_col = None
        her2_col = None
        
        for col in self.clinical_table.columns:
            col_lower = col.lower()
            if 'er_status' in col_lower or 'estrogen_receptor' in col_lower:
                er_col = col
            elif 'pr_status' in col_lower or 'progesterone_receptor' in col_lower:
                pr_col = col
            elif 'her2_status' in col_lower or 'her2' in col_lower:
                her2_col = col
        
        if not (er_col and pr_col and her2_col):
            logger.warning("Could not find ER/PR/HER2 columns for subtype derivation")
            return None
        
        # Derive HR and HER2 status
        def parse_status(val):
            if pd.isna(val):
                return None
            val_str = str(val).lower()
            if 'positive' in val_str or '+' in val_str:
                return True
            elif 'negative' in val_str or '-' in val_str:
                return False
            return None
        
        er_status = self.clinical_table[er_col].apply(parse_status)
        pr_status = self.clinical_table[pr_col].apply(parse_status)
        her2_status = self.clinical_table[her2_col].apply(parse_status)
        
        hr_status = (er_status == True) | (pr_status == True)
        
        # Derive 4-class subtype
        subtype_series = pd.Series('Unknown', index=self.clinical_table.index)
        
        # HR+/HER2-
        mask = (hr_status == True) & (her2_status == False)
        subtype_series[mask] = 'HR+/HER2-'
        
        # HR+/HER2+
        mask = (hr_status == True) & (her2_status == True)
        subtype_series[mask] = 'HR+/HER2+'
        
        # HR-/HER2+
        mask = (hr_status == False) & (her2_status == True)
        subtype_series[mask] = 'HR-/HER2+'
        
        # TNBC (HR-/HER2-)
        mask = (hr_status == False) & (her2_status == False)
        subtype_series[mask] = 'TNBC'
        
        # Count non-Unknown
        n_valid = (subtype_series != 'Unknown').sum()
        if n_valid < len(self.clinical_table) * 0.1:  # Less than 10% valid
            logger.warning("Derived subtype has too few valid values")
            return None
        
        logger.info(f"Derived subtype from ER/PR/HER2: {subtype_series.value_counts().to_dict()}")
        return subtype_series
    
    def _load_brca_labels(self):
        """Load subtype labels from BRCA-data-with-integer-labels.csv file."""
        try:
            from pathlib import Path
            
            labels_path = Path(self.brca_labels_file)
            if not labels_path.exists():
                logger.warning(f"BRCA labels file not found: {self.brca_labels_file}")
                return
            
            # Read BRCA labels file
            df_labels = pd.read_csv(labels_path)
            
            # Check required columns
            if 'sample_id' not in df_labels.columns:
                logger.warning(f"BRCA labels file missing 'sample_id' column: {self.brca_labels_file}")
                return
            
            # Find label column (Subtype or similar)
            label_col = None
            for col in ['Subtype', 'subtype', 'label', 'Label', 'LABEL']:
                if col in df_labels.columns:
                    label_col = col
                    break
            
            if not label_col:
                logger.warning(f"BRCA labels file missing label column (Subtype/label): {self.brca_labels_file}")
                return
            
            # Extract patient IDs and create mapping
            df_labels['patient_id'] = df_labels['sample_id'].apply(extract_patient_id)
            
            # Create patient_id -> subtype mapping (keep first if duplicates)
            self.brca_labels_mapping = df_labels.groupby('patient_id')[label_col].first().to_dict()
            
            # Get unique subtypes
            unique_subtypes = sorted(df_labels[label_col].dropna().unique())
            
            logger.info(f"Loaded BRCA labels for {len(self.brca_labels_mapping)} patients")
            logger.info(f"Subtype distribution: {df_labels[label_col].value_counts().sort_index().to_dict()}")
            logger.info(f"Unique subtypes: {unique_subtypes}")
            
        except Exception as e:
            logger.warning(f"Error loading BRCA labels file: {e}")
            self.brca_labels_mapping = None
    
    def _load_pam50(self):
        """Load PAM50 labels from file."""
        try:
            from pathlib import Path
            
            pam50_path = Path(self.pam50_file)
            if not pam50_path.exists():
                logger.warning(f"PAM50 file not found: {self.pam50_file}")
                return
            
            # Read PAM50 file
            df_pam50 = pd.read_csv(self.pam50_file, sep='\t')
            
            # Check required columns
            if 'Sample' not in df_pam50.columns or 'PAM50' not in df_pam50.columns:
                logger.warning(f"PAM50 file missing required columns (Sample, PAM50): {self.pam50_file}")
                return
            
            # Extract patient IDs and create mapping
            df_pam50['patient_id'] = df_pam50['Sample'].apply(extract_patient_id)
            
            # Create patient_id -> PAM50 mapping (keep first if duplicates)
            self.pam50_mapping = df_pam50.groupby('patient_id')['PAM50'].first().to_dict()
            
            logger.info(f"Loaded PAM50 labels for {len(self.pam50_mapping)} patients")
            logger.info(f"PAM50 distribution: {pd.Series(list(self.pam50_mapping.values())).value_counts().to_dict()}")
            
        except Exception as e:
            logger.warning(f"Error loading PAM50 file: {e}")
            self.pam50_mapping = None
    
    def _discover_or_derive_subtype(self):
        """Discover or derive subtype labels."""
        # Priority 1: Use BRCA labels file if available (highest priority)
        if self.brca_labels_mapping is not None:
            unique_vals = set(self.brca_labels_mapping.values())
            unique_vals = [v for v in unique_vals if pd.notna(v)]
            
            if len(unique_vals) >= 2:
                # 정수 레이블을 문자열로 변환
                unique_vals = sorted([str(v) for v in unique_vals])
                self.subtype_classes = unique_vals + ['Unknown']
                self.subtype_to_idx = {s: i for i, s in enumerate(self.subtype_classes)}
                self.idx_to_subtype = {i: s for s, i in self.subtype_to_idx.items()}
                self.has_subtype = True
                logger.info(f"Using BRCA labels file subtypes: {self.subtype_classes}")
                return
        
        # Priority 2: Use PAM50 if available
        if self.pam50_mapping is not None:
            unique_vals = set(self.pam50_mapping.values())
            unique_vals = [v for v in unique_vals if pd.notna(v) and v != 'Unknown']
            
            if len(unique_vals) >= 2:
                self.subtype_classes = sorted(unique_vals) + ['Unknown']
                self.subtype_to_idx = {s: i for i, s in enumerate(self.subtype_classes)}
                self.idx_to_subtype = {i: s for s, i in self.subtype_to_idx.items()}
                self.has_subtype = True
                logger.info(f"Using PAM50 subtypes: {self.subtype_classes}")
                return
        
        # Priority 2: Try discovery from clinical table
        subtype_col = self._discover_subtype_column()
        
        if subtype_col:
            subtype_series = self.clinical_table[subtype_col]
        else:
            # Priority 3: Try derivation from ER/PR/HER2
            subtype_series = self._derive_subtype_from_hr_her2()
        
        if subtype_series is not None:
            unique_vals = subtype_series.dropna().unique()
            unique_vals = [v for v in unique_vals if v != 'Unknown']
            
            if len(unique_vals) >= 2:
                self.subtype_classes = sorted(unique_vals) + ['Unknown']
                self.subtype_to_idx = {s: i for i, s in enumerate(self.subtype_classes)}
                self.idx_to_subtype = {i: s for s, i in self.subtype_to_idx.items()}
                self.has_subtype = True
                logger.info(f"Subtype classes: {self.subtype_classes}")
            else:
                logger.warning("Subtype has <2 classes, disabling subtype task")
        else:
            logger.info("No subtype available, will only train stage task")
    
    def get_stage_labels(self, patient_ids: List[str]) -> np.ndarray:
        """Get stage labels for patient IDs."""
        stage_col = self._discover_stage_column()
        if not stage_col:
            return np.full(len(patient_ids), self.stage_to_idx['Unknown'], dtype=np.int64)
        
        labels = []
        for pid in patient_ids:
            if pid in self.clinical_table.index:
                stage_val = self.clinical_table.loc[pid, stage_col]
                stage_normalized = self._normalize_stage(stage_val)
                labels.append(self.stage_to_idx[stage_normalized])
            else:
                labels.append(self.stage_to_idx['Unknown'])
        
        return np.array(labels, dtype=np.int64)
    
    def get_subtype_labels(self, patient_ids: List[str]) -> Optional[np.ndarray]:
        """Get subtype labels for patient IDs (if available)."""
        if not self.has_subtype:
            return None
        
        labels = []
        for pid in patient_ids:
            # Priority 1: Use BRCA labels file if available (highest priority)
            if self.brca_labels_mapping is not None and pid in self.brca_labels_mapping:
                subtype_val = self.brca_labels_mapping[pid]
                subtype_str = str(subtype_val) if pd.notna(subtype_val) else 'Unknown'
                if subtype_str in self.subtype_to_idx:
                    labels.append(self.subtype_to_idx[subtype_str])
                else:
                    labels.append(self.subtype_to_idx['Unknown'])
            # Priority 2: Use PAM50 if available
            elif self.pam50_mapping is not None and pid in self.pam50_mapping:
                pam50_val = self.pam50_mapping[pid]
                if pd.notna(pam50_val) and pam50_val in self.subtype_to_idx:
                    labels.append(self.subtype_to_idx[pam50_val])
                else:
                    labels.append(self.subtype_to_idx['Unknown'])
            # Priority 3: Try clinical table
            elif pid in self.clinical_table.index:
                # Try to get from stored subtype column or derived
                # For now, return Unknown if not found
                labels.append(self.subtype_to_idx['Unknown'])
            else:
                labels.append(self.subtype_to_idx['Unknown'])
        
        return np.array(labels, dtype=np.int64)
    
    def get_class_weights_stage(self, patient_ids: List[str]) -> np.ndarray:
        """Compute class weights for stage classification."""
        labels = self.get_stage_labels(patient_ids)
        unique, counts = np.unique(labels, return_counts=True)
        
        total = len(labels)
        weights = np.ones(len(self.stage_classes))
        
        for idx, count in zip(unique, counts):
            if count > 0:
                weights[idx] = total / (len(self.stage_classes) * count)
        
        return weights.astype(np.float32)
    
    def get_class_weights_subtype(self, patient_ids: List[str]) -> Optional[np.ndarray]:
        """Compute class weights for subtype classification."""
        if not self.has_subtype:
            return None
        
        labels = self.get_subtype_labels(patient_ids)
        if labels is None:
            return None
        
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        weights = np.ones(len(self.subtype_classes))
        
        for idx, count in zip(unique, counts):
            if count > 0:
                weights[idx] = total / (len(self.subtype_classes) * count)
        
        return weights.astype(np.float32)

