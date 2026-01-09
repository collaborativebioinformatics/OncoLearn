"""
Cohort loading and filtering for V1 (imaging-present) and V2 (no-imaging) variants.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_cohort_index(cohort_index_path: str) -> pd.DataFrame:
    """Load cohort index parquet/csv file."""
    path = Path(cohort_index_path)
    if path.suffix == '.parquet':
        df = pd.read_parquet(cohort_index_path)
    else:
        df = pd.read_csv(cohort_index_path)
    
    required_cols = ['patient_id', 'clinical_row_id']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"cohort_index missing required columns: {missing}")
    
    logger.info(f"Loaded cohort_index: {len(df)} patients")
    return df


def load_clinical_table(clinical_table_path: str) -> pd.DataFrame:
    """Load clinical table parquet/csv file."""
    path = Path(clinical_table_path)
    if path.suffix == '.parquet':
        df = pd.read_parquet(clinical_table_path)
    else:
        df = pd.read_csv(clinical_table_path)
    
    if 'patient_id' not in df.index.names and 'patient_id' not in df.columns:
        raise ValueError("clinical_table must have 'patient_id' as index or column")
    
    if 'patient_id' in df.columns and 'patient_id' not in df.index.names:
        df = df.set_index('patient_id')
    
    logger.info(f"Loaded clinical_table: {len(df)} patients, {len(df.columns)} features")
    return df


def load_gene_set_table(gene_set_path: str) -> pd.DataFrame:
    """Load gene set table parquet/csv file."""
    path = Path(gene_set_path)
    if path.suffix == '.parquet':
        df = pd.read_parquet(gene_set_path)
    else:
        df = pd.read_csv(gene_set_path)
    
    if 'patient_id' not in df.index.names and 'patient_id' not in df.columns:
        raise ValueError("gene_set_table must have 'patient_id' as index or column")
    
    if 'patient_id' in df.columns and 'patient_id' not in df.index.names:
        df = df.set_index('patient_id')
    
    logger.info(f"Loaded gene_set_table: {len(df)} patients, {len(df.columns)} gene sets")
    return df


def is_imaging_present(row: pd.Series, dicom_root: Optional[str] = None) -> bool:
    """
    Determine if patient has imaging data.
    
    Checks:
    1. has_imaging column (if exists)
    2. imaging_modalities contains "MR" or "MG"
    3. dicom_series_json is non-empty
    4. DICOM files exist in dicom_root (if provided)
    """
    # Check has_imaging column
    if 'has_imaging' in row.index:
        if pd.notna(row['has_imaging']) and row['has_imaging']:
            return True
    
    # Check imaging_modalities
    if 'imaging_modalities' in row.index:
        modalities = row['imaging_modalities']
        if pd.notna(modalities):
            if isinstance(modalities, str):
                # Try to parse as JSON list
                try:
                    mod_list = json.loads(modalities)
                except:
                    mod_list = [modalities]
            elif isinstance(modalities, list):
                mod_list = modalities
            else:
                mod_list = []
            
            if any(m in ['MR', 'MG'] for m in mod_list):
                return True
    
    # Check dicom_series_json
    if 'dicom_series' in row.index:
        dicom_json = row['dicom_series']
        if pd.notna(dicom_json) and dicom_json and dicom_json != '{}':
            try:
                series_dict = json.loads(dicom_json) if isinstance(dicom_json, str) else dicom_json
                if series_dict and len(series_dict) > 0:
                    return True
            except:
                pass
    
    # Optional: check DICOM root (if provided)
    if dicom_root:
        # This is a fallback check - would require scanning files
        # For now, skip this expensive check
        pass
    
    return False


def build_cohorts(
    cohort_index_path: str,
    clinical_table_path: str,
    gene_set_path: str,
    dicom_root: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all data tables and build V1/V2 cohorts.
    
    Returns:
        cohort_index: Full cohort index
        clinical_table: Clinical features
        gene_set_table: Gene set scores
        cohort_v1: V1 cohort (imaging-present) subset of cohort_index
        cohort_v2: V2 cohort (no-imaging) subset of cohort_index
    """
    # Load tables
    cohort_index = load_cohort_index(cohort_index_path)
    clinical_table = load_clinical_table(clinical_table_path)
    gene_set_table = load_gene_set_table(gene_set_path)
    
    # Determine imaging status
    cohort_index['imaging_present'] = cohort_index.apply(
        lambda row: is_imaging_present(row, dicom_root), axis=1
    )
    
    # Build V1 cohort (imaging-present)
    cohort_v1 = cohort_index[cohort_index['imaging_present']].copy()
    
    # Build V2 cohort (no-imaging)
    cohort_v2 = cohort_index[~cohort_index['imaging_present']].copy()
    
    # Filter tables to patients in respective cohorts
    # Note: We keep full tables and filter in Dataset
    
    logger.info(f"V1 cohort (imaging-present): {len(cohort_v1)} patients")
    logger.info(f"V2 cohort (no-imaging): {len(cohort_v2)} patients")
    
    # Log stage distribution if available
    if 'stage' in cohort_v1.columns:
        logger.info(f"V1 stage distribution:\n{cohort_v1['stage'].value_counts()}")
    if 'stage' in cohort_v2.columns:
        logger.info(f"V2 stage distribution:\n{cohort_v2['stage'].value_counts()}")
    
    return cohort_index, clinical_table, gene_set_table, cohort_v1, cohort_v2


def get_cohort_for_variant(
    cohort_index: pd.DataFrame,
    variant: str
) -> pd.DataFrame:
    """Get cohort subset for specified variant."""
    if variant == 'v1_imaging':
        return cohort_index[cohort_index['imaging_present']].copy()
    elif variant == 'v2_no_imaging':
        return cohort_index[~cohort_index['imaging_present']].copy()
    else:
        raise ValueError(f"Unknown variant: {variant}. Must be 'v1_imaging' or 'v2_no_imaging'")




