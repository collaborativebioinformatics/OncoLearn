"""
Fast DICOM reading, series sampling, and caching for V1 variant.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DICOMCache:
    """Simple cache for DICOM pixel arrays."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.access_order: List[str] = []
    
    def get(self, path: str) -> Optional[np.ndarray]:
        """Get cached array or None."""
        if path in self.cache:
            # Update access order
            if path in self.access_order:
                self.access_order.remove(path)
            self.access_order.append(path)
            return self.cache[path]
        return None
    
    def put(self, path: str, array: np.ndarray):
        """Cache array."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru = self.access_order.pop(0)
            del self.cache[lru]
        
        self.cache[path] = array
        if path in self.access_order:
            self.access_order.remove(path)
        self.access_order.append(path)
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()


# Global cache instance
_dicom_cache = DICOMCache(max_size=1000)


def read_dicom_image(dicom_path: str, use_cache: bool = True) -> Optional[np.ndarray]:
    """
    Read DICOM image and return pixel array.
    
    Returns:
        Pixel array (H, W) or None if error
    """
    try:
        # Check cache
        if use_cache:
            cached = _dicom_cache.get(dicom_path)
            if cached is not None:
                return cached
        
        # Read DICOM
        ds = pydicom.dcmread(dicom_path)
        
        # Get pixel array
        if not hasattr(ds, 'pixel_array'):
            logger.warning(f"No pixel_array in {dicom_path}")
            return None
        
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # Handle 3D arrays (multi-slice DICOM)
        # If 3D, take the middle slice or first slice
        if pixel_array.ndim == 3:
            # Use middle slice for better representation
            slice_idx = pixel_array.shape[0] // 2
            pixel_array = pixel_array[slice_idx]
            logger.debug(f"3D DICOM detected, using middle slice {slice_idx} from {pixel_array.shape[0]} slices")
        elif pixel_array.ndim > 3:
            # Flatten extra dimensions (shouldn't happen but handle it)
            logger.warning(f"DICOM with {pixel_array.ndim}D array, shape: {pixel_array.shape}. Flattening to 2D.")
            pixel_array = pixel_array.reshape(-1, pixel_array.shape[-2], pixel_array.shape[-1])[0]
        
        # Ensure 2D array
        if pixel_array.ndim != 2:
            logger.warning(f"Unexpected pixel_array shape: {pixel_array.shape}, expected 2D. Skipping.")
            return None
        
        # Apply rescale slope/intercept if present
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            slope = float(ds.RescaleSlope) if ds.RescaleSlope else 1.0
            intercept = float(ds.RescaleIntercept) if ds.RescaleIntercept else 0.0
            pixel_array = pixel_array * slope + intercept
        
        # Cache
        if use_cache:
            _dicom_cache.put(dicom_path, pixel_array)
        
        return pixel_array
    
    except Exception as e:
        logger.debug(f"Error reading DICOM {dicom_path}: {e}")
        return None


def parse_dicom_series_json(dicom_series_json: str) -> Dict:
    """Parse dicom_series JSON string to dict."""
    if not dicom_series_json or dicom_series_json == '{}':
        return {}
    
    try:
        if isinstance(dicom_series_json, str):
            return json.loads(dicom_series_json)
        return dicom_series_json
    except:
        return {}


def get_all_dicom_paths_from_series(
    dicom_series_json: str,
    dicom_root: str,
    modality: str = 'MR'
) -> Dict[str, List[str]]:
    """
    Get all DICOM file paths organized by series.
    
    Args:
        dicom_series_json: JSON string with series structure
        dicom_root: Root directory for DICOM files
        modality: Preferred modality ('MR' or 'MG')
    
    Returns:
        Dict mapping series_uid to list of DICOM file paths
    """
    series_dict = parse_dicom_series_json(dicom_series_json)
    if not series_dict:
        return {}
    
    series_paths = {}
    
    for study_uid, study_series in series_dict.items():
        for series_uid, series_info in study_series.items():
            series_modality = series_info.get('modality', '')
            if series_modality == modality:
                example_paths = series_info.get('example_paths', [])
                if example_paths:
                    example_path = example_paths[0]
                    example_dir = Path(example_path).parent
                    
                    # Find all DICOM files in this directory, sorted
                    dicom_files = sorted(list(example_dir.glob('*.dcm')) + list(example_dir.glob('*.DCM')))
                    if dicom_files:
                        series_paths[series_uid] = [str(f) for f in dicom_files]
    
    # Fallback: try other modalities if preferred not found
    if not series_paths:
        for study_uid, study_series in series_dict.items():
            for series_uid, series_info in study_series.items():
                example_paths = series_info.get('example_paths', [])
                if example_paths:
                    example_path = example_paths[0]
                    example_dir = Path(example_path).parent
                    dicom_files = sorted(list(example_dir.glob('*.dcm')) + list(example_dir.glob('*.DCM')))
                    if dicom_files:
                        series_paths[series_uid] = [str(f) for f in dicom_files]
    
    return series_paths


def sample_dicom_series_uniform(
    dicom_series_json: str,
    dicom_root: str,
    modality: str = 'MR',
    n_samples: int = 5
) -> List[str]:
    """
    Uniformly sample DICOM file paths from a single series.
    
    Args:
        dicom_series_json: JSON string with series structure
        dicom_root: Root directory for DICOM files
        modality: Preferred modality ('MR' or 'MG')
        n_samples: Number of images to sample (default: 5)
    
    Returns:
        List of DICOM file paths (uniformly sampled)
    """
    series_paths = get_all_dicom_paths_from_series(dicom_series_json, dicom_root, modality)
    
    # Combine all series into one list (sorted by series, then by file)
    all_paths = []
    for series_uid in sorted(series_paths.keys()):
        all_paths.extend(series_paths[series_uid])
    
    if not all_paths:
        return []
    
    # Uniform sampling: if 70 images, sample every 70/5=14th image
    n_total = len(all_paths)
    n_samples = min(n_samples, n_total)
    
    if n_samples == 1:
        # Single sample: take middle
        return [all_paths[n_total // 2]]
    
    # Calculate step size
    step = n_total / n_samples
    
    # Sample uniformly
    sampled_indices = [int(i * step) for i in range(n_samples)]
    sampled = [all_paths[idx] for idx in sampled_indices]
    
    return sampled


def sample_dicom_series(
    dicom_series_json: str,
    dicom_root: str,
    modality: str = 'MR',
    n_samples: int = 10,
    random_seed: Optional[int] = None
) -> List[str]:
    """
    Sample DICOM file paths from series (random sampling, for backward compatibility).
    
    Args:
        dicom_series_json: JSON string with series structure
        dicom_root: Root directory for DICOM files
        modality: Preferred modality ('MR' or 'MG')
        n_samples: Number of images to sample
        random_seed: Random seed for sampling
    
    Returns:
        List of DICOM file paths
    """
    series_paths = get_all_dicom_paths_from_series(dicom_series_json, dicom_root, modality)
    
    # Combine all series into one list
    all_paths = []
    for series_uid in sorted(series_paths.keys()):
        all_paths.extend(series_paths[series_uid])
    
    if not all_paths:
        return []
    
    # Random sample
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random
    
    n_samples = min(n_samples, len(all_paths))
    sampled = rng.choice(all_paths, size=n_samples, replace=False).tolist()
    
    return sampled


def get_all_series_from_patient(
    dicom_series_json: str,
    dicom_root: str,
    modality: str = 'MR'
) -> List[Dict[str, any]]:
    """
    Get all series from a patient, each as a separate sequence.
    
    Args:
        dicom_series_json: JSON string with series structure
        dicom_root: Root directory for DICOM files
        modality: Preferred modality ('MR' or 'MG')
    
    Returns:
        List of dicts, each containing:
        {
            'series_uid': str,
            'modality': str,
            'paths': List[str],  # All DICOM paths in this series (sorted)
            'n_images': int
        }
    """
    series_paths = get_all_dicom_paths_from_series(dicom_series_json, dicom_root, modality)
    series_dict = parse_dicom_series_json(dicom_series_json)
    
    result = []
    for study_uid, study_series in series_dict.items():
        for series_uid, series_info in study_series.items():
            if series_uid in series_paths:
                result.append({
                    'series_uid': series_uid,
                    'modality': series_info.get('modality', modality),
                    'paths': series_paths[series_uid],
                    'n_images': len(series_paths[series_uid])
                })
    
    return result


def load_dicom_series_batch(
    dicom_paths: List[str],
    max_images: int = 20
) -> Optional[torch.Tensor]:
    """
    Load batch of DICOM images.
    
    Returns:
        Tensor of shape (N, H, W) or None if all failed
    """
    images = []
    
    for path in dicom_paths[:max_images]:
        pixel_array = read_dicom_image(path)
        if pixel_array is not None:
            images.append(pixel_array)
    
    if not images:
        return None
    
    # Stack into tensor
    # Pad to same size if needed (simple: use max H, W)
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    
    padded_images = []
    for img in images:
        h, w = img.shape
        padded = np.zeros((max_h, max_w), dtype=np.float32)
        padded[:h, :w] = img
        padded_images.append(padded)
    
    return torch.from_numpy(np.stack(padded_images))

