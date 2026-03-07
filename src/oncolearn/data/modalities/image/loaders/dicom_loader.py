from functools import lru_cache

import numpy as np
from PIL import Image
from pathlib import Path
from .base import BaseImageLoader


@lru_cache(maxsize=1000)
def _read_pixel_array_cached(path_str: str) -> np.ndarray:
    """
    Read and rescale a DICOM pixel array, cached by path.

    Applies RescaleSlope / RescaleIntercept when present (matching the original
    pipeline), then returns the raw float32 array before any normalization.
    Cache size of 1000 matches the original implementation's global LRU cache.
    """
    import pydicom

    dicom = pydicom.dcmread(path_str)
    pixel_array = dicom.pixel_array.astype(np.float32)

    # Apply DICOM rescale tags when present (original pipeline requirement)
    slope = float(getattr(dicom, "RescaleSlope", 1.0))
    intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        pixel_array = pixel_array * slope + intercept

    return pixel_array


class DicomLoader(BaseImageLoader):
    """
    Image loader specialized in loading .dcm / .dicom arrays into PIL images.
    """

    @classmethod
    def can_load(cls, file_path: Path) -> bool:
        return file_path.suffix.lower() in ['.dcm', '.dicom']

    @classmethod
    def load(cls, img_path: Path) -> Image.Image:
        try:
            import pydicom  # noqa: F401 — validate import before calling cached fn
        except ImportError:
            raise ImportError(
                "pydicom required for DICOM files. "
                "Run `uv add oncolearn[image]` to install it."
            )

        try:
            pixel_array = _read_pixel_array_cached(str(img_path))
        except Exception:
            # Fallback to SimpleITK for files pydicom can't handle
            try:
                import SimpleITK as sitk
                image = sitk.ReadImage(str(img_path))
                pixel_array = sitk.GetArrayFromImage(image).astype(np.float32)
                if pixel_array.ndim == 3 and pixel_array.shape[0] == 1:
                    pixel_array = pixel_array[0]
            except Exception:
                raise RuntimeError(f"Failed to load DICOM file: {img_path}")

        # Handle multi-slice volumes: use middle slice (original behaviour)
        if pixel_array.ndim == 3:
            pixel_array = pixel_array[pixel_array.shape[0] // 2]

        # Min-max normalize to [0, 1] then scale to uint8
        pmin, pmax = pixel_array.min(), pixel_array.max()
        pixel_array = (pixel_array - pmin) / (pmax - pmin + 1e-8)
        pixel_array = (pixel_array * 255).astype(np.uint8)

        # Convert to RGB
        if pixel_array.ndim == 2:
            pixel_array = np.stack([pixel_array] * 3, axis=-1)

        return Image.fromarray(pixel_array)
