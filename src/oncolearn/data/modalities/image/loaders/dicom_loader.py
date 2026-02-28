import numpy as np
from PIL import Image
from pathlib import Path
from .base import BaseImageLoader


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
            import pydicom
            import SimpleITK as sitk
        except ImportError:
            raise ImportError(
                "pydicom and SimpleITK required for DICOM files. "
                "Run `uv add oncolearn[image]` to install them."
            )
            
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
