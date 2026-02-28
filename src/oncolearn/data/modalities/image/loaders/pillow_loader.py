from PIL import Image
from pathlib import Path
from .base import BaseImageLoader


class PillowLoader(BaseImageLoader):
    """
    Image loader specialized in loading standard RGB images like PNG or JPEG.
    """
    
    @classmethod
    def can_load(cls, file_path: Path) -> bool:
        return file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        
    @classmethod
    def load(cls, img_path: Path) -> Image.Image:
        try:
            image = Image.open(str(img_path)).convert('RGB')
            return image
        except ImportError:
            raise ImportError(
                "Pillow is required for standard image files. "
                "Run `uv add oncolearn[image]` to install it."
            )
