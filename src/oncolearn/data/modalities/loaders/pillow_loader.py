from PIL import Image
from pathlib import Path
from .base import BaseDataLoader


class PillowLoader(BaseDataLoader):
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
        except (FileNotFoundError, OSError, IOError) as e:
            raise OSError(
                f"Failed to open image file {img_path}: {e}"
            ) from e
