"""
TCIA dataset implementation for downloading imaging data manifests.
"""

import subprocess
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve
from urllib.error import URLError

from ..dataset import Dataset, DataCategory


class TCIADataset(Dataset):
    """
    A TCIA dataset that downloads manifest files and optionally imaging data.
    
    This class handles downloading .tcia manifest files and can optionally
    run the nbia-data-retriever tool to download actual imaging data.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        url: str,
        filename: str,
        default_subdir: str,
        file_type: str = "manifest",
        category: Optional[DataCategory] = None
    ):
        """
        Initialize a TCIA dataset.
        
        Args:
            name: Dataset name
            description: Dataset description
            url: Download URL for the manifest file
            filename: Filename to save as
            default_subdir: Default subdirectory within cohort folder
            file_type: Type of file ("manifest" for .tcia, "data" for supporting data)
            category: Data category (defaults to MANIFEST for manifest files, IMAGE for others)
        """
        super().__init__(name, description)
        
        # Set category based on file_type if not explicitly provided
        if category is not None:
            self.DATA_CATEGORY = category
        elif file_type == "manifest":
            self.DATA_CATEGORY = DataCategory.MANIFEST
        else:
            self.DATA_CATEGORY = DataCategory.IMAGE
            
        self.url = url
        self.filename = filename
        self.default_subdir = default_subdir
        self.file_type = file_type
    
    def download(self, output_dir: Optional[str] = None, download_images: bool = False, verbose: bool = True) -> bool:
        """
        Download the manifest file and optionally the imaging data.
        
        Args:
            output_dir: Directory to save the manifest file
            download_images: If True and file is a .tcia manifest, run nbia-data-retriever
            verbose: Print progress messages
            
        Returns:
            True if successful, False otherwise
        """
        if output_dir is None:
            output_dir = "data/tcia/manifests"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dest_file = output_path / self.filename
        
        # Download the manifest file
        try:
            if verbose:
                print(f"    Downloading {self.filename}...")
            urlretrieve(self.url, dest_file)
            if verbose:
                print(f"    ✓ {self.filename}")
        except URLError as e:
            if verbose:
                print(f"    ✗ {self.filename}: URL error - {e}")
            return False
        except Exception as e:
            if verbose:
                print(f"    ✗ {self.filename}: {e}")
            return False
        
        # If this is a .tcia manifest and download_images is True, run nbia-data-retriever
        if download_images and self.filename.endswith('.tcia'):
            # Download images to data/tcia/<cohort> instead of manifest directory
            images_dir = Path(f"data/tcia/{self.default_subdir}")
            return self._download_images(dest_file, images_dir, verbose)
        
        return True
    
    def _download_images(self, manifest_path: Path, output_dir: Path, verbose: bool = True) -> bool:
        """
        Run nbia-data-retriever CLI tool to download images from a manifest file.
        
        Args:
            manifest_path: Path to the .tcia manifest file
            output_dir: Directory to save downloaded images
            verbose: Print progress messages
            
        Returns:
            True if successful, False otherwise
        """
        if not manifest_path.exists():
            if verbose:
                print(f"    ✗ Manifest file not found: {manifest_path}")
            return False
        
        # Build command
        cmd = [
            "nbia-data-retriever",
            "--cli",
            str(manifest_path),
            "-d",
            str(output_dir)
        ]
        
        if verbose:
            print(f"    Downloading images from {manifest_path.name}...")
            print(f"      Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            if verbose:
                print(f"      ✓ Images downloaded successfully")
                if result.stdout:
                    print(f"      {result.stdout}")
            return True
        except FileNotFoundError:
            if verbose:
                print(f"      ✗ nbia-data-retriever not found. Please install it first.")
            return False
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed with exit code {e.returncode}"
            if e.stderr:
                error_msg += f": {e.stderr}"
            if verbose:
                print(f"      ✗ {error_msg}")
            return False
        except Exception as e:
            if verbose:
                print(f"      ✗ {e}")
            return False
