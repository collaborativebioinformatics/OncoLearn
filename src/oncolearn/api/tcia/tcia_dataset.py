"""
TCIA dataset implementation for downloading imaging data manifests.
"""

from pathlib import Path
from typing import Optional

from oncolearn.utils.download import download_file, run_external_command

from ..dataset import DataCategory, Dataset


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

    def download(self, output_dir: Optional[str] = None, download_images: bool = False, extract: bool = True, verbose: bool = True, confirm: bool = True) -> bool:
        """
        Download the manifest file and optionally the imaging data.

        Args:
            output_dir: Directory to save the manifest file
            download_images: If True and file is a .tcia manifest, run nbia-data-retriever
            extract: Whether to extract gzipped files after download (currently unused for TCIA)
            verbose: Print progress messages
            confirm: Whether to ask for confirmation before downloading

        Returns:
            True if successful, False otherwise
        """
        if output_dir is None:
            output_dir = "data/tcia/manifests"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dest_file = output_path / self.filename

        # Download the manifest file
        success = download_file(
            url=self.url,
            output_dir=str(output_path),
            filename=self.filename,
            dataset_name=self.filename if verbose else None,
            verbose=verbose,
            confirm=confirm
        )

        if not success:
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
                print(f"    [ERROR] Manifest file not found: {manifest_path}")
            return False

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command - nbia-data-retriever requires --cli flag
        # JAVA_TOOL_OPTIONS=-Djava.awt.headless=true should be set in environment
        cmd = [
            "nbia-data-retriever",
            "--cli",
            str(manifest_path),
            "-d",
            str(output_dir)
        ]

        if verbose:
            print(f"    Downloading images from {manifest_path.name}...")
            print(
                "    This may take a while depending on the number and size of images...")

        return run_external_command(
            command=cmd,
            command_name=f"nbia-data-retriever for {manifest_path.name}",
            verbose=verbose,
            stream_output=True  # Stream output for interactive progress display
        )
