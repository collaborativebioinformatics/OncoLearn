"""
Utility functions for UCSC Xena Browser data downloads.
"""

import gzip
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
from typing import Optional


def download_and_extract_gzip(
    url: str,
    output_dir: str,
    filename: str,
    dataset_name: str,
    extract: bool = True
) -> bool:
    """
    Download a gzipped file from a URL and optionally extract it.
    
    Args:
        url: The URL to download from
        output_dir: Directory to save the downloaded file
        filename: Name of the file to save
        dataset_name: Human-readable name of the dataset (for logging)
        extract: Whether to extract the gzip file after download
        
    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dest_file = output_path / filename
    
    print(f"  Downloading: {dataset_name}")
    print(f"    URL: {url}")
    
    try:
        # Download the file
        urlretrieve(url, dest_file)
        size_mb = dest_file.stat().st_size / (1024 * 1024)
        print(f"    Downloaded: {size_mb:.2f} MB")
        
        # Extract if it's a gzip file and extract flag is True
        if extract and filename.endswith('.gz'):
            output_file = dest_file.with_suffix('')
            with gzip.open(dest_file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            dest_file.unlink()
            print(f"    Extracted: {output_file.name}")
        
        return True
        
    except URLError as e:
        print(f"    ERROR: Failed to download - {e}")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False
