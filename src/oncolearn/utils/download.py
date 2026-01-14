"""
Common download utilities for OncoLearn.

This module provides centralized download functionality used across different
data sources (UCSC Xena Browser, TCIA, etc.).
"""

import gzip
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
from urllib.error import URLError
from urllib.request import urlretrieve


def download_file(
    url: str,
    output_dir: str,
    filename: str,
    dataset_name: Optional[str] = None,
    verbose: bool = True
) -> bool:
    """
    Download a file from a URL to a specified directory.
    
    Args:
        url: The URL to download from
        output_dir: Directory to save the downloaded file
        filename: Name of the file to save
        dataset_name: Optional human-readable name of the dataset (for logging)
        verbose: Whether to print progress messages
        
    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dest_file = output_path / filename
    
    if verbose:
        if dataset_name:
            print(f"  Downloading: {dataset_name}")
        else:
            print(f"  Downloading: {filename}")
        print(f"    URL: {url}")
    
    try:
        # Download the file
        urlretrieve(url, dest_file)
        
        if verbose:
            size_mb = dest_file.stat().st_size / (1024 * 1024)
            print(f"    Downloaded: {size_mb:.2f} MB")
        
        return True
        
    except URLError as e:
        if verbose:
            print(f"    ERROR: Failed to download - {e}")
        return False
    except Exception as e:
        if verbose:
            print(f"    ERROR: {e}")
        return False


def download_and_extract_gzip(
    url: str,
    output_dir: str,
    filename: str,
    dataset_name: Optional[str] = None,
    extract: bool = True,
    verbose: bool = True
) -> bool:
    """
    Download a gzipped file from a URL and optionally extract it.
    
    Args:
        url: The URL to download from
        output_dir: Directory to save the downloaded file
        filename: Name of the file to save
        dataset_name: Optional human-readable name of the dataset (for logging)
        extract: Whether to extract the gzip file after download
        verbose: Whether to print progress messages
        
    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dest_file = output_path / filename
    
    if verbose:
        if dataset_name:
            print(f"  Downloading: {dataset_name}")
        else:
            print(f"  Downloading: {filename}")
        print(f"    URL: {url}")
    
    try:
        # Download the file
        urlretrieve(url, dest_file)
        
        if verbose:
            size_mb = dest_file.stat().st_size / (1024 * 1024)
            print(f"    Downloaded: {size_mb:.2f} MB")
        
        # Extract if it's a gzip file and extract flag is True
        if extract and filename.endswith('.gz'):
            output_file = dest_file.with_suffix('')
            with gzip.open(dest_file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            dest_file.unlink()
            if verbose:
                print(f"    Extracted: {output_file.name}")
        
        return True
        
    except URLError as e:
        if verbose:
            print(f"    ERROR: Failed to download - {e}")
        return False
    except Exception as e:
        if verbose:
            print(f"    ERROR: {e}")
        return False


def run_external_command(
    command: List[str],
    command_name: Optional[str] = None,
    verbose: bool = True
) -> bool:
    """
    Run an external command as a subprocess.
    
    Args:
        command: List of command arguments (e.g., ["ls", "-la"])
        command_name: Optional human-readable name of the command (for logging)
        verbose: Whether to print progress and output messages
        
    Returns:
        True if successful, False otherwise
    """
    if verbose:
        cmd_display = command_name if command_name else ' '.join(command)
        print(f"    Running: {cmd_display}")
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        if verbose:
            print("    ✓ Command completed successfully")
            if result.stdout:
                # Print stdout, but limit to avoid too much output
                lines = result.stdout.strip().split('\n')
                if len(lines) <= 10:
                    for line in lines:
                        print(f"      {line}")
                else:
                    for line in lines[:5]:
                        print(f"      {line}")
                    print(f"      ... ({len(lines) - 10} more lines) ...")
                    for line in lines[-5:]:
                        print(f"      {line}")
        
        return True
        
    except FileNotFoundError:
        if verbose:
            cmd_name = command[0] if command else "command"
            print(f"    ✗ Command '{cmd_name}' not found. Please install it first.")
        return False
    except subprocess.CalledProcessError as e:
        if verbose:
            error_msg = f"Command failed with exit code {e.returncode}"
            if e.stderr:
                error_msg += f": {e.stderr}"
            print(f"    ✗ {error_msg}")
        return False
    except Exception as e:
        if verbose:
            print(f"    ✗ {e}")
        return False


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
