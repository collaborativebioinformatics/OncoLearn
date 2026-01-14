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
from urllib.request import urlopen, urlretrieve

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to a human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB", "250.3 MB", "5.2 KB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_file_size_from_url(url: str) -> Optional[int]:
    """
    Get the size of a file from its URL by reading the Content-Length header.
    
    Args:
        url: The URL to check
        
    Returns:
        File size in bytes, or None if size cannot be determined
    """
    try:
        with urlopen(url, timeout=10) as response:
            content_length = response.headers.get('Content-Length')
            if content_length:
                return int(content_length)
    except Exception:
        pass
    return None


def confirm_download(filename: str, size_bytes: int, verbose: bool = True) -> bool:
    """
    Ask the user to confirm a download after displaying file size information.
    
    Args:
        filename: Name of the file to download
        size_bytes: Size of the file in bytes
        verbose: Whether to show prompts (if False, auto-confirms)
        
    Returns:
        True if user confirms or verbose is False, False if user declines
    """
    if not verbose:
        return True
    
    size_str = format_file_size(size_bytes)
    print("\nWARNING: Download Size Warning:")
    print(f"    File: {filename}")
    print(f"    Size: {size_str}")
    
    # Provide context for large files
    if size_bytes > 1024 * 1024 * 1024:  # > 1 GB
        print("    WARNING: This is a large file that may take significant time to download.")
    
    while True:
        response = input("    Do you wish to continue? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            print("    Download cancelled by user.")
            return False
        else:
            print("    Please enter 'y' for yes or 'n' for no.")


def confirm_cohort_download(cohort_name: str, total_size_bytes: int, file_details: List[tuple[str, int]], verbose: bool = True) -> bool:
    """
    Ask the user to confirm a cohort download after displaying total size information.
    
    Args:
        cohort_name: Name of the cohort to download
        total_size_bytes: Total size of all files in bytes
        file_details: List of tuples containing (filename, size_in_bytes) for each file
        verbose: Whether to show prompts (if False, auto-confirms)
        
    Returns:
        True if user confirms or verbose is False, False if user declines
    """
    if not verbose:
        return True
    
    size_str = format_file_size(total_size_bytes)
    
    # Header with cohort name
    print("\n" + "=" * 70)
    print(f"DOWNLOADING COHORT: {cohort_name}")
    print("=" * 70)
    
    # List each file with its size
    print("\nFiles to download:")
    for i, (filename, size) in enumerate(file_details, 1):
        if size > 0:
            file_size_str = format_file_size(size)
            print(f"  {i:2}. {filename:<50} {file_size_str:>10}")
        else:
            print(f"  {i:2}. {filename:<50} {'(size unknown)':>10}")
    
    # Summary section at bottom
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Cohort: {cohort_name}")
    print(f"  Files:  {len(file_details)} file(s)")
    print(f"  Total:  {size_str}")
    
    # Provide context for large downloads
    if total_size_bytes > 1024 * 1024 * 1024:  # > 1 GB
        print("\n  WARNING: This is a large download that may take significant time.")
        print("           Please ensure you have sufficient disk space and bandwidth.")
    
    print("=" * 70)
    
    while True:
        response = input("\nDo you wish to proceed with this download? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            print()
            return True
        elif response in ['n', 'no']:
            print("Download cancelled by user.\n")
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def calculate_urls_total_size(urls: List[str], verbose: bool = False) -> tuple[int, int]:
    """
    Calculate the total size of multiple URLs.
    
    Args:
        urls: List of URLs to check
        verbose: Whether to print progress messages
        
    Returns:
        Tuple of (total_size_bytes, files_with_known_size_count)
    """
    total_size = 0
    known_count = 0
    
    if verbose:
        print("Calculating total download size...")
    
    for url in urls:
        size = get_file_size_from_url(url)
        if size:
            total_size += size
            known_count += 1
    
    return total_size, known_count


def download_file(
    url: str,
    output_dir: str,
    filename: str,
    dataset_name: Optional[str] = None,
    verbose: bool = True,
    confirm: bool = True
) -> bool:
    """
    Download a file from a URL to a specified directory.
    
    Args:
        url: The URL to download from
        output_dir: Directory to save the downloaded file
        filename: Name of the file to save
        dataset_name: Optional human-readable name of the dataset (for logging)
        verbose: Whether to print progress messages
        confirm: Whether to ask for confirmation before downloading
        
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
        # Check file size before downloading
        file_size = get_file_size_from_url(url)
        if file_size:
            if verbose:
                print(f"    Size: {format_file_size(file_size)}")
            
            # Ask for confirmation if file size is available and confirm is True
            if confirm and not confirm_download(filename, file_size, verbose):
                return False
        
        # Download the file with progress bar if tqdm is available
        if TQDM_AVAILABLE and file_size and verbose:
            # Use tqdm progress bar
            with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, 
                     desc="    Downloading", ncols=80) as pbar:
                def reporthook(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    if downloaded <= total_size:
                        pbar.update(block_size)
                
                urlretrieve(url, dest_file, reporthook=reporthook)
        else:
            # Fallback to simple download without progress
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
    verbose: bool = True,
    confirm: bool = True
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
        confirm: Whether to ask for confirmation before downloading
        
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
        # Check file size before downloading
        file_size = get_file_size_from_url(url)
        if file_size:
            if verbose:
                print(f"    Size: {format_file_size(file_size)}")
            
            # Ask for confirmation if file size is available and confirm is True
            if confirm and not confirm_download(filename, file_size, verbose):
                return False
        
        # Download the file with progress bar if tqdm is available
        if TQDM_AVAILABLE and file_size and verbose:
            # Use tqdm progress bar
            with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024,
                     desc="    Downloading", ncols=80) as pbar:
                def reporthook(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    if downloaded <= total_size:
                        pbar.update(block_size)
                
                urlretrieve(url, dest_file, reporthook=reporthook)
        else:
            # Fallback to simple download without progress
            urlretrieve(url, dest_file)
        
        if verbose:
            size_mb = dest_file.stat().st_size / (1024 * 1024)
            print(f"    Downloaded: {size_mb:.2f} MB")
        
        # Extract if it's a gzip file and extract flag is True
        if extract and filename.endswith('.gz'):
            output_file = dest_file.with_suffix('')
            if verbose:
                print("    Extracting...")
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
            print("    [OK] Command completed successfully")
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
            print(f"    [ERROR] Command '{cmd_name}' not found. Please install it first.")
        return False
    except subprocess.CalledProcessError as e:
        if verbose:
            error_msg = f"Command failed with exit code {e.returncode}"
            if e.stderr:
                error_msg += f": {e.stderr}"
            print(f"    [ERROR] {error_msg}")
        return False
    except Exception as e:
        if verbose:
            print(f"    [ERROR] {e}")
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
