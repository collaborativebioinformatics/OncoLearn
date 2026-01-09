# TCIA Data Download Guide

This guide explains how to download medical imaging data from The Cancer Imaging Archive (TCIA) using the NBIA Data Retriever CLI tool that is pre-installed in the OncoLearn Docker container.

## Overview

The NBIA Data Retriever is a command-line tool for downloading DICOM imaging data from TCIA. Our Docker container comes with version 4.4.3 pre-installed and ready to use.

## What is TCIA?

The Cancer Imaging Archive (TCIA) is a public repository of de-identified medical images of cancer. It hosts large collections organized by disease type, imaging modality, or research focus. TCIA provides:

- **Radiology images** in DICOM format (MRI, CT, MG, etc.)
- **Supporting data** including patient outcomes, clinical data, genomics, and expert analyses
- **Public access** to large-scale cancer imaging datasets

## TCGA-BRCA Collection

The Cancer Genome Atlas Breast Invasive Carcinoma (TCGA-BRCA) collection is particularly relevant for oncology research:

- **Collection**: TCGA-BRCA
- **Size**: 88.13 GB
- **Subjects**: 139 patients
- **Studies**: 164 imaging studies
- **Series**: 1,877 image series
- **Images**: 230,167 DICOM files
- **Modalities**: MR (Magnetic Resonance), MG (Mammography)
- **License**: CC BY 3.0

**Collection URL**: https://www.cancerimagingarchive.net/collection/tcga-brca/

### Matched Genomic Data

TCGA-BRCA images can be matched with genomic and clinical data from:
- [Genomic Data Commons (GDC)](https://portal.gdc.cancer.gov/projects/TCGA-BRCA)
- Patient identifiers are consistent across TCIA and TCGA databases

## Using the NBIA Data Retriever CLI

The NBIA Data Retriever is available in the container at:
```bash
/usr/local/bin/nbia-data-retriever
```

### Step 1: Obtain a Manifest File

Before downloading data, you need to create a manifest file (`.tcia` format) that specifies which images to download.

#### Option A: Download from TCIA Web Portal

1. Visit the [TCIA Radiology Portal](https://nbia.cancerimagingarchive.net/nbia-search/)
2. Search for images by:
   - Collection (e.g., "TCGA-BRCA")
   - Modality (e.g., MR, MG)
   - Body part
   - Other criteria
3. Add desired images to your cart
4. Click **Download > Download Cart** or **Download > Download Query**
5. A `manifest-xxx.tcia` file will be downloaded

#### Option B: Download Entire TCGA-BRCA Collection

For the complete TCGA-BRCA collection:
1. Go to https://www.cancerimagingarchive.net/collection/tcga-brca/
2. Click the **DOWNLOAD** button in the data table
3. Save the manifest file (e.g., `tcga-brca-manifest.tcia`)

### Step 2: Transfer Manifest to Container

If you downloaded the manifest on your host machine, transfer it to the container:

```bash
# From your host, copy to the workspace volume
docker cp manifest-xxx.tcia <container-name>:/workspace/data/

# Or place it in your workspace directory before starting the container
```

### Step 3: Run the NBIA Data Retriever CLI

#### Basic Usage

**Important**: The CLI must be run with `DISPLAY` variable set to use headless mode in the container environment.

```bash
DISPLAY=:0 nbia-data-retriever --manifest /path/to/manifest.tcia \
                               --output /workspace/data/tcia-downloads
```

#### Common Options

```bash
# Download with a specific output directory
DISPLAY=:0 nbia-data-retriever \
  --manifest /workspace/data/tcga-brca-manifest.tcia \
  --output /workspace/data/TCGA-BRCA-images

# Use descriptive directory naming (recommended)
# Organizes as: Collection/Patient/Study/Series
DISPLAY=:0 nbia-data-retriever \
  --manifest /workspace/data/TCIA_TCGA-BRCA_09-16-2015.tcia \
  --output /workspace/data/TCIA_BRCA \
  --directory-format descriptive

# Use classic directory naming (default)
# Organizes as: Collection/Patient/StudyInstanceUID/SeriesInstanceUID
DISPLAY=:0 nbia-data-retriever \
  --manifest /workspace/data/manifest.tcia \
  --output /workspace/data/tcia-downloads \
  --directory-format classic

# Enable checksum verification for data integrity
DISPLAY=:0 nbia-data-retriever \
  --manifest /workspace/data/manifest.tcia \
  --output /workspace/data/tcia-downloads \
  --verify-checksum
```

#### CLI Arguments Reference

| Argument | Description | Default |
|----------|-------------|---------|
| `--manifest` | Path to the manifest (.tcia) file | Required |
| `--output` | Destination directory for downloaded images | Required |
| `--directory-format` | `descriptive` or `classic` directory structure | `classic` |
| `--verify-checksum` | Verify file integrity via checksum | Disabled |
| `--help` | Display help information | - |

### Step 4: Monitor Download Progress

The CLI will display:
- Number of series being downloaded
- Progress for each series
- Download status (completed/failed)
- Any network errors

### Directory Structure

#### Descriptive Format (Recommended)
```
output-directory/
├── TCGA-BRCA/                          # Collection name
│   ├── TCGA-A1-A0SB/                   # Patient ID
│   │   ├── 20010101-StudyDesc-12345/   # Study Date + Description + UID suffix
│   │   │   ├── 1-SeriesDesc-67890/     # Series Number + Description + UID suffix
│   │   │   │   ├── 1-01.dcm
│   │   │   │   ├── 1-02.dcm
│   │   │   │   └── ...
```

#### Classic Format
```
output-directory/
├── TCGA-BRCA/                                      # Collection name
│   ├── TCGA-A1-A0SB/                               # Patient ID
│   │   ├── 1.2.840.113654.2.55.123456789/          # Study Instance UID
│   │   │   ├── 1.2.840.113654.2.55.987654321/      # Series Instance UID
│   │   │   │   ├── 1-01.dcm
│   │   │   │   ├── 1-02.dcm
│   │   │   │   └── ...
```

### File Naming Convention

DICOM files are named based on acquisition and instance numbers:
- Format: `{acquisition}-{instance}.dcm`
- Example: `1-01.dcm`, `1-42.dcm`, `2-01.dcm`
- Numbers are zero-padded for natural ordering
- Files are ordered first by acquisition number, then by instance number

## Example: Downloading TCGA-BRCA Data

```bash
# Step 1: Navigate to your workspace
cd /workspace

# Step 2: Create a directory for TCIA downloads
mkdir -p data/tcia-downloads

# Step 3: Download TCGA-BRCA data (after obtaining manifest)
DISPLAY=:0 nbia-data-retriever \
  --manifest data/tcga-brca-manifest.tcia \
  --output data/tcia-downloads/TCGA-BRCA \
  --directory-format descriptive \
  --verify-checksum

# Step 4: Verify download
ls -lh data/tcia-downloads/TCGA-BRCA
```

## Handling Network Errors

The NBIA Data Retriever automatically retries failed downloads up to 4 times by default. If downloads fail:

1. Check your network connection
2. Verify the manifest file is not corrupted
3. Try running the command again - it will resume from where it left off
4. Contact [TCIA Help Desk](mailto:help@cancerimagingarchive.net) if issues persist

## Working with Downloaded Data

After downloading, you can:

1. **Inspect DICOM headers** using pydicom:
   ```python
   import pydicom
   ds = pydicom.dcmread('path/to/file.dcm')
   print(ds)
   ```

2. **Convert to other formats** (NIfTI, PNG, etc.)
3. **Link with clinical/genomic data** using Patient IDs
4. **Perform image analysis** and radiomics

## Important Notes

### Data Usage Policy

When using TCGA-BRCA or any TCIA data, you must:

1. **Cite the dataset**:
   ```
   Lingle, W., Erickson, B. J., Zuley, M. L., et al. (2016). 
   The Cancer Genome Atlas Breast Invasive Carcinoma Collection (TCGA-BRCA) 
   (Version 3) [Data set]. The Cancer Imaging Archive. 
   https://doi.org/10.7937/K9/TCIA.2016.AB2NAZRP
   ```

2. **Acknowledge TCGA** (if applicable):
   ```
   "The results published here are in whole or part based upon data 
   generated by the TCGA Research Network: http://cancergenome.nih.gov/."
   ```

3. **Follow CC BY 3.0 License**: Attribution required, commercial use allowed

### Date Handling

- **TCIA dates**: DICOM dates are offset by 3-10 years for HIPAA compliance
- **TCGA dates**: Expressed as days from diagnosis (index date = 0)
- Longitudinal intervals are preserved within TCIA

### File Sizes

- TCGA-BRCA: **88.13 GB** total
- Plan for adequate storage space
- Consider downloading subsets if storage is limited

## Additional Resources

- [NBIA Data Retriever Documentation](https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide)
- [TCIA Radiology Portal](https://nbia.cancerimagingarchive.net/nbia-search/)
- [TCGA-BRCA Collection Page](https://www.cancerimagingarchive.net/collection/tcga-brca/)
- [GDC Data Portal (genomics)](https://portal.gdc.cancer.gov/projects/TCGA-BRCA)
- [TCIA Help Desk](http://www.cancerimagingarchive.net/support/)

## Troubleshooting

### "No X11 DISPLAY variable" Error
If you see this error:
```
Exception in thread "main" java.awt.HeadlessException: 
No X11 DISPLAY variable was set
```

**Solution**: Prefix the command with `DISPLAY=:0`:
```bash
DISPLAY=:0 nbia-data-retriever --manifest /path/to/manifest.tcia --output /path/to/output
```

### Command Not Found
If `nbia-data-retriever` is not found, verify the symlink:
```bash
ls -la /usr/local/bin/nbia-data-retriever
# Should point to: /opt/nbia-data-retriever/bin/nbia-data-retriever
```

### Permission Denied
Ensure output directory has write permissions:
```bash
chmod -R 755 /workspace/data/tcia-downloads
```

### Incomplete Downloads
Resume interrupted downloads by running the same command again - the tool remembers partial downloads.

## Contact

For questions about:
- **TCIA data**: Contact [TCIA Help Desk](mailto:help@cancerimagingarchive.net)
- **OncoLearn project**: See project README or repository issues
