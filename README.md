# OncoLearn

<img src="./docs/assets/oncoLearn.png" height="200" width="350"/>

![Python](https://img.shields.io/badge/python-3.12%20|%203.13-blue.svg)
![R](https://img.shields.io/badge/R-4.0+-blue.svg)
![uv](https://img.shields.io/badge/uv-package%20manager-green.svg)
![renv](https://img.shields.io/badge/renv-package%20manager-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive toolkit for cancer genomics analysis and biomarker discovery using RNA-seq data from The Cancer Genome Atlas (TCGA). OncoLearn leverages machine learning and statistical methods for cancer subtyping and identifying potential diagnostic and prognostic markers.

## Contributors

Aryan Sharan Guda (aryanshg@andrew.cmu.edu), Seungjin Han (seungjih@andrew.cmu.edu), Seohyun Lee (seohyun4@andrew.cmu.edu), Yosen Lin (yosenl@andrew.cmu.edu), Isha Parikh (parikh.i@northeastern.edu), Diya Patidar (dpatidar@andrew.cmu.edu), Arunannamalai Sujatha Bharath Raj (asujatha@andrew.cmu.edu), Andrew Scouten (yzb2@txstate.edu), Jeffrey Wang (jdw2@andrew.cmu.edu), Qiyu (Charlie) Yang (qiyuy@andrew.cmu.edu), Xinru Zhang, River Zhu (riverz@andrew.cmu.edu), Zhaoyi (Zoey) You (zhaoyiyou.zoey@gmail.com)

## Table of Contents

- [Quickstart](#quickstart)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Option A: Docker Installation](#option-a-docker-installation-recommended)
  - [Option B: Local Installation](#option-b-local-installation)
  - [Recommended VSCode Extensions](#recommended-vscode-extensions)
- [Data](#data)
  - [Download from Xena Browser](#download-from-xena-browser-genomics-data)
  - [Download from TCIA](#download-from-tcia-imaging-data)
- [Documentation](#documentation)
- [License](#license)
- [AI Disclosure](#ai-disclosure)

## Quickstart

1. **Install Docker Desktop** from [docker.com](https://www.docker.com/products/docker-desktop/)

2. **Clone and setup**:
   ```bash
   git clone https://github.com/collaborativebioinformatics/OncoLearn.git
   cd OncoLearn
   git submodule update --init --recursive
   # For NVIDIA GPUs:
   docker compose --profile nvidia up -d
   # For AMD GPUs (native Linux):
   docker compose --profile amd up -d
   # For AMD GPUs (WSL2):
   docker compose --profile amd-wsl up -d
   ```

3. **Download sample data**:
   ```bash
   # Download genomics data from Xena Browser
   docker compose exec dev uv run scripts/download.py --xena --cohorts BRCA
   
   # Download imaging data from TCIA (manifest only)
   docker compose exec dev uv run scripts/download.py --tcia --cohorts BRCA
   
   # Download imaging data from TCIA (manifest + images)
   docker compose exec dev uv run scripts/download.py --tcia --cohorts BRCA --download-images
   ```

4. **Start exploring** with the Jupyter notebooks in [`notebooks/data/`](notebooks/data/)

For detailed setup options and local installation, see [Getting Started](#getting-started).

## Getting Started

### Prerequisites

This project supports two installation methods:

**Option A: Docker (Recommended)**
- Docker Desktop or Docker Engine
- Docker Compose
- VSCode with Dev Containers extension (optional but recommended)

**Option B: Local Installation**
- Python 3.10+
- R 4.0+
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

---

### Option A: Docker Installation (Recommended)

Docker provides a consistent development environment and eliminates dependency and compatibility issues.

1. **Install Docker Desktop**:
   - Download from [docker.com](https://www.docker.com/products/docker-desktop/)
   - Or install Docker Engine on Linux

2. **Clone the repository**:
   ```bash
   git clone https://github.com/collaborativebioinformatics/OncoLearn.git
   cd OncoLearn
   git submodule update --init --recursive
   ```

3. **Start the environment**:
   ```bash
   # For NVIDIA GPUs:
   docker compose --profile nvidia up -d
   
   # For AMD GPUs (native Linux):
   docker compose --profile amd up -d
   
   # For AMD GPUs (WSL2 on Windows):
   docker compose --profile amd-wsl up -d
   ```
   
   > **Note**: The Docker setup includes GPU support for both NVIDIA and AMD GPUs. Choose the appropriate profile based on your hardware:
   > - `nvidia`: For NVIDIA GPUs
   > - `amd`: For AMD GPUs on native Linux
   > - `amd-wsl`: For AMD GPUs on Windows Subsystem for Linux 2 (WSL2)

4. **Open in VSCode Dev Container** (optional):
   - Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
   - Press `F1` → "Dev Containers: Reopen in Container"
   - VSCode will connect to the container with all extensions and tools configured
   - Jupyter notebooks (`.ipynb` files) will work natively in VSCode without a browser

**Useful Docker Commands**:
```bash
# Stop containers
docker compose down

# Rebuild after dependency changes (use your GPU profile)
docker compose --profile nvidia build  # or --profile amd or --profile amd-wsl
docker compose --profile nvidia up -d  # or --profile amd or --profile amd-wsl

# Execute commands in container
docker compose exec dev bash  # NVIDIA
docker compose exec dev-amd bash  # AMD (native Linux)
docker compose exec dev-amd-wsl bash  # AMD (WSL2)

# Add new Python packages
docker compose exec dev uv add <package-name>  # NVIDIA
docker compose exec dev-amd uv add <package-name>  # AMD (native Linux)
docker compose exec dev-amd-wsl uv add <package-name>  # AMD (WSL2)

# View running containers
docker compose ps
```

---

### Option B: Local Installation

1. **Install uv** (if not already installed) from [here](https://docs.astral.sh/uv/getting-started/installation/).

2. **Clone the repository**:
   ```bash
   git clone https://github.com/collaborativebioinformatics/OncoLearn.git
   cd OncoLearn
   git submodule update --init --recursive
   ```

3. **Install Python dependencies**:
   ```bash
   # Install base dependencies
   uv sync

   # Or install with PyTorch extras (choose one based on your hardware):
   uv sync --extra cpu          # CPU-only version
   uv sync --extra cu128        # CUDA 12.8
   uv sync --extra cu130        # CUDA 13.0
   uv sync --extra rocm         # AMD ROCm
   ```

4. **Install R dependencies with renv**:
   ```r
   # Install renv if not already installed
   install.packages("renv")

   # Restore R package dependencies
   renv::restore()
   ```

---

### Recommended VSCode Extensions

For the best development experience, we recommend installing the following VSCode extensions:

- **[Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)** (`ms-python.python`) - IntelliSense, debugging, and linting for Python
- **[Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)** (`charliermarsh.ruff`) - Fast Python linter and formatter
- **[autopep8](https://marketplace.visualstudio.com/items?itemName=ms-python.autopep8)** (`ms-python.autopep8`) - Python code formatter following PEP 8 style guide
- **[R](https://marketplace.visualstudio.com/items?itemName=REditorSupport.r)** (`REditorSupport.r`) - R language support with syntax highlighting and code execution
- **[Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)** (`ms-toolsai.jupyter`) - Interactive Jupyter notebook support
- **[Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)** (`ms-vscode-remote.remote-containers`) - For Docker development (if using Docker)

---

## Data

OncoLearn provides a unified download script for acquiring cancer data from multiple sources:

### Download from Xena Browser (Genomics Data)

```bash
# Download a single cohort (all data types)
uv run scripts/download.py --xena --cohorts BRCA

# Download specific data category
uv run scripts/download.py --xena --cohorts BRCA --category mutation

# Download multiple cohorts
uv run scripts/download.py --xena --cohorts BRCA,LUAD,ACC

# Download all available cohorts
uv run scripts/download.py --xena --all

# List available cohorts
uv run scripts/download.py --xena --list
```

**Available categories:** `clinical`, `mutation`, `cnv`, `mrna`, `mirna`, `protein`, `methylation`

#### Download from TCIA (Imaging Data)

```bash
# Download manifest file only
uv run scripts/download.py --tcia --cohorts BRCA

# Download manifest and images (requires nbia-data-retriever)
uv run scripts/download.py --tcia --cohorts BRCA --download-images

# Download multiple cohorts with images
uv run scripts/download.py --tcia --cohorts BRCA,LUAD --download-images

# List available cohorts
uv run scripts/download.py --tcia --list
```

**Note:** The `--download-images` flag requires the [nbia-data-retriever](https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide) tool to be installed.

### Docker Usage

When using Docker, prefix commands with the container execution:

```bash
# NVIDIA GPU container
docker compose exec dev uv run scripts/download.py --xena --cohorts BRCA

# AMD GPU container
docker compose exec dev-amd uv run scripts/download.py --tcia --cohorts BRCA --download-images
```

---

## Documentation

Comprehensive guides and documentation are available in the [`docs/`](docs/) folder:

- **[TCGA Data Download Guide](docs/TCGA_Data_Download_Guide.md)** - Detailed instructions for downloading and managing TCGA datasets
- **[TCIA Data Download Guide](docs/TCIA_Data_Download_Guide.md)** - Guide for downloading imaging data from TCIA
- **[GitHub Authentication Setup](docs/GitHub_Authentication_Guide.md)** - Configure SSH authentication for GitHub access
- **[Models Documentation](docs/Models.md)** - Overview of machine learning models and architectures

### Project Structure

- **`data/`** - Data storage directory (downloaded TCGA datasets)
- **`docs/`** - Project documentation and guides
- **`notebooks/`** - Jupyter notebooks for data exploration and analysis
- **`scripts/`** - Data download and preprocessing scripts
  - **`download.py`** - Unified CLI for downloading data from Xena Browser and TCIA
  - **`data/download_xena.py`** - Xena Browser download utilities
  - **`data/download_tcia.py`** - TCIA download utilities
- **`src/oncolearn/`** - Core Python package for cancer genomics analysis
- **`src/multimodal/`** - Multimodal learning framework for integrating multi-omic data
- **`configs/`** - Configuration files for training and testing

### Additional Resources

For more information on downloading and working with TCGA data, see the [TCGA Data Download Guide](docs/TCGA_Data_Download_Guide.md).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## AI Disclosure

Artificial intelligence tools, including large language models (LLMs), were used during the development of this project to support writing, clarify technical concepts, and assist in generating code snippets. These tools served as an aid for idea refinement, debugging, and improving the readability of explanations and documentation. All AI-generated text and code were thoroughly reviewed, verified for correctness, and understood in full before being incorporated into this work. The responsibility for all final decisions, interpretations, and implementations remains solely with the contributors.
