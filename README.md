# OncoLearn

<img src="./docs/assets/oncoLearn.png" height="200" width="350"/>

![Python](https://img.shields.io/badge/python-3.12%20|%203.13-blue.svg)
![R](https://img.shields.io/badge/R-4.0+-blue.svg)
![uv](https://img.shields.io/badge/uv-package%20manager-green.svg)
![renv](https://img.shields.io/badge/renv-package%20manager-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive toolkit for cancer genomics analysis and biomarker discovery using RNA-seq data from The Cancer Genome Atlas (TCGA). OncoLearn leverages machine learning and statistical methods for cancer subtyping and identifying potential diagnostic and prognostic markers.

## Contributors

Aryan Sharan Guda (aryanshg@andrew.cmu.edu), Seungjin Han (seungjih@andrew.cmu.edu), Seohyun Lee (seohyun4@andrew.cmu.edu), Yosen Lin (yosenl@andrew.cmu.edu), Isha Parikh (parikh.i@northeastern.edu), Diya Patidar (dpatidar@andrew.cmu.edu), Arunannamalai Sujatha Bharath Raj (asujatha@andrew.cmu.edu), Andrew Scouten (yzb2@txstate.edu), Jeffrey Wang (jdw2@andrew.cmu.edu), Qiyu (Charlie) Yang (qiyuy@andrew.cmu.edu), Xinru Zhang (mayzxr2203@gmail.com), River Zhu (riverz@andrew.cmu.edu), Zhaoyi (Zoey) You (zhaoyiyou.zoey@gmail.com), Heena Dalal (dalalhina@gmail.com/heena.dalal@kcl.ac.uk)

## Table of Contents

- [Quickstart](#quickstart)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Option A: Docker Installation](#option-a-docker-installation-recommended)
  - [Option B: Local Installation](#option-b-local-installation)
  - [Recommended VSCode Extensions](#recommended-vscode-extensions)
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
   docker compose up -d
   ```

3. **Download sample data**:
   ```bash
   docker compose exec dev bash ./scripts/data/download_tcga_brca.sh
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
   # Build and start the container
   docker compose up -d
   ```

4. **Open in VSCode Dev Container** (optional):
   - Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
   - Press `F1` → "Dev Containers: Reopen in Container"
   - VSCode will connect to the container with all extensions and tools configured
   - Jupyter notebooks (`.ipynb` files) will work natively in VSCode without a browser

**Useful Docker Commands**:
```bash
# Stop containers
docker compose down

# Rebuild after dependency changes
docker compose build

# Execute commands in container
docker compose exec dev bash

# Add new Python packages
docker compose exec dev uv add <package-name>

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
