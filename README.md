# Cancer Biomarker Discovery

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![R](https://img.shields.io/badge/R-4.0+-blue.svg)
![uv](https://img.shields.io/badge/uv-package%20manager-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Authors:** River Zhu, Zoey, Arun, Charlie Yang, Diya, Ana Ferreira (bossanblues@gmail.com), Xinru Zhang, Isha Parikh, Aryan Sharan Guda, Seohyun, Yosen Lin (yosenl@andrew.cmu.edu), Andrew Scouten (yzb2@txstate.edu)

## Getting Started

### Prerequisites

This project requires the following dependencies:
- Python 3.12+
- R 4.0+
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

### Installation

1. **Install uv** (if not already installed) from [here](https://docs.astral.sh/uv/getting-started/installation/).

2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Cancer_biomarker_discovery
   ```

3. **Install LLVM** (required for some Python packages):

   - **Linux** (Ubuntu/Debian):
     ```bash
     sudo apt-get update
     sudo apt-get install llvm
     ```

   - **macOS**:
     ```bash
     brew install llvm
     ```

4. **Install Python dependencies**:
   ```bash
   # Install base dependencies
   uv sync

   # Or install with PyTorch extras (for GPU/ML features)
   uv sync --extra pytorch
   ```

5. **Install R dependencies with renv**:
   ```r
   # Install renv if not already installed
   install.packages("renv")

   # Restore R package dependencies
   renv::restore()
   ```

### Recommended VSCode Extensions

For the best development experience, we recommend installing the following VSCode extensions:

- **[Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)** (`ms-python.python`) - IntelliSense, debugging, and linting for Python
- **[Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)** (`charliermarsh.ruff`) - Fast Python linter and formatter
- **[autopep8](https://marketplace.visualstudio.com/items?itemName=ms-python.autopep8)** (`ms-python.autopep8`) - Python code formatter following PEP 8 style guide
- **[R](https://marketplace.visualstudio.com/items?itemName=REditorSupport.r)** (`REditorSupport.r`) - R language support with syntax highlighting and code execution
- **[Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)** (`ms-toolsai.jupyter`) - Interactive Jupyter notebook support

## Flowchart

![Flowchart](./docs/assets/flowchart.png)

## AI Disclosure

Artificial intelligence tools, including large language models (LLMs), were used during the development of this project to support writing, clarify technical concepts, and assist in generating code snippets. These tools served as an aid for idea refinement, debugging, and improving the readability of explanations and documentation. Importantly, all AI-generated text and code were thoroughly reviewed, verified for correctness, and understood in full before being incorporated into this work. No content was used without careful evaluation, and the responsibility for all final decisions, interpretations, and implementations remains solely with the authors.