# OncoLearn

<img src="./docs/assets/oncoLearn.png" height="200" width="350"/>

![Python](https://img.shields.io/badge/python-3.12%20|%203.13-blue.svg)
![R](https://img.shields.io/badge/R-4.0+-blue.svg)
![uv](https://img.shields.io/badge/uv-package%20manager-green.svg)
![renv](https://img.shields.io/badge/renv-package%20manager-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

OncoLearn is a multimodal machine learning toolkit for cancer genomics analysis and biomarker discovery. It integrates genomic, transcriptomic, clinical, and medical imaging data to enable end-to-end training of cancer classification and subtype prediction models across TCGA cohorts.

> **Full documentation is available on the [OncoLearn Wiki](https://github.com/collaborativebioinformatics/OncoLearn/wiki).**

---

## Key Capabilities

| Capability | Description |
|---|---|
| **Data Acquisition** | Download TCGA genomics data from UCSC Xena Browser, imaging data from TCIA, and clinical/molecular data from cBioPortal via a unified CLI |
| **Multimodal Fusion** | Train models that jointly learn from mRNA expression, clinical features, and MRI/pathology images |
| **Pretrained Encoders** | Leverages IBM's RNA BERT (110M) for gene expression and FM-BCMRI for hierarchical 3D image encoding |
| **Pipeline DSL** | Declare data loading pipelines in plain Python — `Load`, `Join`, `Sequence`, and transform nodes compose into arbitrary multi-source workflows |
| **Hyperparameter Optimisation** | Optuna-based HPO over optimizer, loss, scheduler, and model parameters, with optional cross-validation |

---

## Quickstart

```bash
git clone https://github.com/collaborativebioinformatics/OncoLearn.git
cd OncoLearn
git submodule update --init --recursive

# Start the Docker environment (choose your GPU profile)
docker compose --profile nvidia up -d    # NVIDIA
docker compose --profile amd up -d      # AMD (native Linux)
docker compose --profile amd-wsl up -d  # AMD (WSL2)
```

For platform-specific setup, local installation, and full CLI reference, see the [Wiki](https://github.com/collaborativebioinformatics/OncoLearn/wiki).

---

## Documentation

Comprehensive documentation is available on the **[OncoLearn Wiki](https://github.com/collaborativebioinformatics/OncoLearn/wiki)**:

- **[Getting Started](https://github.com/collaborativebioinformatics/OncoLearn/wiki/Getting-Started-Windows)** — Installation guides for Windows, Linux, and Docker
- **[CLI Reference](https://github.com/collaborativebioinformatics/OncoLearn/wiki/CLI)** — `train`, `preprocess`, `xena`, `tcia`, `cbioportal` subcommands
- **[Modeling](https://github.com/collaborativebioinformatics/OncoLearn/wiki/Modeling-Multimodal)** — Encoder architecture, fusion model, and config reference
- **[Pipeline DSL](https://github.com/collaborativebioinformatics/OncoLearn/wiki/Modeling-Pipeline)** — Declare data loading and transformation pipelines in Python
- **[Training Guide](https://github.com/collaborativebioinformatics/OncoLearn/wiki/Guide-Training)** — Config options, variants, Docker usage, and output format
- **[Python API](https://github.com/collaborativebioinformatics/OncoLearn/wiki/API-Xena-Browser)** — Programmatic access to Xena Browser, TCIA, and cBioPortal

---

## Contributors

Heena Dalal (dalalhina@gmail.com / heena.dalal@kcl.ac.uk), Aryan Sharan Guda (aryanshg@andrew.cmu.edu), Seungjin Han (seungjih@andrew.cmu.edu), Seohyun Lee (seohyun4@andrew.cmu.edu), Yosen Lin (yosenl@andrew.cmu.edu), Isha Parikh (parikh.i@northeastern.edu), Diya Patidar (dpatidar@andrew.cmu.edu), Arunannamalai Sujatha Bharath Raj (asujatha@andrew.cmu.edu), Andrew Scouten (yzb2@txstate.edu), Jeffrey Wang (jdw2@andrew.cmu.edu), Qiyu (Charlie) Yang (qiyuy@andrew.cmu.edu), Zhaoyi (Zoey) You (zhaoyiyou.zoey@gmail.com), Xinru Zhang (mayzxr2203@gmail.com), River Zhu (riverz@andrew.cmu.edu)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## AI Disclosure

Artificial intelligence tools, including large language models (LLMs), were used during the development of this project to support writing, clarify technical concepts, and assist in generating code snippets. These tools served as an aid for idea refinement, debugging, and improving the readability of explanations and documentation. All AI-generated text and code were thoroughly reviewed, verified for correctness, and understood in full before being incorporated into this work. The responsibility for all final decisions, interpretations, and implementations remains solely with the contributors.
