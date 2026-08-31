# spaGFM: a scalable graph foundation model for spatial transcriptomics analyses
[![Python](https://img.shields.io/badge/Python_3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch_2.10+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/github-spaGFM_1.0-blue?logo=github&logoColor=white)](https://github.com/yzhong36/spaGFM)

Foundation models offer a promising paradigm for modeling spatial transcriptomics, but capturing tissue context over cellular graphs makes training at scale challenging. We introduce spaGFM, a graph foundation model that serializes cellular neighborhoods through random walks to generate transformer-compatible representations of tissue organization. By replacing classical graph message passing with augmented self-supervised neighborhood reconstruction, spaGFM captures higher-order spatial context while allowing scaling at the atlas level. Pretrained on 43.9 million cells from 132 image-based spatial transcriptomics datasets, spaGFM produces robust representations at both the neighborhood and cell level that transfer across tissues, disease contexts, and spatial technologies. SpaGFM identifies tertiary lymphoid structures across cancer cohorts and spatial platforms, captures cell-level transcriptomic perturbation responses associated with T cell vicinity in spatial CRISPR experiments, and characterizes glomerular organization associated with pathological grade in diabetic kidney disease biopsies. Overall, spaGFM establishes a graph foundation-model framework for learning cellular organization and its functional consequences.


![Alt text](/fig/model.png)

# Installation
**Support platform**: this code is tested on Linux (RHEL 8/9; SLES 15 SP6). We highly recommend using a plaform with GPU for model evaluation.

Clone the repository with submodules so that the external dependencies under
`scFM/` are checked out:

```bash
git clone --recurse-submodules https://github.com/yzhong36/spaGFM.git
cd spaGFM
```

Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate spaGFM
```

This installs `python=3.10`, `torch==2.10.0` for CUDA 12.6, and the
required runtime dependencies for all modules.

Finally, install spaGFM from the repository root:

```bash
pip install -e .
```

# Basic Usage
Please refer to the [tutorials](./tutorials) direcory for basic usage of spaGFM.

# Acknowledgements
We thank the authors of [scGPT](https://github.com/bowang-lab/scGPT) and [scConcept](https://github.com/theislab/scConcept) for their open-source codebases, which we adapted for our work. We also thank the authors of [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) for their excellent library, which we used for model training and evaluation. 
