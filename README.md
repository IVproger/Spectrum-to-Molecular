# Molecular Structure Prediction from Spectral Data  
*Hybrid AI Methods for Automated Analysis*

Welcome to our research project repository. This project aims to develop an **interpretable AI system** that integrates deep learning methods—including Graph Neural Networks (GNNs), diffusion models, and transformer architectures—with classical chemical analysis. Our objective is to automatically reconstruct molecular structures from mass spectrometry data and generate reaction mechanism hypotheses.

## Project Overview

**Key Goals:**

- **Automated Structure Reconstruction:**  
  Develop algorithms to predict molecular structures from mass spectra with high accuracy (≥85% Tanimoto similarity).

- **Reaction Mechanism Hypothesis:**  
  Generate and visualize fragmentation trees to propose reaction mechanisms.

- **Hybrid Methodologies:**  
  Combine:
  - **Graph Neural Networks (GNNs):** To model chemical bonds and molecular fragmentation.
  - **Diffusion Models:** To generate molecular structures under chemical constraints.  
    The *DiffMS* module—integrated from the Coley Group’s work—is a core component in this approach [1].
  - **Transformer Architectures:** To analyze complex spectral patterns.

- **Integration with Analytical Tools:**  
  Enhance processing speed and precision by integrating with established platforms (e.g., RMG-Py, SIRIUS).

## Project Components

### 1. Data Preparation  
- **Data Collection:**  
  Aggregate and annotate spectral datasets from open sources (MONA, HMDB, NIST, GNPS) including SMILE structures and metadata.  
- **Preprocessing:**  
  Implement pipelines to clean, normalize, and structure spectral data for downstream analysis.

### 2. Algorithm Development  
- **Initial Model Development:**  
  Start with an initial molecular structure prediction model.  
- **Hybrid Architecture:**  
  Enhance the initial model by integrating GNNs and diffusion models (via the DiffMS module) to create a robust and interpretable system.

### 3. Validation & Integration  
- **Evaluation:**  
  Test models using metrics such as Tanimoto similarity, MCES, and functional group prediction accuracy.
- **Tool Integration:**  
  Link our system with analytical tools (e.g., RMG-Py, SIRIUS) for improved performance.

## Literature Review

We maintain a comprehensive literature review where all relevant research papers are collected and discussed. You can access the full document [here](https://docs.google.com/document/d/1_v_1drTmpsYIzIAhz48yNV_ucPfhLEJYD8sfS7xAWLM/edit?tab=t.0#heading=h.4sb3482urv5).

## Repository Structure

```
├── data/               # Spectral data files, SMILE structures, and metadata
├── notebooks/          # Jupyter notebooks for experiments and analysis
├── scripts/            # Data preprocessing, initial prediction model, and training pipelines
├── services/           # Backend/API services for project deployment and integration
├── src/                # Main source code, including the integrated DiffMS module for diffusion-based predictions
├── LICENSE             # Project license
├── pyproject.toml      # Project configuration managed by Poetry
├── poetry.lock         # Dependency lock file generated by Poetry
└── README.md           # This file
```


## Installation

### Prerequisites

- **Python:** 3.10 or later  
- **PyTorch:** 2.0 or later  
- **Poetry:** for dependency management (installation instructions available on the [Poetry website](https://python-poetry.org/))

### Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-repo/project.git](https://github.com/IVproger/Spectrum-to-Molecular.git
   cd Spectrum-to-Molecular
   ```

2. **Install Dependencies with Poetry:**

   ```bash
   poetry install
   ```

## Citation & Acknowledgements

- **DiffMS Module:**  
  [1] M. Bohde, M. Manjrekar, R. Wang, S. Ji, and C. W. Coley, "DiffMS: Diffusion Generation of Molecules Conditioned on Mass Spectra," arXiv:2502.09571, 2025, [Online]. Available: https://arxiv.org/abs/2502.09571.

Our research framework and methodology were also inspired by the comprehensive technical guidelines outlined in our internal documentation.

## Project Timeline & Team

**Timeline:**

- **2025:**  
  - Data collection and preprocessing.  
  - Implementation of the initial prediction model.  
  - Publication of initial findings.

- **2026:**  
  - Development of the hybrid architecture (GNN + Diffusion Models).  
  - Presentation at NeurIPS.

- **2027:**  
  - Model optimization and testing on real-world data.  
  - Final publications and industrial collaborations.

## Contributing

We welcome contributions and collaboration from all team members. Please follow these steps:

1. **Create a Feature Branch:**  
   Use a descriptive branch name for your changes.
   
2. **Commit Changes:**  
   Ensure your commit messages are clear and detailed.

3. **Open a Pull Request:**  
   Submit your changes for review before merging.

For any questions or further details, please reach out via our team communication channels.

