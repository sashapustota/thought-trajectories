# ThoughtMiner: Extracting and Analyzing Reasoning Trajectories in Language Models

*Aleksandrs Baskakovs*
*Master Thesis*  
*Aarhus University, Cognitive Science MSc.*

---

## Overview

**ThoughtMiner** is a toolkit for extracting, analyzing, and modeling reasoning trajectories from Large Language Models (LLMs) on math and logic problems.  
It enables you to visualize hidden state trajectories, analyze patterns of correct and incorrect reasoning, and train custom reward models for reasoning quality—all with reproducible, modern tooling.

The core of the project is the `ThoughtMiner` class (see `thoughtminers/miners.py`), which efficiently extracts hidden state vectors step-by-step from LLM completions. The included Jupyter notebooks demonstrate end-to-end workflows: from raw generation to PCA visualization, dataset construction, and LSTM-based reward modeling.

---

## Example Visualization

A typical output from this pipeline is a 3D PCA plot showing the trajectories of model "thoughts" as they reason through math problems. Trajectories from correct and incorrect answers are color-coded for interpretability:

![3D PCA of Reasoning Trajectories](plots/pca_trajectories_3d.png)

---

## Repository Structure

thoughtminers/
│
├── thoughtminers/
│ ├── init.py # Package init; exposes ThoughtMiner class
│ └── miners.py # The ThoughtMiner class for trajectory extraction
│
├── notebooks/
│ └── main_analysis.ipynb # Example and workflow notebook
│
├── plots/
│ └── pca_trajectories_3d.png # Example plot
│
├── pyproject.toml # Modern dependency and metadata management
├── README.md # This file
├── .gitignore # Exclude venv, checkpoints, etc.
└── uv.lock # (Optional) Lockfile for deterministic installs with uv

---

## Getting Started

### Prerequisites

- Python 3.9 or higher (Python 3.13+ recommended)
- [uv](https://github.com/astral-sh/uv) (for fast, modern dependency management)
    ```bash
    pip install uv
    ```

---

### Setup Instructions

1. **Clone the repository:**
    ```bash
    git clone https://github.com/sashapustota/thoughtminers.git
    cd thoughtminers
    ```

2. **Create a virtual environment:**
    ```bash
    uv venv
    ```

3. **Install dependencies and package in editable mode:**
    ```bash
    uv pip install -e .
    ```

4. **(Optional) Register Jupyter kernel for notebooks:**
    ```bash
    uv run ipython kernel install --user --env VIRTUAL_ENV=$(pwd)/.venv --name=thoughtminers
    ```

5. **Launch Jupyter and open the main notebook:**
    ```bash
    uv run jupyter notebook
    ```
    - Select the `thoughtminers` kernel when running the notebook.

---

## Usage

- **Trajectory Extraction:**  
  Use `ThoughtMiner` to extract hidden state vectors from LLM completions, grouped by reasoning steps.

- **Visualization:**  
  The provided notebook walks through visualizing these trajectories with PCA, including color-coding for correctness.

- **Reward Modeling:**  
  Example code shows how to build a dataset of correct/incorrect trajectories, preprocess for fixed-length LSTM input, and train a reward model.

---

## Key Features

- **Reproducible environment**: All dependencies specified in `pyproject.toml` (works out of the box with `uv`).
- **Modern workflow**: Pip-installable package, editable mode, and clean imports.
- **Flexible analysis**: Plug-and-play with other LLMs and datasets.
- **Jupyter-first**: Notebooks for exploratory analysis, publication-ready plots, and easy code customization.

---

## License

Apache-2.0 license. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

Developed by Aleksandrs Baskakovs at Aarhus University, Cognitive Science MSc.

---