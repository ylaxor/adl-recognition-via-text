# Towards Domain-Robust Activity Recognition using Textual Representations of Binary Sensor Events

Language-based representations have recently emerged as a promising approach for cross-domain Human Activity Recognition (HAR) in smart homes, where binary sensor streams are verbalized into natural-language descriptions processed by pretrained encoders. However, prior work has typically fixed both the textualization scheme and the embedding model, leaving open how linguistic design choices influence transferability.

This project presents a comprehensive factorial analysis of textualization and embedding strategies for language-based HAR. We systematically vary (i) how sensor event windows are expressed—across seven sequential and summarized textualizations—and (ii) how they are embedded using lexical (TF–IDF), static (Word2Vec), and contextual (SBERT) encoders. Experiments on four public smart-home datasets under consistent in-domain and cross-domain transfer conditions reveal that textualization design, not encoder complexity, governs performance. Sequential, event-ordered sentences maximize in-domain accuracy, while single-sentence, schema-based summaries—such as the proposed Compound Sensor Summary (CSS)—generalize best across homes.

Our findings establish a reproducible framework for analyzing and designing language-based representations in HAR, demonstrating that linguistic structure—rather than deep contextualization—is the primary determinant of domain robustness in smart-home activity recognition.

## Requirements

### Python Version
- Python >= 3.12

### Dependencies
All dependencies are specified in `pyproject.toml`. Key packages include:
- PyTorch (< 2.7)
- Lightning
- Sentence Transformers
- Scikit-learn
- Gensim
- Streamlit
- Matplotlib, Seaborn
- And more (see `pyproject.toml` for complete list)

## Setup

### Recommended: Using `uv`

The recommended way to set up this project is using [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver.

1. Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Linux/macOS
uv pip install -e .
```

### Alternative: Using `pip`

If you prefer using pip:

```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
pip install -e .
```

## Usage

The project provides two main modes of operation through `main.py`:

### 1. Explore Mode

Launch an interactive Streamlit dashboard to explore datasets and visualizations:

```bash
python main.py --mode explore
```

This will start a Streamlit application for:
- Exploring activity data
- Exploring sensor event data and metadata
- Analyzing seven window textualization methods
- Experimenting with vectorization methods (TF-IDF, Word2Vec, SBERT) and visualizing UMAP projections

### 2. Run Experiment Mode

Execute the full experimental pipeline:

```bash
python main.py --mode runxp
```

This runs the complete experiment workflow as configured in `config/experiment.toml`.

## Project Structure

### Dataset Files

All datasets are located in the `testbed/` directory with the following structure:

```
testbed/
├── casas-aruba/
│   ├── activities.csv      # Activity annotations
│   ├── sensors.csv          # Sensor event data
│   └── meta.toml            # Dataset metadata (sensors, activities)
├── casas-milan/
│   ├── activities.csv
│   ├── sensors.csv
│   └── meta.toml
├── ordonez-a/
│   ├── activities.csv
│   ├── sensors.csv
│   └── meta.toml
└── ordonez-b/
    ├── activities.csv
    ├── sensors.csv
    └── meta.toml
```

Each dataset directory contains:
- **`activities.csv`**: Ground truth activity labels with timestamps
- **`sensors.csv`**: Raw sensor event data
- **`meta.toml`**: Metadata including sorted sensor names and activity labels

### Configuration Files

Configuration files are located in the `config/` directory:

- **`config/data.toml`**: Activity definitions for each testbed
- **`config/experiment.toml`**: Experiment parameters including:
  - Window parameters (`nb_days`, `pad_trunc_length`)
  - Vectorization settings (TF-IDF, Word2Vec, SBERT)
  - Model hyperparameters (MLP, LSTM)
  - Cross-validation settings (`n_splits`)
- **`config/mailing.toml`**: Email notification settings (not included in repository)

#### Setting up Email Notifications

To enable email notifications for experiment completion, create a `config/mailing.toml` file with the following structure:

```toml
[email]
sender = "your-email@gmail.com"
recipient = "recipient-email@gmail.com"
server = "smtp.gmail.com"
port = 587
secret = "your-app-password-here"
```

**Note:** For Gmail, you'll need to generate an [App Password](https://support.google.com/accounts/answer/185833) rather than using your regular password.

### Output Paths

All outputs are stored in the `output/` directory:

```
output/
├── tidy_results.csv         # Experiment results in tidy format
├── figures/                 # Generated plots and visualizations
└── paper/                   # Paper-ready assets
```

- **`output/tidy_results.csv`**: Main results file containing experimental outcomes
- **`output/figures/`**: Visualizations and plots generated during analysis
- **`output/paper/`**: Publication-ready figures and tables

### Imported Results

Pre-computed results are available in the `imported/` directory:

```
imported/
├── tidy_results.csv          # Main experiment results
└── tidy_results_ablation.csv # Ablation study results
```

## Scripts

The `scripts/` directory contains Jupyter notebooks for analysis and visualization:

- **`scripts/paper_assets.ipynb`**: Generates figures and tables for publications
- **`scripts/xp_runner_usage_demo.ipynb`**: Demonstrates how to run experiments programmatically

To use the notebooks:

```bash
jupyter notebook scripts/
```

## Codebase Modules

The core functionality is organized in the `codebase/` package:

- **`domain.py`**: Core domain models and enums (TestBed definitions)
- **`loading.py`**: Dataset loading utilities
- **`preprocessing.py`**: Data preprocessing functions
- **`windowing.py`**: Sliding window generation
- **`textualization.py`**: Window-to-text conversion
- **`vectorization.py`**: Text-to-vector encoding (TF-IDF, Word2Vec, SBERT)
- **`lstm.py`**: LSTM model implementation
- **`experiment.py`**: Experiment orchestration
- **`plotting.py`**: Visualization utilities
- **`reporting.py`**: Results reporting
- **`mailing.py`**: Email notification system

## Quick Start Example

1. Set up the environment:
```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

2. Explore the data interactively:
```bash
python main.py --mode explore
```

3. Run an experiment:
```bash
python main.py --mode runxp
```

or use the notebook to run custom experiments:
```bash
jupyter notebook scripts/xp_runner_usage_demo.ipynb
```

4. Analyze results:
```bash
jupyter notebook scripts/paper_assets.ipynb
```

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{flowar-textual2025,
  title={Revisiting Textual Representations for Domain-Robust Human Activity
Recognition in Smart Homes},
  author={XXXXX},
  journal={Journal Name},
  year={2025},
  volume={XX},
  pages={XXX--XXX},
  doi={10.XXXX/XXXXX}
}
```

Or in plain text:

> XXXX, & XXXXX (2025). Towards Domain-Robust Activity Recognition using Textual
Representations of Binary Sensor Events. *Name*, XX(X), XXX-XXX. https://doi.org/10.XXXX/XXXXX

## License

See LICENSE file for details.
