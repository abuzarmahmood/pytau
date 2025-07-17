<div align="center">
  <img src="docs/pytau_logo.png" alt="PyTau Logo" width="200"/>
  <h1>PyTau</h1>
  <p><strong>Powerful Changepoint Detection for Neural Data</strong></p>

  [![status](https://joss.theoj.org/papers/e3e3d9ce5b59166cef17ee7e9bb9f53c/status.svg)](https://joss.theoj.org/papers/e3e3d9ce5b59166cef17ee7e9bb9f53c)
  [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/abuzarmahmood/pytau/master.svg)](https://results.pre-commit.ci/latest/github/abuzarmahmood/pytau/master)
  [![Pytest](https://github.com/abuzarmahmood/pytau/actions/workflows/pytest_workflow.yml/badge.svg)](https://github.com/abuzarmahmood/pytau/actions/workflows/pytest_workflow.yml)
</div>

## 🚀 What is PyTau?

PyTau is a specialized Python package for detecting state changes in neural data using Bayesian changepoint models. It provides:

- **Streamlined batch inference** on PyMC3-based changepoint models
- **Robust detection** of state transitions in neural firing patterns
- **Comprehensive visualization tools** for model results

## 📚 Documentation

[**Full API Documentation**](https://abuzarmahmood.github.io/pytau/)

## ⚡ Quick Start

```bash
# Create and activate conda environment
conda create -n "pytau_env" python=3.6.13 ipython notebook -y
conda activate pytau_env

# Clone repository
git clone https://github.com/abuzarmahmood/pytau.git

# Install requirements
cd pytau
pip install -r requirements.txt

# Download test data and run example notebook
cd pytau/how_to
bash scripts/download_test_data.sh
cd notebooks
jupyter notebook
```

## 🧠 Key Features

- **Multiple Model Types**:
  - Single-taste Poisson models
  - All-taste hierarchical models
  - Dirichlet process models for automatic state detection
  - Variable sigmoid models for flexible transition shapes

- **Flexible Data Handling**:
  - Support for shuffled, simulated, and actual neural data
  - Comprehensive preprocessing pipeline
  - Automated model selection

- **Powerful Analysis Tools**:
  - State transition detection
  - Cross-region correlation analysis
  - Statistical significance testing

## 📊 Data Organization

PyTau uses a centralized database approach for model management:

- **Centralized Storage**: Models stored in a central location with metadata
- **Comprehensive Tracking**: Each model includes:
  - Animal and session information
  - Region and taste details
  - Model parameters and preprocessing steps
  - Fit statistics and results

## 🔄 Pipeline Architecture

PyTau's modular pipeline ensures reproducible analysis:

### 1️⃣ Model Generation
- **Purpose**: Creates and fits Bayesian changepoint models
- **Components**: Various model types for different neural data structures
- **Input**: Processed spike trains and model parameters
- **Output**: Fitted model with posterior samples

### 2️⃣ Data Preprocessing
- **Purpose**: Prepares raw neural data for modeling
- **Features**: Handles various data transformations (shuffling, simulation)
- **Input**: Raw spike trains with parameters for processing
- **Output**: Binned, processed spike count data

### 3️⃣ I/O Management
- **Purpose**: Handles data loading, model storage, and database management
- **Features**: Automatic model retrieval or fitting based on parameters
- **Operations**: HDF5 data loading, metadata tracking, database integration

### 4️⃣ Batch Processing
- **Purpose**: Enables large-scale model fitting across datasets
- **Features**: Parameter iteration and parallel processing

## 💻 Advanced Usage

### Parallelization
PyTau supports parallel processing using GNU Parallel with isolated Theano compilation directories to prevent clashes. See [single_process.sh](https://github.com/abuzarmahmood/pytau/blob/master/pytau/utils/batch_utils/single_process.sh) and [this implementation](https://github.com/abuzarmahmood/pytau/pull/19/commits/231dd33b846cf278549b1b5815fdae5e76fa14a2).

## 🤝 Contributing

We welcome contributions to PyTau! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to the project.

## 📜 Citation

If you use PyTau in your research, please cite:
```
@article{pytau2023,
  title={PyTau: Bayesian Changepoint Detection for Neural Data},
  author={Mahmood, Abuzar},
  journal={Journal of Open Source Software},
  year={2023}
}
```
