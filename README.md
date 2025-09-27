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

# Install package
cd pytau
pip install -e .  # Install in development mode

# Alternative: Install from requirements file
# pip install -r requirements.txt

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
  - Gaussian changepoint models for continuous data
  - Categorical changepoint models for discrete data

- **Flexible Data Handling**:
  - Support for shuffled, simulated, and actual neural data
  - Comprehensive preprocessing pipeline
  - Automated model selection

- **Powerful Analysis Tools**:
  - State transition detection
  - Cross-region correlation analysis
  - Statistical significance testing

## 📊 Available Models

### Poisson Models (for spike count data)
- **`PoissonChangepoint1D`**: 1D time series changepoint detection
- **`SingleTastePoisson`**: Basic single-taste model with fixed sigmoid transitions
- **`SingleTastePoissonVarsig`**: Variable sigmoid slope inferred from data
- **`SingleTastePoissonVarsigFixed`**: Fixed sigmoid slope with configurable sharpness
- **`SingleTastePoissonDirichlet`**: Automatic state number detection using Dirichlet process
- **`SingleTastePoissonTrialSwitch`**: Models trial-to-trial emission changes
- **`AllTastePoisson`**: Hierarchical model across multiple tastes
- **`AllTastePoissonVarsigFixed`**: All-taste model with fixed sigmoid transitions
- **`AllTastePoissonTrialSwitch`**: All-taste model with trial switching

### Gaussian Models (for continuous data)
- **`GaussianChangepointMean2D`**: Detects changes in mean only
- **`GaussianChangepointMeanVar2D`**: Detects changes in both mean and variance
- **`GaussianChangepointMeanDirichlet`**: Automatic state detection for Gaussian data

### Categorical Models (for discrete categorical data)
- **`CategoricalChangepoint2D`**: Changepoint detection for categorical time series

## 🔧 Inference Methods

### Variational Inference (ADVI)
- **Fast approximate inference** using Automatic Differentiation Variational Inference
- **Convergence monitoring** with customizable tolerance
- **Full-rank ADVI** for better approximation quality

```python
from pytau.changepoint_model import advi_fit

# Fit model using ADVI
model, approx = advi_fit(model, fit=10000, samples=5000, convergence_tol=0.01)
trace = approx.sample(draws=5000)
```

### MCMC Sampling
- **High-quality posterior samples** using NUTS sampler
- **Multi-chain sampling** for convergence diagnostics
- **NumPyro backend support** for improved performance

```python
from pytau.changepoint_model import mcmc_fit, dpp_fit

# Standard MCMC
model, trace, lambda_stack, tau_samples, observations = mcmc_fit(model, samples=1000)

# Dirichlet Process Prior models
dpp_trace = dpp_fit(model, n_chains=4, tune=500, draws=500, use_numpyro=True)
```

### Model Selection
- **Automatic state number detection** using ELBO comparison
- **Cross-validation support** for model comparison

```python
from pytau.changepoint_model import find_best_states

# Find optimal number of states
best_model, model_list, elbo_values = find_best_states(
    data, model_generator, n_fit=5000, n_samples=1000,
    min_states=2, max_states=8
)
```

## 📈 Data Formats

### Input Data Shapes
- **1D**: `(time,)` - Single time series
- **2D**: `(trials, time)` or `(neurons, time)` - Multiple trials or neurons
- **3D**: `(trials, neurons, time)` - Single taste experiments
- **4D**: `(tastes, trials, neurons, time)` - Multi-taste experiments

### Model Selection Guide
| Data Type | Recommended Models |
|-----------|-------------------|
| Single neuron time series | `PoissonChangepoint1D` |
| Multiple trials, single taste | `SingleTastePoisson*` |
| Multiple tastes | `AllTastePoisson*` |
| Continuous neural data | `GaussianChangepoint*` |
| Categorical behavioral data | `CategoricalChangepoint2D` |
| Unknown state number | `*Dirichlet` models |

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
# Trigger CI
