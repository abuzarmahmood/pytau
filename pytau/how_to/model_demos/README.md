# PyTau Model Demonstration Notebooks

This directory contains comprehensive demonstration notebooks for each model type available in PyTau. Each notebook provides:

- **Synthetic data generation** with known changepoints
- **Data visualization** to understand input patterns
- **Model fitting** using variational inference
- **Results analysis** with changepoint detection and parameter estimation
- **Output plots** showing model performance and parameter interpretation

## Available Model Demos

### Gaussian Models
- **`GaussianChangepointMeanVar2D_demo.ipynb`** - Detects changes in both mean and variance of Gaussian data
- **`GaussianChangepointMean2D_demo.ipynb`** - Detects changes in mean only (constant variance)
- **`GaussianChangepointMeanDirichlet_demo.ipynb`** - Uses Dirichlet process to automatically determine number of states

### Poisson Models (Neural Spike Data)
- **`SingleTastePoisson_demo.ipynb`** - Basic Poisson changepoint model for single experimental condition
- **`AllTastePoisson_demo.ipynb`** - Hierarchical model for multiple experimental conditions with shared changepoints

### Categorical Models
- **`CategoricalChangepoint2D_demo.ipynb`** - Detects changepoints in categorical/discrete data sequences

## Quick Start

1. **Choose a model** based on your data type:
   - **Continuous data**: Use Gaussian models
   - **Count data (spikes)**: Use Poisson models
   - **Categorical data**: Use Categorical models

2. **Open the corresponding notebook** in Jupyter

3. **Run all cells** to see the complete demonstration

4. **Adapt the code** to your own data by:
   - Replacing synthetic data generation with your data loading
   - Adjusting model parameters (n_states, dimensions, etc.)
   - Modifying visualization code for your specific needs

## Model Selection Guide

| Data Type | Model | Use Case |
|-----------|-------|----------|
| Continuous (Gaussian) | `GaussianChangepointMeanVar2D` | Both mean and variance change |
| Continuous (Gaussian) | `GaussianChangepointMean2D` | Only mean changes |
| Continuous (Gaussian) | `GaussianChangepointMeanDirichlet` | Unknown number of states |
| Count data (spikes) | `SingleTastePoisson` | Single experimental condition |
| Count data (spikes) | `AllTastePoisson` | Multiple conditions, shared timing |
| Categorical/Discrete | `CategoricalChangepoint2D` | Discrete state sequences |

## Common Workflow

Each notebook follows this general structure:

1. **Import libraries** and set up environment
2. **Generate synthetic data** with known changepoints
3. **Visualize input data** to understand patterns
4. **Create and fit model** using PyMC variational inference
5. **Analyze results** - extract changepoints and parameters
6. **Visualize model fit** - compare inferred vs. true changepoints
7. **Examine parameters** - understand state-specific parameters
8. **Model validation** - compare predictions with observations

## Tips for Using These Notebooks

- **Start with synthetic data** to understand model behavior
- **Adjust n_states** based on your expected number of changepoints
- **Increase inference iterations** for more complex models or larger datasets
- **Use appropriate data preprocessing** (scaling, normalization) for real data
- **Validate results** by comparing with known ground truth when available

## Additional Models

For demonstrations of other model variants (e.g., `SingleTastePoissonVarsig`, `SingleTastePoissonDirichlet`, etc.), refer to the main PyTau documentation and adapt the patterns shown in these core demonstrations.

## Getting Help

- Check the main PyTau documentation: [https://abuzarmahmood.github.io/pytau/](https://abuzarmahmood.github.io/pytau/)
- Review the existing example notebooks in `../notebooks/`
- Examine the model source code in `../../changepoint_model.py`
- Open an issue on the PyTau GitHub repository for specific questions
