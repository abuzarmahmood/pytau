# Inference Methods

PyTau provides multiple inference methods for fitting changepoint models to neural data. This page covers the available inference approaches, their trade-offs, and usage examples.

## Overview

PyTau supports three main inference approaches:

1. **Variational Inference (ADVI)**: Fast approximate inference
2. **MCMC Sampling**: High-quality posterior samples
3. **Model Selection**: Automatic determination of optimal state numbers

## Variational Inference (ADVI)

Automatic Differentiation Variational Inference (ADVI) provides fast approximate inference by optimizing a variational distribution to approximate the posterior.

### Advantages
- **Speed**: Much faster than MCMC, especially for large datasets
- **Scalability**: Handles high-dimensional models efficiently
- **Convergence monitoring**: Built-in convergence diagnostics

### Disadvantages
- **Approximation**: May not capture complex posterior structure
- **Local optima**: Can get stuck in local minima
- **Underestimated uncertainty**: Tends to underestimate posterior variance

### Usage

```python
from pytau.changepoint_model import advi_fit

# Fit model using ADVI
model, approx = advi_fit(
    model,
    fit=10000,           # Number of optimization iterations
    samples=5000,        # Number of samples to draw from approximation
    convergence_tol=0.01 # Convergence tolerance
)

# Draw samples from the approximation
trace = approx.sample(draws=5000)
```

### Parameters

- **fit**: Number of ADVI iterations (default: 10000)
  - More iterations = better approximation but slower
  - Monitor ELBO to check convergence

- **samples**: Number of samples to draw (default: 5000)
  - More samples = better posterior estimates
  - Typically 1000-10000 is sufficient

- **convergence_tol**: Relative change in ELBO for convergence (default: 0.01)
  - Smaller values = stricter convergence
  - 0.001-0.01 is typical

### Full-Rank ADVI

PyTau uses full-rank ADVI by default, which models correlations between parameters:

```python
# Full-rank ADVI (default)
model, approx = advi_fit(model, fit=10000)

# The approximation captures parameter correlations
# Better approximation quality than mean-field ADVI
```

### Monitoring Convergence

```python
# Access ELBO history
elbo_history = approx.hist

# Plot convergence
import matplotlib.pyplot as plt
plt.plot(elbo_history)
plt.xlabel('Iteration')
plt.ylabel('ELBO')
plt.title('ADVI Convergence')
plt.show()
```

## MCMC Sampling

Markov Chain Monte Carlo (MCMC) sampling provides high-quality posterior samples using the No-U-Turn Sampler (NUTS).

### Advantages
- **Accuracy**: Provides exact samples from the posterior (asymptotically)
- **Uncertainty quantification**: Accurate uncertainty estimates
- **Convergence diagnostics**: Rich diagnostics (R-hat, effective sample size)

### Disadvantages
- **Speed**: Much slower than ADVI
- **Scalability**: Can be challenging for very high-dimensional models
- **Tuning**: Requires warm-up/tuning phase

### Standard MCMC

```python
from pytau.changepoint_model import mcmc_fit

# Fit model using MCMC
model, trace, lambda_stack, tau_samples, observations = mcmc_fit(
    model,
    samples=1000,    # Number of samples per chain
    tune=500,        # Number of tuning samples
    chains=4,        # Number of chains
    cores=4          # Number of CPU cores to use
)
```

### Parameters

- **samples**: Number of posterior samples per chain (default: 1000)
  - More samples = better posterior estimates
  - 1000-5000 is typical

- **tune**: Number of tuning/warm-up samples (default: 500)
  - Discarded samples used to tune sampler
  - 500-2000 is typical

- **chains**: Number of independent chains (default: 4)
  - Multiple chains enable convergence diagnostics
  - 4 chains is standard

- **cores**: Number of CPU cores for parallel sampling
  - Set to number of chains for maximum parallelization

### Dirichlet Process Prior Models

For models with Dirichlet process priors (automatic state detection), use the specialized fitting function:

```python
from pytau.changepoint_model import dpp_fit

# Fit Dirichlet process model
dpp_trace = dpp_fit(
    model,
    n_chains=4,
    tune=500,
    draws=500,
    use_numpyro=True  # Use NumPyro backend for better performance
)
```

### NumPyro Backend

PyTau supports the NumPyro backend for improved MCMC performance:

```python
# Use NumPyro for faster sampling
trace = dpp_fit(model, use_numpyro=True)

# NumPyro advantages:
# - Faster sampling (GPU support)
# - Better memory efficiency
# - Improved convergence for complex models
```

### Convergence Diagnostics

```python
import arviz as az

# Convert trace to ArviZ InferenceData
idata = az.from_pymc3(trace)

# Check R-hat (should be < 1.01)
print(az.summary(idata, var_names=['tau']))

# Plot trace
az.plot_trace(idata, var_names=['tau'])

# Check effective sample size
print(az.ess(idata))
```

## Model Selection

PyTau provides tools for automatically determining the optimal number of states using ELBO comparison.

### Automatic State Detection

```python
from pytau.changepoint_model import find_best_states

# Find optimal number of states
best_model, model_list, elbo_values = find_best_states(
    data,
    model_generator,
    n_fit=5000,        # ADVI iterations per model
    n_samples=1000,    # Samples per model
    min_states=2,      # Minimum states to try
    max_states=8       # Maximum states to try
)

# best_model: Model with highest ELBO
# model_list: List of all fitted models
# elbo_values: ELBO for each state number
```

### Plotting Model Comparison

```python
import matplotlib.pyplot as plt

# Plot ELBO vs number of states
states = range(min_states, max_states + 1)
plt.plot(states, elbo_values, 'o-')
plt.xlabel('Number of States')
plt.ylabel('ELBO')
plt.title('Model Selection')
plt.show()

# Higher ELBO = better model fit
# Look for elbow or plateau
```

### Cross-Validation

For more robust model selection, use cross-validation:

```python
from pytau.changepoint_model import cross_validate_states

# K-fold cross-validation
cv_scores = cross_validate_states(
    data,
    model_generator,
    k_folds=5,
    min_states=2,
    max_states=8
)

# Select model with best cross-validation score
best_n_states = cv_scores.argmax() + min_states
```

## Choosing an Inference Method

### Use ADVI when:
- You need fast results
- You're doing exploratory analysis
- You're fitting many models (batch processing)
- Approximate inference is sufficient

### Use MCMC when:
- You need accurate uncertainty estimates
- You're doing final analysis for publication
- You have time for longer computation
- You need convergence diagnostics

### Use Dirichlet Models when:
- Number of states is unknown
- You want data-driven state detection
- You can afford longer inference time

## Inference Best Practices

### 1. Start with ADVI
Begin with ADVI for quick exploration, then use MCMC for final analysis if needed.

### 2. Check Convergence
Always check convergence diagnostics:
- ADVI: Monitor ELBO convergence
- MCMC: Check R-hat < 1.01 and effective sample size

### 3. Multiple Chains
Use multiple chains (4+) for MCMC to detect convergence issues.

### 4. Sufficient Samples
Use enough samples:
- ADVI: 1000-10000 samples
- MCMC: 1000-5000 samples per chain

### 5. Tune Appropriately
For MCMC, use sufficient tuning:
- Simple models: 500 tuning samples
- Complex models: 1000-2000 tuning samples

### 6. Parallel Processing
Use multiple cores for MCMC to speed up sampling:
```python
# Use all available cores
import multiprocessing
n_cores = multiprocessing.cpu_count()
trace = mcmc_fit(model, cores=n_cores, chains=n_cores)
```

### 7. Save Results
Save fitted models for later analysis:
```python
import pickle

# Save model and trace
with open('fitted_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'trace': trace}, f)
```

## Troubleshooting

### ADVI Not Converging
- Increase `fit` iterations
- Try different random seeds
- Check for numerical issues in data (NaNs, infinities)
- Consider data preprocessing (normalization, binning)

### MCMC Divergences
- Increase `tune` samples
- Reparameterize model
- Check for label switching in changepoint models
- Use stronger priors

### Slow Inference
- Use ADVI instead of MCMC
- Reduce data size (coarser binning, fewer trials)
- Use NumPyro backend for MCMC
- Parallelize across multiple cores

### Memory Issues
- Reduce number of samples
- Process data in batches
- Use ADVI instead of MCMC
- Increase system swap space

## Caveats and Limitations

### Single Time-Series State Inference

**Important Note**: Inferring the number of states for single time-series data can be unreliable when using both multi-model comparisons and Dirichlet process prior (DPP) fits.

#### The Challenge

When working with single time-series data (e.g., `PoissonChangepoint1D` with shape `(time,)`), determining the optimal number of states through model comparison can be particularly challenging. Both common approaches have significant limitations:

1. **Multi-model ELBO comparison**: Fitting multiple models with different state numbers and comparing their ELBO values
2. **Dirichlet process prior models**: Using models like `SingleTastePoissonDirichlet` or `GaussianChangepointMeanDirichlet` for automatic state detection

#### Why This Matters

Single time-series data lacks the hierarchical structure present in multi-trial or multi-neuron data. This makes the inference problem substantially more difficult because:

- **Limited data**: A single time series provides less information for the model to distinguish between states
- **Overfitting risk**: Models can easily overfit, assigning states to noise rather than true underlying structure
- **Poor generalization**: ELBO values may not reliably indicate the true number of states
- **Inconsistent results**: Different inference methods (ADVI vs MCMC) or random seeds can yield very different state number estimates

#### Visual Examples

The following images from [GitHub Issue #153](https://github.com/abuzarmahmood/pytau/issues/153) illustrate these challenges:

**Example 1: Multi-model comparison showing poor ELBO discrimination**

![State inference challenge 1](https://github.com/user-attachments/assets/ebeafd14-f52b-43cb-b2a5-4e6342ed38b4)

**Example 2: Dirichlet process fit showing inconsistent state detection**

![State inference challenge 2](https://github.com/user-attachments/assets/2e121bfe-4e30-4f7c-aaea-f4092b4e1c45)

These examples demonstrate that both ELBO-based model selection and DPP-based automatic state detection can struggle with single time-series data, often failing to identify a clear optimal number of states.

#### Recommendations for Better Decision-Making

When working with single time-series data:

1. **Use domain knowledge**: Let your experimental design and scientific hypotheses guide the choice of state numbers rather than relying solely on automated methods

2. **Prefer multi-trial data**: Whenever possible, use multi-trial models (e.g., `SingleTastePoisson` with shape `(trials, time)`) which provide more robust inference through hierarchical structure

3. **Cross-validate carefully**: If you must use single time-series data, consider:
   - Splitting your time series into multiple segments for cross-validation
   - Using held-out data to validate state assignments
   - Comparing predictions across different model specifications

4. **Be conservative**: When in doubt, prefer simpler models (fewer states) that align with your scientific understanding

5. **Inspect results visually**: Always visually inspect the inferred changepoints and state assignments to ensure they make scientific sense

6. **Use multiple inference approaches**: Compare results from both ADVI and MCMC, and examine consistency across multiple random initializations

7. **Report uncertainty**: When publishing results from single time-series analyses, clearly report the limitations and uncertainty in state number selection

#### Alternative Approaches

Consider these alternatives when facing state inference challenges:

- **Aggregate data**: If you have multiple similar time series, consider pooling them into a hierarchical model
- **Fixed state models**: Specify the number of states based on experimental design rather than inferring it
- **Sliding window analysis**: Break long time series into shorter segments and analyze consistency across segments
- **Model averaging**: Rather than selecting a single "best" model, consider averaging predictions across models with different state numbers

### General Model Selection Cautions

Beyond single time-series issues, be aware that:

- ELBO values can be sensitive to initialization and convergence
- Cross-validation scores may be optimistic due to temporal dependencies
- State number inference is fundamentally difficult in changepoint models due to label switching and identifiability issues

Always combine automated model selection with scientific judgment and domain expertise.

## Next Steps

- See [Available Models](models.md) for model descriptions
- See [Data Formats](data_formats.md) for data preparation
- See [Examples and Tutorials](examples.md) for batch processing examples
- Check the [API documentation](api.md) for detailed function signatures
