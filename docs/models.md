# Available Models

PyTau provides a comprehensive suite of changepoint detection models designed for different types of neural data and experimental designs. This page details all available models and their use cases.

## Model Selection Guide

| Data Type | Recommended Models |
|-----------|-------------------|
| Single neuron time series | `PoissonChangepoint1D` |
| Multiple trials, single taste | `SingleTastePoisson*` |
| Multiple tastes | `AllTastePoisson*` |
| Continuous neural data | `GaussianChangepoint*` |
| Categorical behavioral data | `CategoricalChangepoint2D` |
| Unknown state number | `*Dirichlet` models |

## Poisson Models

Poisson models are designed for spike count data from neural recordings. They model the firing rate of neurons and detect when these rates change over time.

### PoissonChangepoint1D

**Purpose**: 1D time series changepoint detection

**Use Case**: Single neuron or single trial analysis where you want to detect changes in firing rate over time.

**Data Shape**: `(time,)`

**Key Features**:
- Simplest Poisson model
- Ideal for exploratory analysis
- Fast inference

### SingleTastePoisson

**Purpose**: Basic single-taste model with fixed sigmoid transitions

**Use Case**: Multi-trial experiments with a single stimulus (taste) where you want to detect consistent changepoints across trials.

**Data Shape**: `(trials, neurons, time)` or `(trials, time)`

**Key Features**:
- Fixed sigmoid transition shape
- Assumes consistent changepoint timing across trials
- Hierarchical structure across neurons

### SingleTastePoissonVarsig

**Purpose**: Variable sigmoid slope inferred from data

**Use Case**: When transition sharpness is unknown and should be learned from data.

**Data Shape**: `(trials, neurons, time)` or `(trials, time)`

**Key Features**:
- Infers transition sharpness from data
- More flexible than fixed sigmoid
- Longer inference time

### SingleTastePoissonVarsigFixed

**Purpose**: Fixed sigmoid slope with configurable sharpness

**Use Case**: When you want to specify transition sharpness based on prior knowledge.

**Data Shape**: `(trials, neurons, time)` or `(trials, time)`

**Key Features**:
- User-specified transition sharpness
- Faster than variable sigmoid
- Good for batch processing with consistent parameters

### SingleTastePoissonDirichlet

**Purpose**: Automatic state number detection using Dirichlet process

**Use Case**: When the number of states is unknown and should be inferred from data.

**Data Shape**: `(trials, neurons, time)` or `(trials, time)`

**Key Features**:
- Automatically determines optimal number of states
- Uses Dirichlet process prior
- Requires MCMC sampling (no ADVI support)
- Longer inference time but more flexible

### SingleTastePoissonTrialSwitch

**Purpose**: Models trial-to-trial emission changes

**Use Case**: When changepoint times are consistent but firing rates vary across trials.

**Data Shape**: `(trials, neurons, time)` or `(trials, time)`

**Key Features**:
- Fixed changepoint times across trials
- Trial-specific emission rates
- Useful for detecting trial-to-trial variability

### AllTastePoisson

**Purpose**: Hierarchical model across multiple tastes

**Use Case**: Multi-taste experiments where you want to detect both taste-specific and shared changepoints.

**Data Shape**: `(tastes, trials, neurons, time)`

**Key Features**:
- Hierarchical structure across tastes
- Can detect shared patterns across stimuli
- More complex inference

### AllTastePoissonVarsigFixed

**Purpose**: All-taste model with fixed sigmoid transitions

**Use Case**: Multi-taste experiments with specified transition sharpness.

**Data Shape**: `(tastes, trials, neurons, time)`

**Key Features**:
- Fixed sigmoid slope across all tastes
- Faster than variable sigmoid version
- Good for batch processing

### AllTastePoissonTrialSwitch

**Purpose**: All-taste model with trial switching

**Use Case**: Multi-taste experiments with trial-to-trial variability in emissions.

**Data Shape**: `(tastes, trials, neurons, time)`

**Key Features**:
- Trial-specific emission rates
- Taste-specific changepoint structure
- Captures complex experimental designs

## Gaussian Models

Gaussian models are designed for continuous neural data such as LFP signals, calcium imaging, or other continuous measurements.

### GaussianChangepointMean2D

**Purpose**: Detects changes in mean only

**Use Case**: When variance is stable but mean changes over time.

**Data Shape**: `(trials, time)` or `(neurons, time)`

**Key Features**:
- Assumes constant variance
- Faster inference than mean+variance model
- Good for data with stable noise levels

### GaussianChangepointMeanVar2D

**Purpose**: Detects changes in both mean and variance

**Use Case**: When both signal level and noise characteristics change.

**Data Shape**: `(trials, time)` or `(neurons, time)`

**Key Features**:
- Models both mean and variance changes
- More flexible but slower inference
- Better for complex signals

### GaussianChangepointMeanDirichlet

**Purpose**: Automatic state detection for Gaussian data

**Use Case**: Continuous data with unknown number of states.

**Data Shape**: `(trials, time)` or `(neurons, time)`

**Key Features**:
- Dirichlet process prior for state number
- Detects changes in mean only
- Requires MCMC sampling

## Categorical Models

Categorical models are designed for discrete categorical data such as behavioral states or decoded neural states.

### CategoricalChangepoint2D

**Purpose**: Changepoint detection for categorical time series

**Use Case**: Behavioral data or decoded states where you want to detect transitions between discrete categories.

**Data Shape**: `(trials, time)` with integer category labels

**Key Features**:
- Models categorical distributions
- Detects changes in category probabilities
- Useful for behavioral analysis

**Known Issues**:
- ELBO values may become infinite during inference ([Issue #186](https://github.com/abuzarmahmood/pytau/issues/186))
- This is a known issue and the model is being re-evaluated
- Despite this issue, the model has been successfully used in research (see [Mahmood et al., 2025](https://www.biorxiv.org/content/10.1101/2025.10.01.679845v1.full))
- If you encounter infinite ELBO values, consider tracking model parameters as described in the [PyMC documentation](https://www.pymc.io/projects/examples/en/2022.01.0/variational_inference/variational_api_quickstart.html#tracking-parameters)

## Model Naming Conventions

PyTau models follow consistent naming conventions:

- **Dimension suffix** (1D, 2D): Indicates the dimensionality of input data
- **Varsig**: Variable sigmoid transition shape
- **VarsigFixed**: Fixed sigmoid with configurable slope
- **Dirichlet**: Uses Dirichlet process prior for automatic state detection
- **TrialSwitch**: Models trial-to-trial variability in emissions
- **AllTaste**: Hierarchical model across multiple stimuli/conditions

## Choosing the Right Model

### Start Simple
Begin with basic models (e.g., `SingleTastePoisson`) and add complexity only if needed.

### Consider Your Data
- **Spike counts**: Use Poisson models
- **Continuous signals**: Use Gaussian models
- **Categorical data**: Use Categorical models

### Unknown States
If you don't know how many states to expect, use Dirichlet models, but be prepared for longer inference times.

### Transition Shape
- Use **Varsig** if transition sharpness is unknown
- Use **VarsigFixed** if you have prior knowledge about transitions
- Fixed sigmoid models are faster for batch processing

### Trial Variability
Use **TrialSwitch** models if you expect changepoint times to be consistent but firing rates to vary across trials.

## Next Steps

- See [Inference Methods](inference.md) for details on fitting these models
- See [Data Formats](data_formats.md) for information on preparing your data
- Check the [API documentation](api.md) for detailed function signatures
