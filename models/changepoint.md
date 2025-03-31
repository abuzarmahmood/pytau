# Changepoint Models

PyTau provides several changepoint models for analyzing neural data. These models are designed to detect changes in neural activity patterns over time.

## Available Models

### Gaussian Models
- **GaussianChangepointMeanVar2D**: Model for gaussian data on 2D array detecting changes in both mean and variance.
- **GaussianChangepointMeanDirichlet**: Model for gaussian data on 2D array detecting changes only in the mean. Number of states determined using dirichlet process prior.
- **GaussianChangepointMean2D**: Model for gaussian data on 2D array detecting changes only in the mean.

### Poisson Models for Single Taste
- **SingleTastePoissonDirichlet**: Model for changepoint on single taste using dirichlet process prior.
- **SingleTastePoisson**: Model for changepoint on single taste.
- **SingleTastePoissonVarsig**: Model for changepoint on single taste with variable sigmoid slope.
- **SingleTastePoissonVarsigFixed**: Model for changepoint on single taste with fixed sigmoid slope.
- **SingleTastePoissonTrialSwitch**: Model assuming only emissions change across trials, with changepoint distribution remaining constant.

### Poisson Models for All Tastes
- **AllTastePoisson**: Model to fit changepoint to all tastes.
- **AllTastePoissonVarsigFixed**: Model to fit changepoint to all tastes with fixed sigmoid.
- **AllTastePoissonTrialSwitch**: Model for all tastes assuming only emissions change across trials.

## Usage Example

```python
import numpy as np
from pytau.changepoint_model import SingleTastePoisson, advi_fit

# Generate or load your spike data
# spike_array should be a 3D array: trials x neurons x time
spike_array = ...

# Create model
model_instance = SingleTastePoisson(spike_array, states=3)
model = model_instance.generate_model()

# Run inference
model, trace, lambda_stack, tau_samples, observations = advi_fit(model, fit=10000, samples=1000)

# Analyze results
# tau_samples contains the inferred changepoint times
# lambda_stack contains the inferred firing rates
```

## Model Selection

To find the optimal number of states for your data:

```python
from pytau.changepoint_model import find_best_states, SingleTastePoisson

# Find best number of states
best_model, model_list, elbo_values = find_best_states(
    data=spike_array,
    model_generator=lambda data, states: SingleTastePoisson(data, states).generate_model(),
    n_fit=10000,
    n_samples=1000,
    min_states=2,
    max_states=6
)
```
