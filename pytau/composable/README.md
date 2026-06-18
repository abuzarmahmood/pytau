# Composable Changepoint Models

This module provides a flexible framework for creating changepoint models by combining different components. This addresses issue #21 by allowing users to mix and match:

1. **Changepoint types**: Fixed vs Dirichlet process
2. **Change types**: Sigmoid vs step transitions
3. **Likelihood types**: Gaussian vs Poisson
4. **Emission models**: Different parameter structures

## Architecture

The composable framework consists of three main component types:

### ChangepointComponent
Handles the generation of changepoint (tau) variables with different prior types and change types.

- `FixedChangepoint`: Fixed number of changepoints with Beta priors
- `DirichletChangepoint`: Dirichlet process for automatic state detection

### EmissionComponent
Handles the generation of emission rate/mean variables for different likelihood types.

- `PoissonEmission`: Lambda (rate) parameters for Poisson models
- `GaussianEmission`: Mu (mean) and optionally sigma (variance) parameters

### LikelihoodComponent
Handles the generation of observation likelihood given emissions and changepoint weights.

- `PoissonLikelihood`: For spike count data
- `GaussianLikelihood`: For continuous data

## Usage Examples

### Basic Usage

```python
from pytau.composable import (
    FixedChangepoint, PoissonEmission, PoissonLikelihood, ComposableModel
)

# Create components
changepoint_comp = FixedChangepoint(n_states=3, change_type='sigmoid')
emission_comp = PoissonEmission(data.shape, n_states=3)
likelihood_comp = PoissonLikelihood()

# Combine into model
model = ComposableModel(data, changepoint_comp, emission_comp, likelihood_comp)
pymc_model = model.generate_model()
```

### Convenience Functions

```python
from pytau.composable import create_poisson_fixed_sigmoid, create_model

# Using convenience function
model = create_poisson_fixed_sigmoid(data, n_states=3)

# Using generic create_model function
model = create_model('poisson_fixed_sigmoid', data, n_states=3)
```

### Available Model Combinations

- `poisson_fixed_sigmoid`: Poisson likelihood, fixed changepoints, sigmoid transitions
- `poisson_fixed_step`: Poisson likelihood, fixed changepoints, step transitions
- `poisson_dirichlet_sigmoid`: Poisson likelihood, Dirichlet process, sigmoid transitions
- `gaussian_fixed_sigmoid`: Gaussian likelihood, fixed changepoints, sigmoid transitions
- `gaussian_dirichlet_step`: Gaussian likelihood, Dirichlet process, step transitions

## Extending the Framework

### Adding New Changepoint Types

```python
from pytau.composable.base import ChangepointComponent

class MyChangepointComponent(ChangepointComponent):
    def generate_changepoints(self, time_length):
        # Implement your changepoint logic
        return tau, weight_stack
```

### Adding New Emission Types

```python
from pytau.composable.base import EmissionComponent

class MyEmissionComponent(EmissionComponent):
    def generate_emissions(self, data_array):
        # Implement your emission logic
        return emission_parameters
```

### Adding New Likelihood Types

```python
from pytau.composable.base import LikelihoodComponent

class MyLikelihoodComponent(LikelihoodComponent):
    def generate_likelihood(self, data_array, emissions, weight_stack):
        # Implement your likelihood logic
        return observation_variable
```

## Benefits

1. **Modularity**: Each component can be developed and tested independently
2. **Flexibility**: Easy to create new model combinations
3. **Reusability**: Components can be reused across different models
4. **Maintainability**: Changes to one component don't affect others
5. **Extensibility**: Easy to add new component types

## Backward Compatibility

The composable framework is designed to coexist with existing PyTau models. All existing model classes continue to work as before, while the new composable framework provides additional flexibility for users who need it.
