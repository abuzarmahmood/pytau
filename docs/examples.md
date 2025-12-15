# Examples and Tutorials

This page provides detailed examples and tutorials for using PyTau to analyze neural data with changepoint models.

## Complete Analysis Workflow

Below is a comprehensive example demonstrating the full workflow from data loading to visualization:

```python
from pytau.changepoint_io import FitHandler
from pytau.changepoint_analysis import PklHandler
import matplotlib.pyplot as plt
from pytau.utils.plotting import plot_changepoint_raster, plot_state_firing_rates

# Initialize fit handler
fh = FitHandler(
    data_dir='/path/to/data',
    taste_num=1,
    region_name='GC',
    experiment_name='example_experiment'
)

# Set preprocessing parameters
fh.set_preprocess_params(
    time_lims=[0, 2000],  # Time window in ms
    bin_width=10,         # Bin width in ms
    data_transform=None   # No transformation
)

# Set model parameters
fh.set_model_params(
    states=3,             # Number of states to fit
    fit=5000,             # ADVI iterations
    samples=1000,         # Number of posterior samples
    model_kwargs={}       # Additional model parameters
)

# Run the full pipeline
fh.load_spike_trains()
fh.preprocess_data()
fh.create_model()
fh.run_inference()
fh.save_fit_output()
```

## Analyzing Fitted Models

After fitting, analyze the results using the `PklHandler` class:

```python
from pytau.changepoint_analysis import PklHandler

# Load fitted model
pkl_handler = PklHandler('/path/to/saved/model.pkl')

# Access model components
tau = pkl_handler.tau  # Changepoint times
firing = pkl_handler.firing  # Firing rate analysis

# Analyze significant neurons
significant_neurons = firing.anova_significant_neurons

# Access transition analysis data
transition_snippets = firing.transition_snips
pairwise_significant = firing.pairwise_significant_neurons
```

## Visualization Examples

### Spike Rasters with Changepoint Overlays

```python
import matplotlib.pyplot as plt
from pytau.utils.plotting import plot_changepoint_raster

# Plot spike rasters with changepoint overlays
fig, ax = plt.subplots(figsize=(10, 6))
plot_changepoint_raster(pkl_handler.processed_spikes,
                        pkl_handler.tau.scaled_mode_tau,
                        plot_lims=[0, 2000])
plt.title('Spike Rasters with Changepoint Overlays')
plt.show()
```

### State-Dependent Firing Rates

```python
from pytau.utils.plotting import plot_state_firing_rates

# Plot state-dependent firing rates
plot_state_firing_rates(pkl_handler.processed_spikes,
                        pkl_handler.tau.scaled_mode_tau)
plt.title('State-Specific Neuron Activity')
plt.show()
```

### Transition-Aligned Activity

```python
from pytau.utils.plotting import plot_aligned_state_firing

# Plot transition-aligned activity
plot_aligned_state_firing(pkl_handler.processed_spikes,
                          pkl_handler.tau.scaled_mode_tau,
                          window_radius=300)
plt.title('Transition-Aligned Neural Activity')
plt.show()
```

### State Timing Overview

```python
from pytau.utils.plotting import plot_changepoint_overview

# Plot state timing overview
plot_changepoint_overview(pkl_handler.tau.scaled_mode_tau,
                         plot_lims=[0, 2000])
plt.title('State Timing Overview')
plt.show()
```

## Model Selection

PyTau implements several types of changepoint models. For detailed descriptions of all available models, see the [Available Models](models.md) documentation.

Quick example of creating a model:

```python
from pytau.changepoint_model import single_taste_poisson

# Create single taste Poisson model
model = single_taste_poisson(spike_array, states=3)
```

## Statistical Analysis

For detailed information on statistical analysis tools, see the API documentation. Here are quick examples:

```python
from pytau.changepoint_analysis import get_state_firing, calc_significant_neurons_firing
from pytau.changepoint_analysis import get_transition_snips, calc_significant_neurons_snippets

# State-dependent firing rate analysis
state_firing = get_state_firing(spike_array, tau_array)
significant_neurons = calc_significant_neurons_firing(state_firing, p_val=0.05)

# Transition analysis
transition_snips = get_transition_snips(spike_array, tau_array, window_radius=300)
pairwise_significant = calc_significant_neurons_snippets(transition_snips, p_val=0.05)
```

## Batch Processing

For processing multiple datasets or parameter configurations, use the `FitHandler` class in a loop:

```python
from pytau.changepoint_io import FitHandler

# Process multiple datasets
for dataset in datasets:
    fh = FitHandler(
        data_dir=dataset['path'],
        taste_num=dataset['taste'],
        region_name=dataset['region'],
        experiment_name=dataset['name']
    )

    # Set parameters and run pipeline
    fh.set_preprocess_params(time_lims=[0, 2000], bin_width=10, data_transform=None)
    fh.set_model_params(states=3, fit=5000, samples=1000)
    fh.load_spike_trains()
    fh.preprocess_data()
    fh.create_model()
    fh.run_inference()
    fh.save_fit_output()
```

## Advanced Usage

### Model Comparison

For detailed information on model selection and comparison methods, see the [Inference Methods](inference.md#model-selection) documentation.

Quick example:

```python
# Compare models with different state numbers
elbo_values = []
for n_states in range(2, 8):
    fh.set_model_params(states=n_states, fit=5000, samples=1000)
    fh.create_model()
    fh.run_inference()
    elbo_values.append(fh.model.approx.hist[-1])

# Plot ELBO comparison
plt.plot(range(2, 8), elbo_values, 'o-')
plt.xlabel('Number of States')
plt.ylabel('ELBO')
plt.title('Model Comparison')
plt.show()
```

## Tutorials

Comprehensive tutorials are available in the repository's `how_to` directory:

### Jupyter Notebooks

Step-by-step walkthroughs demonstrating package functionality are available in `pytau/how_to/notebooks/`:

- `PyTau_Walkthrough_with_handlers.ipynb`: Complete workflow using FitHandler
- `PyTau_Walkthrough_no_handlers.ipynb`: Manual workflow without handlers

Additional model-specific notebooks are in `pytau/how_to/model_notebooks/`:

- `SingleTastePoisson.ipynb`: Single taste Poisson model examples
- `AllTastePoisson.ipynb`: Multi-taste model examples
- `GaussianChangepointMean2D.ipynb`: Gaussian changepoint models
- And more...

### Example Scripts

Ready-to-run Python scripts in `pytau/how_to/scripts/`:

- `fit_with_fit_handler.py`: Fit models using FitHandler
- `fit_manually.py`: Manual model fitting workflow

### Test Data

Scripts to download test datasets:

```bash
cd pytau/how_to/scripts
bash download_test_data.sh
```

This downloads example neural data for practicing with the package.

## Next Steps

- See [Available Models](models.md) for detailed model descriptions
- See [Inference Methods](inference.md) for fitting strategies
- See [Data Formats](data_formats.md) for data preparation
- Check the [API documentation](api.md) for function signatures
