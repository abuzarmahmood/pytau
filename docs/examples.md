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

## Model Types and Features

PyTau implements several types of changepoint models to accommodate different analysis needs:

### Single Taste Poisson Models

For analyzing single-taste responses with Poisson emission distributions:

```python
from pytau.changepoint_model import single_taste_poisson

# Create single taste Poisson model
model = single_taste_poisson(spike_array, states=3)
```

### Variable Sigmoid Models

Models with learnable transition sharpness:

```python
from pytau.changepoint_model import single_taste_poisson_varsig

# Create variable sigmoid model
model = single_taste_poisson_varsig(spike_array, states=3)
```

### Fixed Sigmoid Models

Models with fixed transition sharpness:

```python
from pytau.changepoint_model import single_taste_poisson_varsig_fixed

# Create fixed sigmoid model
model = single_taste_poisson_varsig_fixed(spike_array, states=3, inds_span=1)
```

### All-Taste Models

For analyzing responses across multiple stimuli:

```python
from pytau.changepoint_model import all_taste_poisson

# Create all-taste model
model = all_taste_poisson(spike_array, states=3)
```

### Dirichlet Process Models

For automatically determining the number of states:

```python
from pytau.changepoint_model import single_taste_poisson_dirichlet

# Create Dirichlet process model
model = single_taste_poisson_dirichlet(spike_array, max_states=10)
```

## Statistical Analysis Tools

### State-Dependent Firing Rate Analysis

```python
from pytau.changepoint_analysis import get_state_firing

# Get state-dependent firing rates
state_firing = get_state_firing(spike_array, tau_array)
```

### Significance Testing

```python
from pytau.changepoint_analysis import calc_significant_neurons_firing

# Calculate significant neurons based on state-dependent firing
significant_neurons = calc_significant_neurons_firing(state_firing, p_val=0.05)
```

### Transition Analysis

```python
from pytau.changepoint_analysis import get_transition_snips, calc_significant_neurons_snippets

# Get transition snippets
transition_snips = get_transition_snips(spike_array, tau_array, window_radius=300)

# Calculate significant neurons based on transition-triggered activity
pairwise_significant = calc_significant_neurons_snippets(transition_snips, p_val=0.05)
```

## Batch Processing

For processing multiple datasets or parameter configurations:

```python
from pytau.changepoint_io import DatabaseHandler

# Initialize database handler
db_handler = DatabaseHandler(database_path='/path/to/database.csv')

# Process multiple datasets
for dataset in datasets:
    fh = FitHandler(
        data_dir=dataset['path'],
        taste_num=dataset['taste'],
        region_name=dataset['region'],
        experiment_name=dataset['name']
    )

    # Set parameters and run pipeline
    fh.set_preprocess_params(time_lims=[0, 2000], bin_width=10)
    fh.set_model_params(states=3, fit=5000, samples=1000)
    fh.load_spike_trains()
    fh.preprocess_data()
    fh.create_model()
    fh.run_inference()
    fh.save_fit_output()

    # Add to database
    db_handler.add_fit(fh.get_metadata())
```

## Advanced Usage

### Custom Model Parameters

```python
# Set custom model parameters
fh.set_model_params(
    states=4,
    fit=10000,
    samples=2000,
    model_kwargs={
        'tau_prior_mean': 1000,  # Custom prior for changepoint times
        'tau_prior_std': 200,
        'lambda_prior_alpha': 2,  # Custom prior for firing rates
        'lambda_prior_beta': 1
    }
)
```

### Data Transformations

```python
# Apply data transformations
fh.set_preprocess_params(
    time_lims=[0, 2000],
    bin_width=10,
    data_transform='sqrt'  # Square root transformation
)
```

### Model Comparison

```python
from pytau.utils.plotting import plot_elbo_history

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

Step-by-step walkthroughs demonstrating package functionality:

- `01_basic_usage.ipynb`: Introduction to basic PyTau workflow
- `02_model_comparison.ipynb`: Comparing different model types
- `03_batch_processing.ipynb`: Processing multiple datasets
- `04_visualization.ipynb`: Creating publication-quality figures
- `05_statistical_analysis.ipynb`: Analyzing state-dependent activity

### Example Scripts

Ready-to-run Python scripts:

- `fit_single_model.py`: Fit a single changepoint model
- `batch_fit_models.py`: Batch process multiple datasets
- `analyze_results.py`: Analyze and visualize fitted models
- `compare_models.py`: Compare models with different parameters

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
