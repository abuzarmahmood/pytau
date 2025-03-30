---
title: 'pytau: A Python package for streamlined changepoint model analysis in neuroscience'
tags:
  - Python
  - neuroscience
  - changepoint analysis
  - time-series
  - bayesian modeling
authors:
  - name: Abuzar Mahmood
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: Independent Researcher
   index: 1
date: 30 March 2025
bibliography: paper.bib
---

# Summary

Analyzing complex biological data, particularly time-series data from neuroscience experiments, often requires sophisticated statistical modeling to identify significant changes in system dynamics. Changepoint models are crucial for detecting abrupt shifts or transitions in neural activity patterns. `pytau` is a Python software package designed to perform streamlined, batched inference for changepoint models across different parameter grids and datasets. It provides tools to efficiently query and analyze the results from sets of fitted models, facilitating the study of dynamic processes in biological systems, such as neural ensemble activity in response to stimuli. The package integrates with PyMC3 for Bayesian inference and provides utilities for data preprocessing, model fitting, and result visualization.

# Statement of need

Understanding how neural populations encode information often involves analyzing activity changes over time, potentially across different experimental conditions, parameters, or subjects. Fitting and comparing complex models like Bayesian changepoint models across numerous datasets or parameter settings can be computationally intensive and logistically challenging. There is a need for tools that streamline this process, enabling researchers to efficiently apply these models in batch, manage the results, and compare outcomes across conditions. `pytau` aims to fill this gap by providing a modularized pipeline specifically for fitting and analyzing changepoint models applied to neuroscience data, enabling efficient comparisons and analysis.

The package offers several key advantages:

1. **Batch processing**: Automates the fitting of models across multiple datasets and parameter configurations
2. **Database management**: Organizes and tracks model fits for easy retrieval and comparison
3. **Visualization tools**: Provides specialized plotting functions for changepoint model results
4. **Flexible model specification**: Supports various changepoint model configurations for different analysis needs
5. **Statistical analysis**: Includes tools for significance testing of state-dependent neural activity

These features make `pytau` particularly valuable for neuroscientists studying state transitions in neural activity, such as taste processing, decision-making, or learning paradigms.

# Implementation and architecture

`pytau` is implemented in Python and built on several key libraries including NumPy, SciPy, PyMC3, and Matplotlib. The package is organized into several modules:

1. **changepoint_model.py**: Contains model definitions for various changepoint models including Poisson and Gaussian models for neural data
2. **changepoint_io.py**: Handles data loading, preprocessing, and result storage through the `FitHandler` and `DatabaseHandler` classes
3. **changepoint_analysis.py**: Provides tools for analyzing fitted models, including significance testing and visualization
4. **changepoint_preprocess.py**: Contains functions for data preprocessing, binning, and transformations
5. **utils/**: Contains utility functions for plotting, data handling, and batch processing

The core workflow in `pytau` involves:

1. Loading neural data (typically spike trains) using the `EphysData` class
2. Preprocessing the data for model fitting with functions from `changepoint_preprocess`
3. Defining and fitting changepoint models using PyMC3-based implementations in `changepoint_model`
4. Storing results in a queryable database managed by `DatabaseHandler`
5. Analyzing and visualizing the results with functions from `changepoint_analysis`

The mathematical foundation of the package is Bayesian changepoint detection. For a time series $X = \{x_1, x_2, ..., x_T\}$, we model the data as having $K$ distinct states with transitions at times $\tau = \{\tau_1, \tau_2, ..., \tau_{K-1}\}$. The emission distribution within each state $k$ is parameterized by $\theta_k$:

$$p(x_t | \theta_k) \textrm{ for } \tau_{k-1} < t \leq \tau_k$$

For neural spike train data, the package implements Poisson emission models:

$$x_t \sim \textrm{Poisson}(\lambda_k) \textrm{ for } \tau_{k-1} < t \leq \tau_k$$

Where $\lambda_k$ represents the firing rate in state $k$.

# Example usage

Below is a simple example of using `pytau` to fit a changepoint model to neural data:

```python
from pytau.changepoint_io import FitHandler

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

After fitting, the results can be analyzed using the `PklHandler` class:

```python
from pytau.changepoint_analysis import PklHandler

# Load fitted model
pkl_handler = PklHandler('/path/to/saved/model.pkl')

# Access model components
tau = pkl_handler.tau  # Changepoint times
firing = pkl_handler.firing  # Firing rate analysis

# Analyze significant neurons
significant_neurons = firing.anova_significant_neurons
```

This example demonstrates the streamlined workflow for fitting a changepoint model to taste response data and analyzing the results.

# Model types and features

`pytau` implements several types of changepoint models to accommodate different analysis needs:

1. **Single taste Poisson models**: For analyzing single-taste responses with Poisson emission distributions
   ```python
   # From changepoint_model.py
   single_taste_poisson(spike_array, states, **kwargs)
   ```

2. **Variable sigmoid models**: Models with learnable transition sharpness
   ```python
   # From changepoint_model.py
   single_taste_poisson_varsig(spike_array, states, **kwargs)
   ```

3. **Fixed sigmoid models**: Models with fixed transition sharpness
   ```python
   # From changepoint_model.py
   single_taste_poisson_varsig_fixed(spike_array, states, inds_span=1)
   ```

4. **All-taste models**: For analyzing responses across multiple tastes
   ```python
   # From changepoint_model.py
   all_taste_poisson(spike_array, states, **kwargs)
   ```

5. **Dirichlet process models**: For automatically determining the number of states
   ```python
   # From changepoint_model.py
   single_taste_poisson_dirichlet(spike_array, max_states=10, **kwargs)
   ```

The package also provides tools for statistical analysis of fitted models, including:

1. **State-dependent firing rate analysis**:
   ```python
   # From changepoint_analysis.py
   get_state_firing(spike_array, tau_array)
   ```

2. **Significance testing**:
   ```python
   # From changepoint_analysis.py
   calc_significant_neurons_firing(state_firing, p_val=0.05)
   ```

3. **Transition analysis**:
   ```python
   # From changepoint_analysis.py
   get_transition_snips(spike_array, tau_array, window_radius=300)
   ```

# Comparison with existing tools

Several tools exist for changepoint detection, including:

1. **ruptures**: A Python package for offline change point detection
2. **bayesloop**: A probabilistic programming framework for time series analysis
3. **PyChange**: A Python package for change point detection in time series

`pytau` differs from these tools in its specific focus on neuroscience applications, particularly for analyzing neural ensemble data across multiple experimental conditions. It provides specialized functionality for:

1. Handling multi-trial, multi-neuron spike train data
2. Batch processing across parameter grids
3. Database management for model comparison
4. Specialized visualization for neural data
5. Statistical analysis of state-dependent neural activity

While general-purpose changepoint detection tools are valuable, `pytau` addresses the specific needs of neuroscientists analyzing state transitions in neural population activity.

# Acknowledgements

We acknowledge contributions from collaborators and support during the development of this project. Special thanks to the PyMC3 development team for providing the Bayesian modeling framework that powers the core functionality of `pytau`.

# References
