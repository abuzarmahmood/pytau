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
    orcid: 0000-0002-2448-4818
    affiliation: "1,2"
affiliations:
 - name: Swartz Foundation Computational Neuroscience Fellow, Volen Center for Complex Systems, Brandeis University, Waltham, MA, USA
   index: 1
 - name: Department of Psychology, Brandeis University, Waltham, MA, USA
   index: 2
date: 30 March 2025
bibliography: paper.bib
---

# Summary

Neural activity often exhibits sharp transitions between distinct states, captured by changepoint models [@Seidemann1994; @Jones2007; @Saravani2019]. `pytau` is a Python package for batched Bayesian changepoint inference across parameter grids and datasets. It integrates with PyMC3 to provide uncertainty estimates critical for noisy neuroscience data with small sample sizes and low channel counts, and includes tools for preprocessing, model fitting, result visualization, and statistical analysis. The package has been successfully applied in published research examining taste processing [@Mahmood2023; @Mahmood2025] and taste aversion learning [@Flores2023], and is currently being utilized in several ongoing neuroscience studies [@MazzioInPrep; @BaasThomasInPrep; @CaliaBoganInPrep; @Maigler2024].

# Statement of need

Understanding how neural populations encode information often involves analyzing activity changes over time across different experimental conditions, parameters, or subjects. Fitting and comparing Bayesian changepoint models across numerous datasets or parameter settings is computationally intensive and logistically challenging. Existing changepoint detection tools lack the specialized functionality needed for neuroscience applications, particularly for handling multi-trial, multi-neuron spike train data.

`pytau` addresses this gap by providing a modularized pipeline specifically designed for neuroscience data. The package offers several key advantages:

1. **Batch processing**: Automates model fitting across multiple datasets and parameter configurations
2. **Database management**: Organizes and tracks model fits for easy retrieval and comparison
3. **Visualization tools**: Provides specialized plotting functions including raster plots with overlaid changepoints, state-dependent firing rate visualizations, and transition-aligned activity plots
4. **Statistical analysis**: Includes tools for significance testing of state-dependent neural activity, such as ANOVA-based detection of neurons with significant state-dependent firing and pairwise t-tests for transition-triggered activity

Its adoption in studies of taste processing [@Mahmood2023; @Mahmood2025; @Maigler2024], taste aversion learning [@Flores2023], and ingestive behavior [@BaasThomasInPrep] demonstrates its practical utility for researchers studying state transitions in neural activity. Detailed documentation is available at [https://abuzarmahmood.github.io/pytau/](https://abuzarmahmood.github.io/pytau/).

![**Spike rasters with changepoint overlays** visualize inferred changepoints across trials and neurons](figs/state_raster_overlay.png)

# Implementation and architecture

`pytau` is implemented in Python and built on NumPy, SciPy, PyMC3, and Matplotlib [@numpy; @pymc3]. The package is organized into several modules:

1. **changepoint_model.py**: Contains model definitions for various changepoint models including Poisson and Gaussian models (see [Available Models](https://abuzarmahmood.github.io/pytau/models/))
2. **changepoint_io.py**: Handles data loading, preprocessing, and result storage through `FitHandler` and `DatabaseHandler` classes
3. **changepoint_analysis.py**: Provides tools for analyzing fitted models, including significance testing and visualization
4. **changepoint_preprocess.py**: Contains functions for data preprocessing, binning, and transformations (see [Data Formats](https://abuzarmahmood.github.io/pytau/data_formats/))

The mathematical foundation is Bayesian changepoint detection. For a time series $X = \{x_1, x_2, ..., x_T\}$, the package models $K$ states with transitions at $\tau = \{\tau_1, \tau_2, ..., \tau_{K-1}\}$ and Poisson emissions: $x_t \sim \textrm{Poisson}(\lambda_k)$ for $\tau_{k-1} < t \leq \tau_k$, where $\lambda_k$ is the firing rate in state $k$.

The package employs Automatic Differentiation Variational Inference (ADVI) [@kucukelbir2016automatic] for fast posterior approximation and Markov Chain Monte Carlo (MCMC) with the No-U-Turn Sampler (NUTS) [@hoffman2011no] for precise inference (see [Inference Methods](https://abuzarmahmood.github.io/pytau/inference/)). A key feature is modeling state transitions with sigmoid functions, which enables continuous parameter exploration and detection of gradual changes in neural dynamics.

# Example usage

`pytau` provides a streamlined workflow through the `FitHandler` class for model fitting and `PklHandler` for analysis:

```python
from pytau.changepoint_io import FitHandler

# Initialize fit handler
fh = FitHandler(data_dir='/path/to/data', taste_num=1,
                region_name='GC', experiment_name='example')

# Set preprocessing and model parameters
fh.set_preprocess_params(time_lims=[0, 2000], bin_width=10)
fh.set_model_params(states=3, fit=5000, samples=1000)

# Run the full pipeline
fh.load_spike_trains()
fh.preprocess_data()
fh.create_model()
fh.run_inference()
fh.save_fit_output()
```

After fitting, results can be analyzed using the `PklHandler` class:

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

The package includes visualization tools for examining neural activity with overlaid changepoints, analyzing state-dependent firing rates, and visualizing transition-aligned activity. Comprehensive tutorials and detailed examples are available in the [documentation](https://abuzarmahmood.github.io/pytau/examples/), including Jupyter notebooks and example scripts with test datasets in the repository's `how_to` directory.

![**State timing overview** shows state durations and transition time distributions across trials](figs/state_timing_overview.png)







# State of the field

Several tools exist for changepoint detection, including `ruptures` [@ruptures] for offline change point detection, `bayesloop` for probabilistic time series analysis, `PyChange` for general time series changepoint detection, and Bayesian online changepoint detection methods [@adams2007bayesian; @fearnhead2007line]. While these general-purpose tools are valuable, `pytau` differs by focusing specifically on neuroscience applications. It provides specialized functionality for handling multi-trial, multi-neuron spike train data, batch processing across parameter grids, database management for model comparison, and neuroscience-specific visualization and statistical analysis tools. This specialization makes `pytau` particularly suited for researchers analyzing state transitions in neural population activity across different experimental paradigms.

# Acknowledgements

We acknowledge support from the Katz Lab and the PyMC development team.

# References
