# PyTau

This is a python package to perform streamlined, batched inference for changepoint models over parameter grids and datasets, and efficiently analyze sets of fitted models.

## Models

PyTau uses changepoint models to detect shifts in data distributions. These models are designed to identify points in time where the statistical properties of a sequence of observations change. The package supports various types of changepoint models, including Gaussian and Poisson models. Each model is optimized for different types of data and use cases.

### Available Models

#### Gaussian Changepoint Models
These models detect changes in both mean and variance or only in the mean for Gaussian data. They are suitable for continuous data where changes in statistical properties like mean and variance are of interest.

- `gaussian_changepoint_mean_var_2d`: Detects changes in both mean and variance.
- `gaussian_changepoint_mean_dirichlet`: Uses a Dirichlet process prior to determine the number of states, detecting changes only in the mean.
- `gaussian_changepoint_mean_2d`: Detects changes only in the mean.

#### Poisson Changepoint Models
These models are used for spike train data, detecting changes in the rate of events. They are ideal for count data where the rate of occurrence is the primary focus.

- `single_taste_poisson`: Models changepoints for single taste without hierarchical structure.
- `single_taste_poisson_dirichlet`: Uses a Dirichlet process prior for single taste spike train data.
- `single_taste_poisson_varsig`: Uses variable sigmoid slope inferred from data.
- `single_taste_poisson_varsig_fixed`: Uses sigmoid with given slope.
- `all_taste_poisson`: Fits changepoints across multiple tastes.
- `all_taste_poisson_varsig_fixed`: Fits changepoints across multiple tastes with fixed sigmoid slope.
- `single_taste_poisson_trial_switch`: Assumes only emissions change across trials.
- `all_taste_poisson_trial_switch`: Fits changepoints across multiple tastes with trial switching.

### Fitting Methods
PyTau provides several fitting methods for these models:

- **ADVI (Automatic Differentiation Variational Inference)**: A method for approximating the posterior distribution.
- **MCMC (Markov Chain Monte Carlo)**: A sampling method for drawing samples from the posterior distribution.
- **DPP (Dirichlet Process Prior)**: Used for models with a Dirichlet process prior to determine the number of states.

## Installation

To set up PyTau, follow these steps:

```bash
# Create and activate conda environment
conda create -n "pytau_env" python=3.6.13 ipython notebook -y
conda activate pytau_env

# Clone repository
cd ~/Desktop
git clone https://github.com/abuzarmahmood/pytau.git

# Install requirements from specified file
cd pytau
pip install -r requirements.txt

# Test everything is working by running notebook
cd pytau/how_to
bash scripts/download_test_data.sh
cd notebooks
jupyter notebook
# Run a notebook
```

## Usage

PyTau provides a streamlined workflow for fitting changepoint models to neural data. Here's a basic example of how to use the package:

```python
import pytau
from pytau.changepoint_model import single_taste_poisson, advi_fit

# Load your spike data (trials x neurons x time)
spike_array = load_your_data()

# Create a model with desired number of states
model = single_taste_poisson(spike_array, states=3)

# Fit the model using ADVI
model, approx, lambda_stack, tau_samples, data = advi_fit(model, fit=10000, samples=1000)

# Analyze the results
# tau_samples contains the changepoint times
# lambda_stack contains the emission rates
```

### Loading Saved Models

To load and analyze previously saved models:

```python
from pytau.changepoint_analysis import PklHandler

# Load a saved model
handler = PklHandler('/path/to/saved_model.pkl')

# Access ELBO history (new in v0.1.3)
last_elbo = handler.elbo_hist[-1]  # Get the final ELBO value
all_elbo = handler.elbo_hist        # Get the complete ELBO history

# Access other model components
tau_samples = handler.tau_array
lambda_samples = handler.lambda_array
processed_data = handler.processed_spikes
```

For more detailed usage examples, refer to the notebooks in the `pytau/how_to/notebooks` directory.

## Workflows

A typical workflow with PyTau involves the following steps:

1. **Data Preparation**: Organize your spike train data into the required format (typically trials x neurons x time).
2. **Model Selection**: Choose the appropriate changepoint model based on your data characteristics (e.g., single_taste_poisson for single taste data).
3. **Inference**: Run the model to detect changepoints using methods like ADVI or MCMC.
4. **Analysis**: Analyze the results to understand the detected changepoints and their implications.

PyTau also provides tools for batch processing and database management to streamline the analysis of large datasets:

- `FitHandler`: Handles the pipeline of model fitting including loading data, preprocessing, fitting models, and saving results.
- `DatabaseHandler`: Manages transactions with the model database, allowing for efficient storage and retrieval of fitted models.
- `PklHandler`: Loads and provides convenient access to saved model files, including the new `elbo_hist` property for easy access to ELBO values.

## Data Organization

PyTau organizes fitted models in a central location, accessed by indexing an info file that contains model parameters and metadata. The metadata is also stored in the model file to enable recreation of the info file if needed.

The database for fitted models includes columns for:
- Model save path
- Animal Name
- Session date
- Taste Name
- Region Name
- Experiment name
- Fit Date
- Model parameters (Model Type, Data Type, States Num, Fit steps, Time lims, Bin Width)

Data stored in models includes:
- Model
- Approximation
- Lambda (emission rates)
- Tau (changepoint times)
- Data used to fit model
- Raw data (before preprocessing)
- Model and data details/parameters

## Pipeline

PyTau's pipeline consists of several components:

1. **Model file**: Generates models and performs inference (e.g., changepoint_model.py)
2. **Data pre-processing code**: Handles different data types (shuffle, simulated, actual)
3. **I/O helper code**: Loads data, processes it, and feeds it to modeling code
4. **Run script**: Iterates over data for batch processing

## Parallelization

Parallelization is currently performed using GNU Parallel by setting a separate Theano compiledir for each job. This prevents compilation clashes and allows for efficient batch processing of multiple models.
