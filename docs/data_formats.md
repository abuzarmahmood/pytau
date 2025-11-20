# Data Formats

PyTau is designed to work with various neural data formats and experimental designs. This page describes the expected data formats, preprocessing options, and best practices for preparing your data.

## Input Data Shapes

PyTau models accept data in different dimensionalities depending on the experimental design:

### 1D: Single Time Series
**Shape**: `(time,)`

**Use Case**: Single neuron or single trial analysis

**Example**:
```python
import numpy as np

# Single neuron spike counts over time
data_1d = np.array([5, 3, 7, 2, 1, 0, 0, 1, 2, 4])
# 10 time bins

from pytau.changepoint_model import PoissonChangepoint1D
model = PoissonChangepoint1D(data_1d, states=2)
```

### 2D: Multiple Trials or Neurons
**Shape**: `(trials, time)` or `(neurons, time)`

**Use Case**: 
- Multiple trials of a single neuron
- Multiple neurons recorded simultaneously
- Single condition experiments

**Example**:
```python
# 20 trials, 100 time bins
data_2d = np.random.poisson(lam=5, size=(20, 100))

from pytau.changepoint_model import single_taste_poisson
model = single_taste_poisson(data_2d, states=3)
```

### 3D: Trials × Neurons × Time
**Shape**: `(trials, neurons, time)`

**Use Case**: 
- Multi-neuron recordings
- Single stimulus/condition
- Population analysis

**Example**:
```python
# 30 trials, 15 neurons, 100 time bins
data_3d = np.random.poisson(lam=3, size=(30, 15, 100))

from pytau.changepoint_model import single_taste_poisson
model = single_taste_poisson(data_3d, states=3)
```

### 4D: Tastes × Trials × Neurons × Time
**Shape**: `(tastes, trials, neurons, time)`

**Use Case**:
- Multi-stimulus experiments
- Taste/odor discrimination studies
- Condition comparison

**Example**:
```python
# 4 tastes, 30 trials per taste, 15 neurons, 100 time bins
data_4d = np.random.poisson(lam=4, size=(4, 30, 15, 100))

from pytau.changepoint_model import all_taste_poisson
model = all_taste_poisson(data_4d, states=3)
```

## Data Types

### Spike Count Data (Poisson Models)

**Format**: Integer counts per time bin

**Preprocessing**:
```python
import numpy as np

# Raw spike times (in seconds)
spike_times = np.array([0.1, 0.15, 0.23, 0.45, 0.67, ...])

# Bin into counts
bin_width = 0.05  # 50 ms bins
time_bins = np.arange(0, 2.0, bin_width)
spike_counts, _ = np.histogram(spike_times, bins=time_bins)

# spike_counts is now ready for PyTau
```

**Best Practices**:
- Choose bin width based on firing rate (typically 10-100 ms)
- Ensure sufficient counts per bin (avoid too many zeros)
- Consider smoothing for very sparse data

### Continuous Data (Gaussian Models)

**Format**: Floating-point values

**Examples**:
- Local field potentials (LFP)
- Calcium imaging traces
- Behavioral measurements

**Preprocessing**:
```python
# Raw continuous signal
lfp_signal = load_lfp_data()

# Downsample if needed
from scipy.signal import resample
downsampled = resample(lfp_signal, num=100)

# Normalize (optional but recommended)
normalized = (downsampled - downsampled.mean()) / downsampled.std()

from pytau.changepoint_model import gaussian_changepoint_mean_2d
model = gaussian_changepoint_mean_2d(normalized, states=3)
```

**Best Practices**:
- Normalize data (z-score or min-max)
- Remove artifacts and outliers
- Consider filtering (bandpass, lowpass)

### Categorical Data (Categorical Models)

**Format**: Integer category labels

**Examples**:
- Behavioral states
- Decoded neural states
- Discrete choices

**Preprocessing**:
```python
# Behavioral states: 0=rest, 1=approach, 2=consume
behavior = np.array([0, 0, 0, 1, 1, 2, 2, 2, 0, 0])

from pytau.changepoint_model import categorical_changepoint_2d
model = categorical_changepoint_2d(behavior, states=2, n_categories=3)
```

**Best Practices**:
- Use consecutive integers starting from 0
- Ensure all categories are represented
- Consider temporal resolution

## Data Preprocessing

PyTau provides built-in preprocessing options for different analysis needs:

### Actual Data
Use your real experimental data as-is:
```python
from pytau.changepoint_io import FitHandler

handler = FitHandler(
    data_type='actual',
    # ... other parameters
)
```

### Shuffled Data
Shuffle trials to create null distribution:
```python
handler = FitHandler(
    data_type='shuffled',
    # Randomly shuffles trial order
)
```

### Simulated Data
Generate synthetic data for testing:
```python
handler = FitHandler(
    data_type='simulated',
    # Generates data from known changepoint model
)
```

## Time Binning

### Choosing Bin Width

The bin width affects model performance and interpretation:

**Too Small** (< 10 ms):
- Many zero counts
- Noisy estimates
- Slower inference

**Too Large** (> 200 ms):
- Loss of temporal resolution
- Missed fast transitions
- Oversmoothing

**Recommended**:
- Start with 50 ms bins
- Adjust based on firing rate
- Higher firing rate → smaller bins possible

### Binning Example

```python
import numpy as np

def bin_spike_times(spike_times, bin_width, time_range):
    """
    Bin spike times into counts.
    
    Parameters
    ----------
    spike_times : array-like
        Spike times in seconds
    bin_width : float
        Width of each bin in seconds
    time_range : tuple
        (start_time, end_time) in seconds
    
    Returns
    -------
    spike_counts : ndarray
        Binned spike counts
    bin_edges : ndarray
        Edges of time bins
    """
    bin_edges = np.arange(time_range[0], time_range[1], bin_width)
    spike_counts, _ = np.histogram(spike_times, bins=bin_edges)
    return spike_counts, bin_edges

# Example usage
spike_times = np.array([0.1, 0.15, 0.23, 0.45, 0.67, 0.89, 1.2, 1.5])
counts, edges = bin_spike_times(spike_times, bin_width=0.1, time_range=(0, 2))
```

## Data Quality Checks

### Check for Issues

```python
import numpy as np

def check_data_quality(data):
    """Check data for common issues."""
    
    # Check for NaNs
    if np.any(np.isnan(data)):
        print("⚠️ Warning: Data contains NaN values")
    
    # Check for infinities
    if np.any(np.isinf(data)):
        print("⚠️ Warning: Data contains infinite values")
    
    # Check for negative values (spike counts should be non-negative)
    if np.any(data < 0):
        print("⚠️ Warning: Data contains negative values")
    
    # Check for excessive zeros
    zero_fraction = np.mean(data == 0)
    if zero_fraction > 0.8:
        print(f"⚠️ Warning: {zero_fraction*100:.1f}% of data is zero")
    
    # Check variance
    if np.var(data) < 1e-10:
        print("⚠️ Warning: Data has very low variance")
    
    print("✅ Data quality check complete")

# Example
check_data_quality(spike_counts)
```

### Handle Missing Data

```python
# Remove trials with missing data
def remove_incomplete_trials(data):
    """Remove trials containing NaN or Inf."""
    # Assuming data shape is (trials, neurons, time)
    valid_trials = ~np.any(np.isnan(data) | np.isinf(data), axis=(1, 2))
    return data[valid_trials]

cleaned_data = remove_incomplete_trials(data_3d)
```

## Data Organization

### File Structure

Organize your data files consistently:

```
project/
├── data/
│   ├── animal1/
│   │   ├── session1/
│   │   │   ├── spikes.h5
│   │   │   └── metadata.json
│   │   └── session2/
│   │       ├── spikes.h5
│   │       └── metadata.json
│   └── animal2/
│       └── ...
└── models/
    └── fitted_models/
```

### HDF5 Format

PyTau works well with HDF5 files for large datasets:

```python
import h5py

# Save data
with h5py.File('spikes.h5', 'w') as f:
    f.create_dataset('spike_counts', data=spike_counts)
    f.create_dataset('time_bins', data=time_bins)
    f.attrs['bin_width'] = 0.05
    f.attrs['animal'] = 'mouse1'
    f.attrs['session'] = '2024-01-15'

# Load data
with h5py.File('spikes.h5', 'r') as f:
    spike_counts = f['spike_counts'][:]
    time_bins = f['time_bins'][:]
    bin_width = f.attrs['bin_width']
```

## Example Workflows

### Single Neuron Analysis

```python
import numpy as np
from pytau.changepoint_model import PoissonChangepoint1D, advi_fit

# Load spike times
spike_times = load_spike_times('neuron1.npy')

# Bin spikes
bin_width = 0.05  # 50 ms
time_range = (0, 2.0)  # 2 seconds
bin_edges = np.arange(time_range[0], time_range[1], bin_width)
spike_counts, _ = np.histogram(spike_times, bins=bin_edges)

# Fit model
model = PoissonChangepoint1D(spike_counts, states=3)
model, approx = advi_fit(model, fit=10000, samples=5000)
```

### Multi-Trial Population Analysis

```python
import numpy as np
from pytau.changepoint_model import single_taste_poisson, advi_fit

# Load data: (trials, neurons, time)
data = load_population_data('session1.h5')

# Check data quality
check_data_quality(data)

# Fit model
model = single_taste_poisson(data, states=4)
model, approx = advi_fit(model, fit=10000, samples=5000)
```

### Multi-Taste Experiment

```python
import numpy as np
from pytau.changepoint_model import all_taste_poisson, advi_fit

# Load data: (tastes, trials, neurons, time)
data = load_multitaste_data('experiment1.h5')

# Fit hierarchical model
model = all_taste_poisson(data, states=3)
model, approx = advi_fit(model, fit=15000, samples=5000)
```

## Common Issues and Solutions

### Issue: Too Many Zeros
**Solution**: Increase bin width or combine neurons

### Issue: Unequal Trial Lengths
**Solution**: Pad with NaN and mask, or truncate to shortest trial

### Issue: Different Sampling Rates
**Solution**: Resample to common rate before analysis

### Issue: Outlier Trials
**Solution**: Use robust preprocessing or remove outliers

### Issue: Memory Errors
**Solution**: Process data in batches or reduce dimensionality

## Best Practices Summary

1. **Choose appropriate bin width** based on firing rate
2. **Normalize continuous data** for better convergence
3. **Check data quality** before fitting models
4. **Use consistent data organization** across experiments
5. **Save preprocessing parameters** for reproducibility
6. **Start with simple models** on subset of data
7. **Validate results** with held-out data

## Next Steps

- See [Available Models](models.md) for model selection
- See [Inference Methods](inference.md) for fitting models
- See [Pipeline Architecture](pipeline.md) for batch processing
- Check the [API documentation](api.md) for detailed function signatures
