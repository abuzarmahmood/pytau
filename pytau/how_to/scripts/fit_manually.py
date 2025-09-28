# Fit a model to data manually

import os
import sys
from glob import glob

import numpy as np
import pylab as plt
import tables
from scipy import stats

from pytau.changepoint_model import advi_fit, single_taste_poisson
from pytau.changepoint_preprocess import preprocess_single_taste
from pytau.utils import plotting

try:
    pytau_base_dir = os.path.dirname(os.path.abspath(__file__))
except:
    pytau_base_dir = os.path.dirname(os.getcwd())
print(f"Using PyTAU base dir: {pytau_base_dir}")

# Write to MODEL_SAVE_DIR.params
param_file_path = f"{pytau_base_dir}/pytau/config/MODEL_SAVE_DIR.params"
save_model_path = f"{pytau_base_dir}/pytau/how_to/examples/saved_models"
with open(param_file_path, "w") as f:
    f.write(save_model_path)

# Find hf5 file
h5_path = glob(os.path.join(pytau_base_dir, "**", "*.h5"), recursive=True)[0]

# Load spikes
wanted_dig_in_ind = 0
with tables.open_file(h5_path, "r") as hf5:
    dig_in_list = hf5.list_nodes("/spike_trains")
    wanted_dig_in = dig_in_list[wanted_dig_in_ind]
    spike_train = wanted_dig_in.spike_array[:]

# Reshape + Bin Spikes
time_lims = [2000, 4000]
bin_width = 50
binned_spike_array = preprocess_single_taste(
    spike_array=spike_train,
    time_lims=time_lims,
    bin_width=bin_width,
    data_transform=None,
)

# Create and fit model
n_fit = 40000
n_samples = 20000
n_states = 4
model = single_taste_poisson(binned_spike_array, n_states)
model, approx, lambda_stack, tau_samples, fit_data = advi_fit(
    model=model, fit=n_fit, samples=n_samples
)

# Extract changepoint values
int_tau = np.vectorize(np.int)(tau_samples)
mode_tau = np.squeeze(stats.mode(int_tau, axis=0)[0])
scaled_mode_tau = (mode_tau * bin_width) + time_lims[0]

# Plot ELBO over iterations, should be flat by the end
fig, ax = plotting.plot_elbo_history(approx)
plt.show()

# Overlay raster plot with states
fig, ax = plotting.plot_changepoint_raster(
    spike_train, scaled_mode_tau, [1500, 4000])
plt.show()

# Overview of changepoint positions
fig, ax = plotting.plot_changepoint_overview(scaled_mode_tau, [1500, 4000])
plt.show()

# Aligned spiking
fig, ax = plotting.plot_aligned_state_firing(spike_train, scaled_mode_tau, 300)
plt.show()

# Plot mean firing rates per state
fig, ax = plotting.plot_state_firing_rates(spike_train, scaled_mode_tau)
plt.show()
