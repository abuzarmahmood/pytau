import numpy as np
import os
from pytau.changepoint_analysis import (
    get_transition_snips,
    get_state_firing,
    calc_significant_neurons_firing,
    calc_significant_neurons_snippets,
    _firing,
    _tau,
    PklHandler
)

def test_get_transition_snips():
    spike_array = np.random.rand(5, 10, 1000)  # Example spike array
    tau_array = np.array([[100, 300], [200, 400], [150, 350], [250, 450], [300, 500]])
    window_radius = 50
    result = get_transition_snips(spike_array, tau_array, window_radius)
    assert result.shape == (5, 10, 2 * window_radius, 2)

def test_get_state_firing():
    spike_array = np.random.rand(5, 10, 1000)
    tau_array = np.array([[100, 300], [200, 400], [150, 350], [250, 450], [300, 500]])
    result = get_state_firing(spike_array, tau_array)
    assert result.shape == (5, 3, 10)

def test_calc_significant_neurons_firing():
    state_firing = np.random.rand(5, 3, 10)
    p_val = 0.05
    anova_p_val_array, anova_sig_neurons = calc_significant_neurons_firing(state_firing, p_val)
    assert len(anova_p_val_array) == 10

def test_calc_significant_neurons_snippets():
    transition_snips = np.random.rand(5, 10, 100, 2)
    p_val = 0.05
    pairwise_p_val_array, pairwise_sig_neurons = calc_significant_neurons_snippets(transition_snips, p_val)
    assert pairwise_p_val_array.shape == (10, 2)

def test_firing_class():
    tau_instance = _tau(np.random.rand(5, 2), {"preprocess": {"time_lims": [0, 1000], "bin_width": 1}})
    processed_spikes = np.random.rand(5, 10, 1000)
    metadata = {"data": {"data_dir": "/path/to/data", "region_name": "region", "taste_num": "all"}}
    firing_instance = _firing(tau_instance, processed_spikes, metadata)
    assert firing_instance.state_firing.shape == (5, 3, 10)

def test_tau_class():
    tau_array = np.random.rand(5, 2)
    metadata = {"preprocess": {"time_lims": [0, 1000], "bin_width": 1}}
    tau_instance = _tau(tau_array, metadata)
    assert tau_instance.raw_tau.shape == (5, 2)

def test_pkl_handler():
    file_path = "/path/to/fake.pkl"
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            pass  # Create an empty file for testing
    handler = PklHandler(file_path)
    assert handler.file_name_base == "fake"
