import numpy as np
import pytest
from pytau.changepoint_preprocess import preprocess_single_taste, preprocess_all_taste

def test_preprocess_single_taste_trial_shuffled():
    spike_array = np.random.rand(10, 5, 100)
    time_lims = [0, 100]
    bin_width = 10
    result = preprocess_single_taste(spike_array, time_lims, bin_width, 'trial_shuffled')
    assert result.shape == (10, 5, 10)

def test_preprocess_single_taste_spike_shuffled():
    spike_array = np.random.rand(10, 5, 100)
    time_lims = [0, 100]
    bin_width = 10
    result = preprocess_single_taste(spike_array, time_lims, bin_width, 'spike_shuffled')
    assert result.shape == (10, 5, 10)

def test_preprocess_single_taste_simulated():
    spike_array = np.random.rand(10, 5, 100)
    time_lims = [0, 100]
    bin_width = 10
    result = preprocess_single_taste(spike_array, time_lims, bin_width, 'simulated')
    assert result.shape == (10, 5, 10)

def test_preprocess_single_taste_none():
    spike_array = np.random.rand(10, 5, 100)
    time_lims = [0, 100]
    bin_width = 10
    result = preprocess_single_taste(spike_array, time_lims, bin_width, None)
    assert result.shape == (10, 5, 10)

def test_preprocess_all_taste_trial_shuffled():
    spike_array = np.random.rand(3, 10, 5, 100)
    time_lims = [0, 100]
    bin_width = 10
    result = preprocess_all_taste(spike_array, time_lims, bin_width, 'trial_shuffled')
    assert result.shape == (3, 10, 5, 10)

def test_preprocess_all_taste_spike_shuffled():
    spike_array = np.random.rand(3, 10, 5, 100)
    time_lims = [0, 100]
    bin_width = 10
    result = preprocess_all_taste(spike_array, time_lims, bin_width, 'spike_shuffled')
    assert result.shape == (3, 10, 5, 10)

def test_preprocess_all_taste_simulated():
    spike_array = np.random.rand(3, 10, 5, 100)
    time_lims = [0, 100]
    bin_width = 10
    result = preprocess_all_taste(spike_array, time_lims, bin_width, 'simulated')
    assert result.shape == (3, 10, 5, 10)

def test_preprocess_all_taste_none():
    spike_array = np.random.rand(3, 10, 5, 100)
    time_lims = [0, 100]
    bin_width = 10
    result = preprocess_all_taste(spike_array, time_lims, bin_width, None)
    assert result.shape == (3, 10, 5, 10)
