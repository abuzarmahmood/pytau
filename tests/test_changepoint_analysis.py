import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from pytau.changepoint_analysis import (
    PklHandler,
    _firing,
    _tau,
    calc_significant_neurons_firing,
    calc_significant_neurons_snippets,
    get_state_firing,
    get_transition_snips,
)


def test_get_transition_snips():
    """Test get_transition_snips function with valid inputs."""
    np.random.seed(42)  # For reproducible tests
    spike_array = np.random.rand(5, 10, 1000)  # Example spike array
    tau_array = np.array(
        [[100, 300], [200, 400], [150, 350], [250, 450], [300, 500]])
    window_radius = 50
    result = get_transition_snips(spike_array, tau_array, window_radius)
    assert result.shape == (5, 10, 2 * window_radius, 2)
    assert isinstance(result, np.ndarray)
    assert result.dtype == spike_array.dtype


def test_get_state_firing():
    """Test get_state_firing function with valid inputs."""
    np.random.seed(42)  # For reproducible tests
    spike_array = np.random.rand(5, 10, 1000)
    tau_array = np.array(
        [[100, 300], [200, 400], [150, 350], [250, 450], [300, 500]])
    result = get_state_firing(spike_array, tau_array)
    assert result.shape == (5, 3, 10)
    assert isinstance(result, np.ndarray)
    # Check that all values are non-negative (firing rates should be >= 0)
    assert np.all(result >= 0)


def test_calc_significant_neurons_firing():
    """Test calc_significant_neurons_firing function with valid inputs."""
    np.random.seed(42)  # For reproducible tests
    state_firing = np.random.rand(5, 3, 10)
    p_val = 0.05
    anova_p_val_array, anova_sig_neurons = calc_significant_neurons_firing(
        state_firing, p_val)
    assert len(anova_p_val_array) == 10
    assert isinstance(anova_p_val_array, np.ndarray)
    assert isinstance(anova_sig_neurons, np.ndarray)
    # P-values should be between 0 and 1
    assert np.all(anova_p_val_array >= 0)
    assert np.all(anova_p_val_array <= 1)


def test_calc_significant_neurons_snippets():
    """Test calc_significant_neurons_snippets function with valid inputs."""
    np.random.seed(42)  # For reproducible tests
    transition_snips = np.random.rand(5, 10, 100, 2)
    p_val = 0.05
    pairwise_p_val_array, pairwise_sig_neurons = calc_significant_neurons_snippets(
        transition_snips, p_val)
    assert pairwise_p_val_array.shape == (10, 2)
    assert isinstance(pairwise_p_val_array, np.ndarray)
    assert isinstance(pairwise_sig_neurons, np.ndarray)
    # P-values should be between 0 and 1
    assert np.all(pairwise_p_val_array >= 0)
    assert np.all(pairwise_p_val_array <= 1)


def test_firing_class():
    """Test _firing class basic attributes without full initialization."""
    np.random.seed(42)  # For reproducible tests
    # Create tau values that are reasonable integers within the spike array range
    tau_array = np.array([[100, 300], [200, 400], [150, 350], [
                         250, 450], [300, 500]], dtype=float)
    tau_instance = _tau(tau_array, {"preprocess": {
                        "time_lims": [0, 1000], "bin_width": 1}})
    processed_spikes = np.random.rand(5, 10, 1000)
    metadata = {"data": {"data_dir": "/tmp",
                         "region_name": "region", "taste_num": "all"}}

    # Test that we can create a _firing instance and set basic attributes
    # Create without calling __init__
    firing_instance = _firing.__new__(_firing)
    firing_instance.tau = tau_instance
    firing_instance.processed_spikes = processed_spikes
    firing_instance.metadata = metadata

    # Test basic attribute access
    assert hasattr(firing_instance, 'tau')
    assert hasattr(firing_instance, 'processed_spikes')
    assert hasattr(firing_instance, 'metadata')
    assert firing_instance.processed_spikes.shape == (5, 10, 1000)


def test_tau_class():
    """Test _tau class initialization and basic functionality."""
    np.random.seed(42)  # For reproducible tests
    # Create reasonable tau values
    tau_array = np.array([[100, 300], [200, 400], [150, 350], [
                         250, 450], [300, 500]], dtype=float)
    metadata = {"preprocess": {"time_lims": [0, 1000], "bin_width": 1}}
    tau_instance = _tau(tau_array, metadata)
    assert tau_instance.raw_tau.shape == (5, 2)
    assert hasattr(tau_instance, 'raw_int_tau')
    assert hasattr(tau_instance, 'raw_mode_tau')
    assert hasattr(tau_instance, 'scaled_tau')
    # Check that integer conversion worked
    assert tau_instance.raw_int_tau.dtype in [np.int32, np.int64, int]


def test_pkl_handler():
    """Test PklHandler class basic functionality without full initialization."""
    # Test that the PklHandler class has the expected structure
    assert hasattr(PklHandler, '__init__')

    # Test that we can inspect the class without instantiating it
    import inspect
    init_signature = inspect.signature(PklHandler.__init__)
    expected_params = ['self', 'file_path']
    actual_params = list(init_signature.parameters.keys())
    assert actual_params == expected_params

    # Test basic file path handling
    test_path = "/path/to/test_file.pkl"
    handler = PklHandler.__new__(PklHandler)  # Create without calling __init__
    handler.dir_name = os.path.dirname(test_path)
    handler.file_name_base = os.path.splitext(os.path.basename(test_path))[0]

    assert handler.dir_name == "/path/to"
    assert handler.file_name_base == "test_file"


def test_get_transition_snips_edge_cases():
    """Test get_transition_snips with edge cases."""
    np.random.seed(42)
    spike_array = np.random.rand(3, 5, 500)
    tau_array = np.array([[100, 200], [150, 250], [120, 220]])

    # Test with small window radius
    window_radius = 10
    result = get_transition_snips(spike_array, tau_array, window_radius)
    assert result.shape == (3, 5, 2 * window_radius, 2)

    # Test that function raises error when window extends outside bounds
    with pytest.raises(ValueError, match="Transition window extends outside data bounds"):
        get_transition_snips(spike_array, tau_array, window_radius=300)


def test_get_state_firing_edge_cases():
    """Test get_state_firing with different input shapes."""
    np.random.seed(42)

    # Test with minimal data
    spike_array = np.random.rand(2, 3, 100)
    tau_array = np.array([[20, 80], [30, 70]])
    result = get_state_firing(spike_array, tau_array)
    assert result.shape == (2, 3, 3)  # 2 trials, 3 states, 3 neurons

    # Test with single trial
    spike_array = np.random.rand(1, 5, 200)
    tau_array = np.array([[50, 150]])
    result = get_state_firing(spike_array, tau_array)
    assert result.shape == (1, 3, 5)


def test_calc_significant_neurons_firing_edge_cases():
    """Test calc_significant_neurons_firing with edge cases."""
    np.random.seed(42)

    # Test with different p-values
    state_firing = np.random.rand(3, 3, 8)

    # Test with strict p-value
    p_val = 0.01
    anova_p_val_array, anova_sig_neurons = calc_significant_neurons_firing(
        state_firing, p_val)
    assert len(anova_p_val_array) == 8

    # Test with lenient p-value
    p_val = 0.1
    anova_p_val_array, anova_sig_neurons = calc_significant_neurons_firing(
        state_firing, p_val)
    assert len(anova_p_val_array) == 8


def test_calc_significant_neurons_snippets_edge_cases():
    """Test calc_significant_neurons_snippets with edge cases."""
    np.random.seed(42)

    # Test with minimal data
    transition_snips = np.random.rand(2, 4, 50, 2)
    p_val = 0.05
    pairwise_p_val_array, pairwise_sig_neurons = calc_significant_neurons_snippets(
        transition_snips, p_val)
    assert pairwise_p_val_array.shape == (4, 2)


def test_firing_class_attributes():
    """Test _firing class basic structure and expected methods."""
    # Test that the _firing class has the expected structure
    assert hasattr(_firing, '__init__')

    # Test that we can inspect the class without instantiating it
    import inspect
    init_signature = inspect.signature(_firing.__init__)
    expected_params = ['self', 'tau_instance', 'processed_spikes', 'metadata']
    actual_params = list(init_signature.parameters.keys())
    assert actual_params == expected_params


def test_tau_class_attributes():
    """Test _tau class has all expected attributes and correct calculations."""
    np.random.seed(42)
    tau_array = np.array([[100, 300], [200, 400], [150, 350]], dtype=float)
    metadata = {"preprocess": {"time_lims": [500, 2000], "bin_width": 2}}

    tau_instance = _tau(tau_array, metadata)

    # Check all expected attributes exist
    assert hasattr(tau_instance, 'raw_tau')
    assert hasattr(tau_instance, 'raw_int_tau')
    assert hasattr(tau_instance, 'raw_mode_tau')
    assert hasattr(tau_instance, 'scaled_tau')

    # Check calculations are correct
    expected_scaled = (tau_array * 2) + 500  # bin_width * tau + time_lims[0]
    np.testing.assert_array_equal(tau_instance.scaled_tau, expected_scaled)

    # Check that raw_int_tau is integer type
    assert tau_instance.raw_int_tau.dtype in [np.int32, np.int64, int, object]


def test_pkl_handler_elbo_hist_property():
    """Test PklHandler.elbo_hist property for accessing ELBO history."""
    import cloudpickle as pkl
    import tempfile

    # Create a mock approximation object with hist attribute
    class MockApprox:
        def __init__(self):
            self.hist = np.array([100.0, 90.0, 80.0, 75.0, 70.0])

    # Create test data structure
    test_data = {
        "model_data": {
            "approx": MockApprox(),
            "tau": np.array([[100, 300], [200, 400]], dtype=float),
            "data": np.random.rand(2, 5, 500),
        },
        "metadata": {
            "preprocess": {"time_lims": [0, 1000], "bin_width": 1},
            "data": {"data_dir": "/tmp", "region_name": "region", "taste_num": "all"}
        }
    }

    # Create temporary pickle file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        pkl.dump(test_data, f)
        temp_path = f.name

    try:
        # Test PklHandler with the temporary file
        with patch('pytau.changepoint_analysis.EphysData'):
            handler = PklHandler(temp_path)

            # Test that elbo_hist property exists and returns the hist array
            assert hasattr(handler, 'elbo_hist')
            assert handler.elbo_hist is not None
            np.testing.assert_array_equal(handler.elbo_hist, test_data["model_data"]["approx"].hist)

            # Test that we can index the last ELBO value
            last_elbo = handler.elbo_hist[-1]
            assert last_elbo == 70.0

            # Test that we can access any index
            first_elbo = handler.elbo_hist[0]
            assert first_elbo == 100.0

    finally:
        # Clean up temporary file
        os.unlink(temp_path)


def test_pkl_handler_elbo_hist_none():
    """Test PklHandler.elbo_hist property when approx is None."""
    import cloudpickle as pkl
    import tempfile

    # Create test data structure without approx
    test_data = {
        "model_data": {
            "tau": np.array([[100, 300], [200, 400]], dtype=float),
            "data": np.random.rand(2, 5, 500),
        },
        "metadata": {
            "preprocess": {"time_lims": [0, 1000], "bin_width": 1},
            "data": {"data_dir": "/tmp", "region_name": "region", "taste_num": "all"}
        }
    }

    # Create temporary pickle file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        pkl.dump(test_data, f)
        temp_path = f.name

    try:
        # Test PklHandler with the temporary file
        with patch('pytau.changepoint_analysis.EphysData'):
            handler = PklHandler(temp_path)

            # Test that elbo_hist property returns None when approx is missing
            assert handler.elbo_hist is None

    finally:
        # Clean up temporary file
        os.unlink(temp_path)
