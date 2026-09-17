import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from pytau.changepoint_analysis import (
    PklHandler,
    _firing,
    _tau,
    calc_collapsed_transitions,
    calc_dynamic_neurons,
    calc_firing_drift_anova,
    calc_significant_neurons_firing,
    calc_significant_neurons_snippets,
    calc_state_trial_uniformity,
    calc_transition_randomness,
    get_fit_diagnostics,
    get_state_firing,
    get_time_binned_firing,
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


##############################
# Data-quality diagnostics tests
##############################

def test_calc_firing_drift_anova_detects_drift():
    """Most neurons ramping up across trial blocks should be flagged."""
    np.random.seed(0)
    n_bins, n_neurons = 20, 10
    block_size, n_blocks = 15, 4
    n_trials = block_size * n_blocks
    spike_array = np.zeros((n_trials, n_neurons, n_bins))
    for block_i in range(n_blocks):
        sl = slice(block_i * block_size, (block_i + 1) * block_size)
        for neuron in range(n_neurons):
            rate = 1.0 + (4.0 * block_i if neuron < 8 else 0.0)
            spike_array[sl, neuron, :] = np.random.poisson(
                rate, size=(block_size, n_bins))

    neuron_p, drift_neurons, pop_p, is_drift = calc_firing_drift_anova(
        spike_array, n_trial_bins=n_blocks)
    assert is_drift is True
    assert pop_p < 0.05
    assert len(drift_neurons) >= 6


def test_calc_firing_drift_anova_no_drift_on_iid_noise():
    """i.i.d. noise with no trial-order structure should not be flagged."""
    np.random.seed(1)
    spike_array = np.random.poisson(2.0, size=(40, 10, 20)).astype(float)
    _, _, _, is_drift = calc_firing_drift_anova(spike_array, n_trial_bins=4)
    assert is_drift is False


def test_calc_firing_drift_anova_too_few_trials():
    """Too few trials per block should yield nan, not a crash or false flag."""
    np.random.seed(2)
    spike_array = np.random.poisson(1.0, size=(2, 5, 10)).astype(float)
    neuron_p, drift_neurons, pop_p, is_drift = calc_firing_drift_anova(
        spike_array, n_trial_bins=4)
    assert np.all(np.isnan(neuron_p))
    assert np.isnan(pop_p)
    assert is_drift is False
    assert len(drift_neurons) == 0


def test_calc_state_trial_uniformity_detects_confined_state():
    """A state present only in a specific trial-order block should be flagged."""
    n_bins, n_trials = 100, 40
    tau = np.zeros((n_trials, 1))
    tau[:10, 0] = 2       # state 0 absent (duration 2 < 5% of 100 = 5)
    tau[10:, 0] = 60      # state 0 present (duration 60)
    _, nonuniform_states = calc_state_trial_uniformity(tau, n_bins, n_trial_blocks=4)
    assert 0 in nonuniform_states


def test_calc_state_trial_uniformity_no_flag_when_uniform():
    """A state present in a random, non-trial-order-dependent way should
    not be flagged."""
    np.random.seed(3)
    n_bins, n_trials = 100, 40
    tau = np.random.uniform(30, 70, size=(n_trials, 1))
    _, nonuniform_states = calc_state_trial_uniformity(tau, n_bins, n_trial_blocks=4)
    assert len(nonuniform_states) == 0


def test_calc_state_trial_uniformity_state_present_everywhere_not_flagged():
    """A state present in every single trial has nothing to test against
    (deferred to calc_collapsed_transitions) and should return nan, not a
    flag."""
    n_bins, n_trials = 100, 40
    tau = np.full((n_trials, 1), 50.0)
    state_p, nonuniform_states = calc_state_trial_uniformity(tau, n_bins)
    assert np.isnan(state_p[0])
    assert 0 not in nonuniform_states


def test_calc_collapsed_transitions_detects_edge_collapse():
    n_bins = 1000
    tau = np.tile(np.array([2.0, 500.0]), (20, 1))
    edge_mask, _, warnings_list = calc_collapsed_transitions(tau, n_bins)
    assert edge_mask[:, 0].all()
    assert len(warnings_list) == 1
    assert "edge" in warnings_list[0]


def test_calc_collapsed_transitions_detects_merge():
    n_bins = 1000
    tau = np.tile(np.array([500.0, 502.0]), (20, 1))
    _, merged_mask, warnings_list = calc_collapsed_transitions(tau, n_bins)
    assert merged_mask.all()
    assert any("merged" in w for w in warnings_list)


def test_calc_collapsed_transitions_clean_no_warning():
    tau_array = np.array(
        [[100, 300], [200, 400], [150, 350], [250, 450], [300, 500]], dtype=float)
    n_bins = 1000
    _, _, warnings_list = calc_collapsed_transitions(tau_array, n_bins)
    assert warnings_list == []


def test_calc_collapsed_transitions_single_changepoint_no_crash():
    n_bins = 1000
    tau = np.tile(np.array([500.0]), (10, 1))
    _, merged_mask, _ = calc_collapsed_transitions(tau, n_bins)
    assert merged_mask.shape == (10, 0)


def test_calc_transition_randomness_flags_uniform_draw():
    np.random.seed(4)
    n_bins, n_trials = 1000, 300
    tau_uniform = np.random.uniform(0, n_bins, size=(n_trials, 1))
    stat_array, random_transitions = calc_transition_randomness(
        tau_uniform, n_bins, method="chisquare")
    assert 0 in random_transitions
    assert stat_array[0] > 0.05


def test_calc_transition_randomness_entropy_method_flags_uniform_draw():
    np.random.seed(5)
    n_bins, n_trials = 1000, 300
    tau_uniform = np.random.uniform(0, n_bins, size=(n_trials, 1))
    _, random_transitions = calc_transition_randomness(
        tau_uniform, n_bins, method="entropy")
    assert 0 in random_transitions


def test_calc_transition_randomness_no_flag_for_consistent_transition():
    """A tight, consistent transition should NOT be flagged as random --
    and its chi-squared p-value should be small (i.e. the opposite
    direction of the 'flagged' case above)."""
    np.random.seed(6)
    n_bins, n_trials = 1000, 300
    tau_consistent = np.clip(
        np.random.normal(500, 20, size=(n_trials, 1)), 0, n_bins)
    stat_array, random_transitions = calc_transition_randomness(
        tau_consistent, n_bins, method="chisquare")
    assert len(random_transitions) == 0
    assert stat_array[0] < 0.05


def test_calc_transition_randomness_too_few_trials_returns_nan():
    n_bins = 1000
    tau = np.array([[100.0], [200.0], [300.0]])
    stat_array, random_transitions = calc_transition_randomness(
        tau, n_bins, method="chisquare")
    assert np.isnan(stat_array[0])
    assert len(random_transitions) == 0


def test_calc_transition_randomness_invalid_method_raises():
    tau = np.array([[100.0], [200.0]])
    with pytest.raises(ValueError):
        calc_transition_randomness(tau, 1000, method="bogus")


def test_get_time_binned_firing_shape():
    np.random.seed(7)
    spike_array = np.random.rand(5, 10, 1000)
    result = get_time_binned_firing(spike_array, n_bins=10)
    assert result.shape == (5, 10, 10)


def test_get_time_binned_firing_raises_on_non_divisible_bins():
    spike_array = np.random.rand(5, 10, 1000)
    with pytest.raises(ValueError):
        get_time_binned_firing(spike_array, n_bins=7)


def test_calc_dynamic_neurons_selects_ramping_neurons():
    np.random.seed(8)
    n_trials, n_neurons, n_time = 20, 10, 100
    spike_array = np.zeros((n_trials, n_neurons, n_time))
    time_axis = np.arange(n_time)
    for neuron in range(n_neurons):
        if neuron < 5:
            rate = 0.2 + 4.8 * (time_axis / n_time)
        else:
            rate = np.full(n_time, 1.0)
        spike_array[:, neuron, :] = np.random.poisson(
            rate, size=(n_trials, n_time))

    _, _, dynamic_neurons = calc_dynamic_neurons(spike_array, n_bins=10)
    for neuron in range(5):
        assert neuron in dynamic_neurons


def test_calc_dynamic_neurons_flat_neurons_mostly_not_selected():
    np.random.seed(9)
    spike_array = np.random.poisson(1.0, size=(20, 10, 100)).astype(float)
    _, _, dynamic_neurons = calc_dynamic_neurons(
        spike_array, n_bins=10, dynamic_p_val=0.01)
    assert len(dynamic_neurons) <= 2


def test_calc_dynamic_neurons_does_not_break_calc_significant_neurons_firing():
    """Regression: calc_significant_neurons_firing must remain a 2-tuple
    return, unaffected by calc_dynamic_neurons reusing it as its engine."""
    np.random.seed(10)
    state_firing = np.random.rand(5, 3, 10)
    result = calc_significant_neurons_firing(state_firing, 0.05)
    assert len(result) == 2


def test_get_fit_diagnostics_flags_only_expected_check():
    n_trials, n_neurons, n_bins = 20, 5, 1000
    np.random.seed(11)
    spike_array = np.random.poisson(
        1.0, size=(n_trials, n_neurons, n_bins)).astype(float)
    # Only trip the edge-collapse check; keep everything else well-behaved.
    tau_array = np.tile(np.array([5.0, 500.0]), (n_trials, 1))
    warnings_list, diagnostics = get_fit_diagnostics(
        spike_array, tau_array, n_bins)
    assert len(warnings_list) == 1
    assert "edge" in warnings_list[0]
    assert set(diagnostics.keys()) == {
        "drift_anova", "state_trial_uniformity",
        "collapsed_transitions", "transition_randomness",
    }


def test_pkl_handler_get_diagnostics_is_opt_in():
    """get_diagnostics should not run automatically; calling it explicitly
    should populate .warnings/.diagnostics."""
    np.random.seed(13)
    n_trials, n_neurons, n_bins = 20, 5, 1000
    spike_array = np.random.poisson(
        1.0, size=(n_trials, n_neurons, n_bins)).astype(float)
    tau_array = np.tile(np.array([300.0, 700.0]), (n_trials, 1))

    handler = PklHandler.__new__(PklHandler)
    handler.processed_spikes = spike_array
    handler.tau = _tau(
        tau_array, {"preprocess": {"time_lims": [0, n_bins], "bin_width": 1}}, n_trials)

    assert not hasattr(handler, "warnings")
    warnings_list = handler.get_diagnostics()
    assert warnings_list == handler.warnings
    assert isinstance(handler.diagnostics, dict)
