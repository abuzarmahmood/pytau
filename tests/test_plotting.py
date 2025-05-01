"""
Tests for the plotting module.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pytau.utils.plotting import (
    plot_aligned_state_firing,
    plot_changepoint_overview,
    plot_changepoint_raster,
    plot_elbo_history,
    plot_state_firing_rates,
    raster,
)


@pytest.fixture
def mock_spike_array():
    """Create a mock spike array for testing."""
    return np.random.randint(0, 2, size=(5, 10, 100))


@pytest.fixture
def mock_tau():
    """Create a mock tau array for testing."""
    # Create a mock tau array with random state assignments
    tau = np.zeros((100,))
    # Assign random states (0, 1, 2) to different time segments
    tau[0:30] = 0
    tau[30:60] = 1
    tau[60:] = 2
    return tau


def test_raster_basic():
    """Test basic functionality of the raster plot function."""
    # Create test data
    spikes = np.random.randint(0, 2, size=(5, 100))

    # Test with default parameters
    fig, ax = plt.subplots()
    result = raster(spikes, ax=ax)

    # Check that the function returns the axis
    assert isinstance(result, Axes)

    # Test with no axis provided
    result = raster(spikes)
    assert isinstance(result, Axes)

    # Clean up
    plt.close("all")


def test_raster_appearance():
    """Test appearance customization of the raster plot."""
    # Create test data
    spikes = np.random.randint(0, 2, size=(5, 100))

    # Test with custom color and alpha
    fig, ax = plt.subplots()
    result = raster(spikes, ax=ax, color="red", alpha=0.8)

    # Check that the function returns the axis
    assert isinstance(result, Axes)

    # Clean up
    plt.close("all")


def test_plot_changepoint_raster(mock_spike_array, mock_tau):
    """Test the plot_changepoint_raster function."""
    # Test with default parameters
    fig = plot_changepoint_raster(mock_spike_array[0], mock_tau)

    # Check that the function returns a figure
    assert isinstance(fig, Figure)

    # Test with custom plot limits
    fig = plot_changepoint_raster(
        mock_spike_array[0], mock_tau, plot_lims=[10, 90])
    assert isinstance(fig, Figure)

    # Clean up
    plt.close("all")


def test_plot_changepoint_overview(mock_tau):
    """Test the plot_changepoint_overview function."""
    # Test with required parameters
    fig = plot_changepoint_overview(mock_tau, plot_lims=[0, 100])

    # Check that the function returns a figure
    assert isinstance(fig, Figure)

    # Clean up
    plt.close("all")


@pytest.mark.parametrize("window_radius", [100, 300])
def test_plot_aligned_state_firing(mock_spike_array, mock_tau, window_radius):
    """Test the plot_aligned_state_firing function with different window radii."""
    # Test with different window radii
    fig = plot_aligned_state_firing(
        mock_spike_array, mock_tau, window_radius=window_radius)

    # Check that the function returns a figure
    assert isinstance(fig, Figure)

    # Clean up
    plt.close("all")


def test_plot_state_firing_rates(mock_spike_array, mock_tau):
    """Test the plot_state_firing_rates function."""
    # Test with required parameters
    fig = plot_state_firing_rates(mock_spike_array, mock_tau)

    # Check that the function returns a figure
    assert isinstance(fig, Figure)

    # Clean up
    plt.close("all")


def test_plot_elbo_history():
    """Test the plot_elbo_history function."""

    # Create a mock fit model with ELBO history
    class MockFitModel:
        def __init__(self):
            self.hist = np.random.rand(100)

    mock_model = MockFitModel()

    # Test with default parameters
    fig = plot_elbo_history(mock_model)

    # Check that the function returns a figure
    assert isinstance(fig, Figure)

    # Test with custom final window
    fig = plot_elbo_history(mock_model, final_window=0.1)
    assert isinstance(fig, Figure)

    # Clean up
    plt.close("all")
