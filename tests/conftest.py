"""
Pytest configuration file for pytau tests.
"""

import numpy as np
import pytest


@pytest.fixture
def small_test_data_2d():
    """Return a small 2D test dataset."""
    return np.random.normal(size=(5, 50))


@pytest.fixture
def small_test_data_3d():
    """Return a small 3D test dataset."""
    return np.random.poisson(lam=1.0, size=(3, 5, 50))


@pytest.fixture
def small_test_data_4d():
    """Return a small 4D test dataset."""
    return np.random.poisson(lam=1.0, size=(2, 3, 5, 50))
