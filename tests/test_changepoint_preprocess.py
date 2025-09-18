import numpy as np
import pytest

from pytau.changepoint_preprocess import preprocess_all_taste, preprocess_single_taste


class TestPreprocessSingleTaste:
    def setup_method(self):
        # Setup code here
        self.spike_array = np.random.randint(0, 2, (10, 5, 1000))
        self.time_lims = [200, 800]
        self.bin_width = 50

    @pytest.mark.parametrize("data_transform", [
        "trial_shuffled", "spike_shuffled", "simulated", None, "None"
    ])
    def test_preprocess_single_taste(self, data_transform):
        result = preprocess_single_taste(
            self.spike_array, self.time_lims, self.bin_width, data_transform)
        # Check if the shape is as expected after binning
        assert result.shape == (10, 5, 12)


class TestPreprocessAllTaste:
    def setup_method(self):
        # Setup code here
        self.spike_array = np.random.randint(0, 2, (4, 10, 5, 1000))
        self.time_lims = [200, 800]
        self.bin_width = 50

    @pytest.mark.parametrize("data_transform", [
        "trial_shuffled", "spike_shuffled", "simulated", None, "None"
    ])
    def test_preprocess_all_taste(self, data_transform):
        result = preprocess_all_taste(
            self.spike_array, self.time_lims, self.bin_width, data_transform)
        # Check if the shape is as expected after binning
        assert result.shape == (4, 10, 5, 12)
