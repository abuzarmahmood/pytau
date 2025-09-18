import pytest
import numpy as np
from pytau.changepoint_preprocess import preprocess_single_taste, preprocess_all_taste

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
        result = preprocess_single_taste(self.spike_array, self.time_lims, self.bin_width, data_transform)
        assert result.shape == (10, 5, 12)  # Check if the shape is as expected after binning

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
        result = preprocess_all_taste(self.spike_array, self.time_lims, self.bin_width, data_transform)
        assert result.shape == (4, 10, 5, 12)  # Check if the shape is as expected after binning
