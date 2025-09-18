import numpy as np
import pytest

from pytau.changepoint_preprocess import preprocess_all_taste, preprocess_single_taste


class TestPreprocessSingleTaste:
    """Test suite for preprocess_single_taste function."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducible tests
        self.spike_array = np.random.randint(0, 2, (10, 5, 100))
        self.time_lims = [0, 100]
        self.bin_width = 10

    def test_shape_output(self):
        """Test that output shape is correct for all transforms."""
        transforms = ['trial_shuffled',
                      'spike_shuffled', 'simulated', None, 'None']
        expected_shape = (10, 5, 10)

        for transform in transforms:
            result = preprocess_single_taste(
                self.spike_array, self.time_lims, self.bin_width, transform)
            assert result.shape == expected_shape, f"Failed for transform: {transform}"

    def test_none_transform_preserves_data(self):
        """Test that None transform preserves original data after binning."""
        result_none = preprocess_single_taste(
            self.spike_array, self.time_lims, self.bin_width, None)
        result_none_str = preprocess_single_taste(
            self.spike_array, self.time_lims, self.bin_width, 'None')

        # Both None and 'None' should give same result
        np.testing.assert_array_equal(result_none, result_none_str)

        # Check that binning is correct
        expected = np.sum(
            self.spike_array[..., self.time_lims[0]:self.time_lims[1]].reshape(
                *self.spike_array.shape[:-1], -1, self.bin_width
            ), axis=-1
        )
        np.testing.assert_array_equal(result_none, expected)

    def test_trial_shuffled_different_from_original(self):
        """Test that trial shuffled data is different from original."""
        np.random.seed(42)
        original = preprocess_single_taste(
            self.spike_array, self.time_lims, self.bin_width, None)

        np.random.seed(123)  # Different seed
        shuffled = preprocess_single_taste(
            self.spike_array, self.time_lims, self.bin_width, 'trial_shuffled')

        # Should be different (with high probability)
        assert not np.array_equal(original, shuffled)

    def test_spike_shuffled_different_from_original(self):
        """Test that spike shuffled data is different from original."""
        np.random.seed(42)
        original = preprocess_single_taste(
            self.spike_array, self.time_lims, self.bin_width, None)

        np.random.seed(123)  # Different seed
        shuffled = preprocess_single_taste(
            self.spike_array, self.time_lims, self.bin_width, 'spike_shuffled')

        # Should be different (with high probability)
        assert not np.array_equal(original, shuffled)

    def test_simulated_data_properties(self):
        """Test that simulated data has reasonable properties."""
        # Use data with clear firing patterns
        spike_array = np.zeros((5, 3, 100))
        spike_array[:, 0, 10:20] = 1  # High firing in specific window

        result = preprocess_single_taste(
            spike_array, self.time_lims, self.bin_width, 'simulated')

        # Should be non-negative integers (binned spike counts)
        assert np.all(result >= 0)
        assert np.issubdtype(result.dtype, np.integer)

    def test_invalid_transform_raises_exception(self):
        """Test that invalid transform raises exception."""
        with pytest.raises(Exception, match="data_transform must be of type"):
            preprocess_single_taste(
                self.spike_array, self.time_lims, self.bin_width, 'invalid_transform')

    def test_different_time_limits(self):
        """Test with different time limits."""
        time_lims = [10, 90]
        result = preprocess_single_taste(
            self.spike_array, time_lims, self.bin_width, None)

        expected_shape = (10, 5, 8)  # (90-10)/10 = 8 bins
        assert result.shape == expected_shape

    def test_different_bin_widths(self):
        """Test with different bin widths."""
        bin_widths = [5, 20, 25]
        expected_bins = [20, 5, 4]  # 100/bin_width

        for bin_width, expected_bin_count in zip(bin_widths, expected_bins):
            result = preprocess_single_taste(
                self.spike_array, self.time_lims, bin_width, None)
            assert result.shape == (10, 5, expected_bin_count)

    def test_output_dtype_is_integer(self):
        """Test that output is integer type."""
        result = preprocess_single_taste(
            self.spike_array, self.time_lims, self.bin_width, None)
        assert np.issubdtype(result.dtype, np.integer)


class TestPreprocessAllTaste:
    """Test suite for preprocess_all_taste function."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducible tests
        self.spike_array = np.random.randint(0, 2, (3, 10, 5, 100))
        self.time_lims = [0, 100]
        self.bin_width = 10

    def test_shape_output(self):
        """Test that output shape is correct for all transforms."""
        transforms = ['trial_shuffled',
                      'spike_shuffled', 'simulated', None, 'None']
        expected_shape = (3, 10, 5, 10)

        for transform in transforms:
            result = preprocess_all_taste(
                self.spike_array, self.time_lims, self.bin_width, transform)
            assert result.shape == expected_shape, f"Failed for transform: {transform}"

    def test_none_transform_preserves_data(self):
        """Test that None transform preserves original data after binning."""
        result_none = preprocess_all_taste(
            self.spike_array, self.time_lims, self.bin_width, None)
        result_none_str = preprocess_all_taste(
            self.spike_array, self.time_lims, self.bin_width, 'None')

        # Both None and 'None' should give same result
        np.testing.assert_array_equal(result_none, result_none_str)

        # Check that binning is correct
        expected = np.sum(
            self.spike_array[..., self.time_lims[0]:self.time_lims[1]].reshape(
                *self.spike_array.shape[:-1], -1, self.bin_width
            ), axis=-1
        )
        np.testing.assert_array_equal(result_none, expected)

    def test_trial_shuffled_different_from_original(self):
        """Test that trial shuffled data is different from original."""
        np.random.seed(42)
        original = preprocess_all_taste(
            self.spike_array, self.time_lims, self.bin_width, None)

        np.random.seed(123)  # Different seed
        shuffled = preprocess_all_taste(
            self.spike_array, self.time_lims, self.bin_width, 'trial_shuffled')

        # Should be different (with high probability)
        assert not np.array_equal(original, shuffled)

    def test_spike_shuffled_different_from_original(self):
        """Test that spike shuffled data is different from original."""
        np.random.seed(42)
        original = preprocess_all_taste(
            self.spike_array, self.time_lims, self.bin_width, None)

        np.random.seed(123)  # Different seed
        shuffled = preprocess_all_taste(
            self.spike_array, self.time_lims, self.bin_width, 'spike_shuffled')

        # Should be different (with high probability)
        assert not np.array_equal(original, shuffled)

    def test_simulated_data_properties(self):
        """Test that simulated data has reasonable properties."""
        # Use data with clear firing patterns
        spike_array = np.zeros((2, 5, 3, 100))
        spike_array[:, :, 0, 10:20] = 1  # High firing in specific window

        result = preprocess_all_taste(
            spike_array, self.time_lims, self.bin_width, 'simulated')

        # Should be non-negative integers (binned spike counts)
        assert np.all(result >= 0)
        assert np.issubdtype(result.dtype, np.integer)

    def test_invalid_transform_raises_exception(self):
        """Test that invalid transform raises exception."""
        with pytest.raises(Exception, match="data_transform must be of type"):
            preprocess_all_taste(
                self.spike_array, self.time_lims, self.bin_width, 'invalid_transform')

    def test_different_time_limits(self):
        """Test with different time limits."""
        time_lims = [10, 90]
        result = preprocess_all_taste(
            self.spike_array, time_lims, self.bin_width, None)

        expected_shape = (3, 10, 5, 8)  # (90-10)/10 = 8 bins
        assert result.shape == expected_shape

    def test_different_bin_widths(self):
        """Test with different bin widths."""
        bin_widths = [5, 20, 25]
        expected_bins = [20, 5, 4]  # 100/bin_width

        for bin_width, expected_bin_count in zip(bin_widths, expected_bins):
            result = preprocess_all_taste(
                self.spike_array, self.time_lims, bin_width, None)
            assert result.shape == (3, 10, 5, expected_bin_count)

    def test_output_dtype_is_integer(self):
        """Test that output is integer type."""
        result = preprocess_all_taste(
            self.spike_array, self.time_lims, self.bin_width, None)
        assert np.issubdtype(result.dtype, np.integer)

    def test_consistency_across_tastes(self):
        """Test that each taste is processed consistently."""
        # Create identical data for all tastes
        single_taste_data = np.random.randint(0, 2, (10, 5, 100))
        all_taste_data = np.stack([single_taste_data] * 3, axis=0)

        # Process with all_taste function
        result_all = preprocess_all_taste(
            all_taste_data, self.time_lims, self.bin_width, None)

        # Process each taste individually
        result_single = preprocess_single_taste(
            single_taste_data, self.time_lims, self.bin_width, None)

        # Each taste should match the single taste result
        for taste_idx in range(3):
            np.testing.assert_array_equal(result_all[taste_idx], result_single)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_array_handling(self):
        """Test behavior with minimal arrays."""
        # Minimal valid array
        spike_array = np.zeros((1, 1, 10))
        time_lims = [0, 10]
        bin_width = 5

        result = preprocess_single_taste(
            spike_array, time_lims, bin_width, None)
        assert result.shape == (1, 1, 2)

    def test_single_bin_case(self):
        """Test when time window equals bin width."""
        spike_array = np.random.randint(0, 2, (5, 3, 20))
        time_lims = [0, 10]
        bin_width = 10

        result = preprocess_single_taste(
            spike_array, time_lims, bin_width, None)
        assert result.shape == (5, 3, 1)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        spike_array = np.random.randint(0, 2, (5, 3, 50))
        time_lims = [0, 50]
        bin_width = 10

        # Run twice with same seed
        np.random.seed(42)
        result1 = preprocess_single_taste(
            spike_array, time_lims, bin_width, 'trial_shuffled')

        np.random.seed(42)
        result2 = preprocess_single_taste(
            spike_array, time_lims, bin_width, 'trial_shuffled')

        np.testing.assert_array_equal(result1, result2)
