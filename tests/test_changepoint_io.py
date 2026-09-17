"""
Tests for the changepoint_io module.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from pytau.changepoint_io import DatabaseHandler, FitHandler

# Mock easygui and theano before importing pytau modules
sys.modules['easygui'] = Mock()
mock_theano = Mock()
mock_theano.__version__ = '1.0.0'
mock_theano.config = Mock()
mock_theano.config.compiledir = '/tmp/theano'
sys.modules['theano'] = mock_theano


class TestFitHandler(unittest.TestCase):
    """Test cases for FitHandler class."""

    def test_initialization_missing_experiment_name(self):
        """Test that FitHandler raises exception when experiment_name is missing."""
        with self.assertRaises(Exception) as context:
            FitHandler(data_dir='path/to/data',
                       taste_num=1, region_name='region')
        self.assertIn("experiment name", str(context.exception))

    def test_initialization_invalid_laser_type(self):
        """Test that FitHandler raises exception for invalid laser_type."""
        with self.assertRaises(Exception) as context:
            FitHandler(
                data_dir='path/to/data',
                taste_num=1,
                region_name='region',
                experiment_name='exp',
                laser_type='invalid'
            )
        self.assertIn("laser_type", str(context.exception))

    def test_initialization_invalid_taste_num(self):
        """Test that FitHandler raises exception for invalid taste_num."""
        with self.assertRaises(Exception) as context:
            FitHandler(
                data_dir='path/to/data',
                taste_num='invalid',
                region_name='region',
                experiment_name='exp'
            )
        self.assertIn("taste_num", str(context.exception))

    @patch('pytau.changepoint_io.EphysData')
    def test_initialization_success(self, mock_ephys_data):
        """Test successful FitHandler initialization."""
        mock_ephys_data.return_value = Mock()

        handler = FitHandler(
            data_dir='path/to/data',
            taste_num=1,
            region_name='region',
            experiment_name='exp'
        )

        self.assertEqual(handler.data_dir, 'path/to/data')
        self.assertEqual(handler.taste_num, 1)
        self.assertEqual(handler.region_name, 'region')
        self.assertEqual(handler.experiment_name, 'exp')
        mock_ephys_data.assert_called_once_with('path/to/data')

    @patch('pytau.changepoint_io.EphysData')
    def test_set_preprocess_params(self, mock_ephys_data):
        """Test setting preprocessing parameters."""
        mock_ephys_data.return_value = Mock()

        handler = FitHandler(
            data_dir='path/to/data',
            taste_num=1,
            region_name='region',
            experiment_name='exp'
        )

        handler.set_preprocess_params(
            time_lims=(0, 100),
            bin_width=10,
            data_transform='log'
        )

        self.assertEqual(handler.preprocess_params['bin_width'], 10)
        self.assertEqual(handler.preprocess_params['time_lims'], (0, 100))
        self.assertEqual(handler.preprocess_params['data_transform'], 'log')

    @patch('pytau.changepoint_io.EphysData')
    def test_load_spike_trains(self, mock_ephys_data):
        """Test loading spike trains."""
        mock_ephys_instance = Mock()
        test_data = np.random.poisson(1, (5, 10, 100))
        mock_ephys_instance.return_region_spikes.return_value = test_data
        mock_ephys_data.return_value = mock_ephys_instance

        handler = FitHandler(
            data_dir='path/to/data',
            taste_num=1,
            region_name='region',
            experiment_name='exp'
        )

        handler.load_spike_trains()
        self.assertIsNotNone(handler.data)
        mock_ephys_instance.return_region_spikes.assert_called_once()

    @patch('pytau.changepoint_io.EphysData')
    def test_preprocess_data(self, mock_ephys_data):
        """Test data preprocessing."""
        mock_ephys_data.return_value = Mock()

        handler = FitHandler(
            data_dir='path/to/data',
            taste_num=1,
            region_name='region',
            experiment_name='exp'
        )
        handler.data = np.random.poisson(1, (5, 10, 100))

        # Mock the preprocessor
        mock_preprocessor = Mock()
        mock_preprocessor.return_value = np.random.poisson(1, (5, 50))
        mock_preprocessor.__name__ = 'mock_preprocessor'
        handler.preprocessor = mock_preprocessor
        handler.preprocess_params = {'bin_width': 10}

        handler.preprocess_data()
        self.assertIsNotNone(handler.preprocessed_data)
        mock_preprocessor.assert_called_once()

    @patch('pytau.changepoint_io.EphysData')
    def test_create_model(self, mock_ephys_data):
        """Test model creation."""
        mock_ephys_data.return_value = Mock()

        handler = FitHandler(
            data_dir='path/to/data',
            taste_num=1,
            region_name='region',
            experiment_name='exp'
        )
        handler.preprocessed_data = np.random.poisson(1, (5, 50))

        # Mock the model template
        mock_model_template = Mock()
        mock_model_template.__name__ = 'MockModelTemplate'
        mock_model_instance = Mock()
        mock_model_template.return_value = mock_model_instance
        handler.model_template = mock_model_template
        handler.model_params = {
            'states': 3,
            'model_kwargs': {'param1': 'value1'}
        }

        handler.create_model()
        self.assertIsNotNone(handler.model)
        mock_model_template.assert_called_once()

    @patch('pytau.changepoint_io.EphysData')
    def test_run_inference(self, mock_ephys_data):
        """Test running inference."""
        mock_ephys_data.return_value = Mock()

        handler = FitHandler(
            data_dir='path/to/data',
            taste_num=1,
            region_name='region',
            experiment_name='exp'
        )

        # Mock the model and its fit method
        mock_model = Mock()
        mock_model.fit.return_value = {
            'model': 'fitted_model', 'trace': 'trace_data'}
        handler.model = mock_model

        # Mock required attributes
        handler.model_params = {'fit': 'advi', 'samples': 1000}
        mock_inference_func = Mock()
        mock_inference_func.return_value = {
            'model': 'fitted_model', 'trace': 'trace_data'}
        mock_inference_func.__name__ = 'advi_fit'
        handler.inference_func = mock_inference_func

        handler.run_inference()
        self.assertIn('model', handler.inference_outs)
        mock_inference_func.assert_called_once()

    @patch('pytau.changepoint_io.EphysData')
    def test_save_fit_output(self, mock_ephys_data):
        """Test saving fit output."""
        mock_ephys_data.return_value = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            handler = FitHandler(
                data_dir='path/to/data',
                taste_num=1,
                region_name='region',
                experiment_name='exp'
            )

            # Mock the database handler
            mock_db_handler = Mock()
            mock_db_handler.model_save_path = os.path.join(
                temp_dir, 'test_model')
            mock_db_handler.aggregate_metadata.return_value = {
                'test': 'metadata'}
            handler.database_handler = mock_db_handler

            # Mock inference outputs and required attributes
            handler.inference_outs = {
                'model': 'fitted_model', 'trace': 'trace_data'}
            handler.preprocess_params = {'bin_width': 10}
            handler.model_params = {
                'states': 3,
                'model_kwargs': {'param1': 'value1'}
            }
            handler.preprocessor = Mock()
            handler.preprocessor.__name__ = 'mock_preprocessor'
            handler.model_template = Mock()
            handler.model_template.__name__ = 'MockModel'
            handler.inference_func = Mock()
            handler.inference_func.__name__ = 'mock_inference_func'

            # Mock the save methods
            with patch('pytau.changepoint_io.pickle.dump'), \
                    patch('builtins.open', create=True):
                handler.save_fit_output()
                mock_db_handler.write_to_database.assert_called_once()

            # No tau/lambda/data/approx in inference_outs -> no sidecar written
            self.assertFalse(os.path.exists(
                mock_db_handler.model_save_path + '.npz'))

    @patch('pytau.changepoint_io.EphysData')
    def test_save_fit_output_writes_numpy_sidecar(self, mock_ephys_data):
        """save_fit_output should also write a lightweight .npz sidecar with
        just the numpy-serializable arrays (tau/lambda/data/elbo history)."""
        mock_ephys_data.return_value = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            handler = FitHandler(
                data_dir='path/to/data',
                taste_num=1,
                region_name='region',
                experiment_name='exp'
            )

            mock_db_handler = Mock()
            model_save_path = os.path.join(temp_dir, 'test_model')
            mock_db_handler.model_save_path = model_save_path
            mock_db_handler.aggregate_metadata.return_value = {
                'test': 'metadata'}
            handler.database_handler = mock_db_handler

            np.random.seed(0)
            tau_array = np.random.rand(5, 2)
            lambda_array = np.random.rand(5, 3, 10)
            data_array = np.random.poisson(1, (5, 10, 100))
            fake_approx = Mock()
            fake_approx.hist = np.array([100.0, 50.0, 10.0])

            handler.inference_outs = {
                'model': 'fitted_model', 'approx': fake_approx,
                'lambda': lambda_array, 'tau': tau_array, 'data': data_array,
            }
            handler.preprocess_params = {'bin_width': 10}
            handler.model_params = {
                'states': 3, 'model_kwargs': {'param1': 'value1'}}
            handler.preprocessor = Mock()
            handler.preprocessor.__name__ = 'mock_preprocessor'
            handler.model_template = Mock()
            handler.model_template.__name__ = 'MockModel'
            handler.inference_func = Mock()
            handler.inference_func.__name__ = 'mock_inference_func'

            with patch('pytau.changepoint_io.pickle.dump'), \
                    patch('builtins.open', create=True):
                handler.save_fit_output()

            npz_path = model_save_path + '.npz'
            self.assertTrue(os.path.exists(npz_path))
            with np.load(npz_path) as npz:
                np.testing.assert_array_equal(npz['tau_array'], tau_array)
                np.testing.assert_array_equal(
                    npz['lambda_array'], lambda_array)
                np.testing.assert_array_equal(
                    npz['processed_spikes'], data_array)
                np.testing.assert_array_equal(
                    npz['elbo_hist'], fake_approx.hist)

    @patch('pytau.changepoint_io.EphysData')
    def test_save_fit_output_sidecar_skips_missing_arrays(self, mock_ephys_data):
        """If only some keys (e.g. tau/data, no lambda, no approx) are
        present, the sidecar should only contain what's available."""
        mock_ephys_data.return_value = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            handler = FitHandler(
                data_dir='path/to/data',
                taste_num=1,
                region_name='region',
                experiment_name='exp'
            )

            mock_db_handler = Mock()
            model_save_path = os.path.join(temp_dir, 'test_model')
            mock_db_handler.model_save_path = model_save_path
            mock_db_handler.aggregate_metadata.return_value = {
                'test': 'metadata'}
            handler.database_handler = mock_db_handler

            np.random.seed(1)
            tau_array = np.random.rand(5, 2)
            data_array = np.random.poisson(1, (5, 10, 100))

            handler.inference_outs = {
                'model': 'fitted_model', 'tau': tau_array, 'data': data_array}
            handler.preprocess_params = {'bin_width': 10}
            handler.model_params = {'states': 3, 'model_kwargs': {}}
            handler.preprocessor = Mock()
            handler.preprocessor.__name__ = 'mock_preprocessor'
            handler.model_template = Mock()
            handler.model_template.__name__ = 'MockModel'
            handler.inference_func = Mock()
            handler.inference_func.__name__ = 'mock_inference_func'

            with patch('pytau.changepoint_io.pickle.dump'), \
                    patch('builtins.open', create=True):
                handler.save_fit_output()

            npz_path = model_save_path + '.npz'
            self.assertTrue(os.path.exists(npz_path))
            with np.load(npz_path) as npz:
                self.assertEqual(
                    set(npz.files), {'tau_array', 'processed_spikes'})

    @patch('pytau.changepoint_io.EphysData')
    def test_save_fit_output_sidecar_much_smaller_than_pickle(self, mock_ephys_data):
        """The .npz sidecar should be a small fraction of the size of a
        comparably-shaped full pickle (which also carries model/approx)."""
        mock_ephys_data.return_value = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            handler = FitHandler(
                data_dir='path/to/data',
                taste_num=1,
                region_name='region',
                experiment_name='exp'
            )

            mock_db_handler = Mock()
            model_save_path = os.path.join(temp_dir, 'test_model')
            mock_db_handler.model_save_path = model_save_path
            mock_db_handler.aggregate_metadata.return_value = {
                'test': 'metadata'}
            handler.database_handler = mock_db_handler

            np.random.seed(2)
            tau_array = np.random.rand(2000, 3)
            lambda_array = np.random.rand(4, 2000, 30)
            data_array = np.random.poisson(1, (30, 30, 40))
            # Stand-in for a large, opaque pymc model/approx blob.
            dummy_model_blob = np.random.rand(2_000_000)

            handler.inference_outs = {
                'model': dummy_model_blob, 'approx': dummy_model_blob,
                'lambda': lambda_array, 'tau': tau_array, 'data': data_array,
            }
            handler.preprocess_params = {'bin_width': 10}
            handler.model_params = {'states': 3, 'model_kwargs': {}}
            handler.preprocessor = Mock()
            handler.preprocessor.__name__ = 'mock_preprocessor'
            handler.model_template = Mock()
            handler.model_template.__name__ = 'MockModel'
            handler.inference_func = Mock()
            handler.inference_func.__name__ = 'mock_inference_func'

            # Let the real pkl write happen this time (no mocking of
            # pickle.dump/open), so we get a real file size to compare.
            handler.save_fit_output()

            pkl_path = model_save_path + '.pkl'
            npz_path = model_save_path + '.npz'
            self.assertTrue(os.path.exists(pkl_path))
            self.assertTrue(os.path.exists(npz_path))
            pkl_size = os.path.getsize(pkl_path)
            npz_size = os.path.getsize(npz_path)
            self.assertLess(npz_size, pkl_size / 5)


class TestDatabaseHandler(unittest.TestCase):
    """Test cases for DatabaseHandler class."""

    @patch('os.path.exists')
    def test_initialization_no_database(self, mock_exists):
        """Test DatabaseHandler initialization when database doesn't exist."""
        mock_exists.return_value = False

        handler = DatabaseHandler()
        self.assertIsNotNone(handler.model_database_path)
        self.assertTrue(handler.model_database_path.endswith('.csv'))
        self.assertFalse(hasattr(handler, 'fit_database'))

    @patch('os.path.exists')
    @patch('pytau.changepoint_io.pd.read_csv')
    def test_initialization_with_database(self, mock_read_csv, mock_exists):
        """Test DatabaseHandler initialization when database exists."""
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({'col1': [1, 2, 3]})

        handler = DatabaseHandler()
        self.assertIsNotNone(handler.model_database_path)
        self.assertTrue(hasattr(handler, 'fit_database'))

    @patch('os.path.exists')
    @patch('pytau.changepoint_io.pd.read_csv')
    def test_show_duplicates_with_database(self, mock_read_csv, mock_exists):
        """Test show_duplicates with existing database."""
        mock_exists.return_value = True
        sample_data = pd.DataFrame({
            'experiment_name': ['exp1', 'exp1', 'exp2'],
            'taste_num': [1, 1, 2],
            'region_name': ['region1', 'region1', 'region2'],
            'exp.model_id': ['id1', 'id2', 'id3'],
            'exp.save_path': ['path1', 'path2', 'path3'],
            'exp.fit_date': ['date1', 'date2', 'date3']
        })
        mock_read_csv.return_value = sample_data

        handler = DatabaseHandler()
        duplicates, dup_inds = handler.show_duplicates()

        self.assertIsInstance(duplicates, pd.DataFrame)
        self.assertIsInstance(dup_inds, pd.Series)

    @patch('os.path.exists')
    @patch('pytau.changepoint_io.pd.read_csv')
    def test_check_mismatched_paths_with_database(self, mock_read_csv, mock_exists):
        """Test check_mismatched_paths with existing database."""
        mock_exists.return_value = True
        sample_data = pd.DataFrame({
            'exp.save_path': ['/path/to/model1', '/path/to/model2']
        })
        mock_read_csv.return_value = sample_data

        handler = DatabaseHandler()
        result = handler.check_mismatched_paths()

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)  # Should return 3 items

    @patch('os.path.exists')
    def test_ingest_fit_data(self, mock_exists):
        """Test ingesting fit data."""
        mock_exists.return_value = False

        handler = DatabaseHandler()
        test_data = {'key1': 'value1', 'key2': 'value2'}

        handler.ingest_fit_data(test_data)

        self.assertEqual(handler.external_metadata, test_data)

    @patch('os.path.exists')
    def test_aggregate_metadata_no_data(self, mock_exists):
        """Test aggregate_metadata raises exception when no data ingested."""
        mock_exists.return_value = False

        handler = DatabaseHandler()

        with self.assertRaises(Exception):
            handler.aggregate_metadata()

    @patch('os.path.exists')
    def test_aggregate_metadata_with_data(self, mock_exists):
        """Test aggregate_metadata with ingested data."""
        mock_exists.return_value = False

        handler = DatabaseHandler()
        handler.fit_data = {
            'experiment_name': 'test_exp',
            'taste_num': 1,
            'region_name': 'test_region'
        }

        # This should not raise an exception
        try:
            handler.aggregate_metadata()
        except Exception as e:
            # If it raises an exception, it should be a specific one we expect
            # (like missing required fields), not a generic "no data" exception
            self.assertNotIn("no data", str(e).lower())

    @patch('os.path.exists')
    @patch('pytau.changepoint_io.pd.read_csv')
    @patch('pytau.changepoint_io.pd.DataFrame.to_csv')
    def test_write_to_database(self, mock_to_csv, mock_read_csv, mock_exists):
        """Test writing to database."""
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame()

        handler = DatabaseHandler()
        handler.external_metadata = {
            'preprocess': {'bin_width': 10, 'preprocessor_name': 'test_preprocessor'},
            'model': {'states': 3, 'model_kwargs': {'param1': 'value1'}, 'model_template_name': 'TestModel', 'inference_func_name': 'test_inference'}
        }

        # Mock the required attributes for aggregate_metadata
        handler.data_dir = '/path/to/data'
        handler.data_basename = 'test_data'
        handler.animal_name = 'test_animal'
        handler.session_date = '2023-01-01'
        handler.taste_num = 1
        handler.laser_type = None
        handler.region_name = 'test_region'
        handler.model_save_path = '/path/to/model'
        handler.experiment_name = 'test_exp'
        handler.model_id = 'test_model_id'
        handler.fit_date = '2023-01-01'

        handler.write_to_database()

        mock_to_csv.assert_called_once()


if __name__ == '__main__':
    unittest.main()
