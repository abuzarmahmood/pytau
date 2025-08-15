import os
import unittest

import pandas as pd

from pytau.changepoint_io import DatabaseHandler, FitHandler


class TestFitHandler(unittest.TestCase):
    def setUp(self):
        self.handler = FitHandler(
            data_dir='path/to/data',
            taste_num=1,
            region_name='region',
            experiment_name='exp'
        )

    def test_initialization(self):
        with self.assertRaises(Exception):
            FitHandler(data_dir='path/to/data',
                       taste_num=1, region_name='region')

    def test_set_preprocess_params(self):
        self.handler.set_preprocess_params(time_lims=(
            0, 100), bin_width=10, data_transform='log')
        self.assertEqual(self.handler.preprocess_params['bin_width'], 10)

    def test_load_spike_trains(self):
        self.handler.load_spike_trains()
        self.assertIsNotNone(self.handler.data)

    def test_preprocess_data(self):
        self.handler.preprocess_data()
        self.assertIsNotNone(self.handler.preprocessed_data)

    def test_create_model(self):
        self.handler.create_model()
        self.assertIsNotNone(self.handler.model)

    def test_run_inference(self):
        self.handler.run_inference()
        self.assertIn('model', self.handler.inference_outs)

    def test_save_fit_output(self):
        self.handler.save_fit_output()
        # Check if the output files are created
        self.assertTrue(os.path.exists(
            self.handler.database_handler.model_save_path + ".pkl"))


class TestDatabaseHandler(unittest.TestCase):
    def setUp(self):
        self.handler = DatabaseHandler()

    def test_initialization(self):
        self.assertIsNotNone(self.handler.model_database_path)

    def test_show_duplicates(self):
        duplicates, _ = self.handler.show_duplicates()
        self.assertIsInstance(duplicates, pd.DataFrame)

    def test_check_mismatched_paths(self):
        mismatches = self.handler.check_mismatched_paths()
        self.assertIsInstance(mismatches, tuple)

    def test_aggregate_metadata(self):
        with self.assertRaises(Exception):
            self.handler.aggregate_metadata()

    def test_write_to_database(self):
        # Assuming some metadata is ingested
        self.handler.ingest_fit_data({'dummy_key': 'dummy_value'})
        self.handler.write_to_database()
        self.assertTrue(os.path.isfile(self.handler.model_database_path))


if __name__ == '__main__':
    unittest.main()
