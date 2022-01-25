"""
Pipeline to handle model fitting from data extraction to saving results
"""

import json
import os
import pickle
import shutil
import uuid
from datetime import date, datetime
from glob import glob

import numpy as np
import pandas as pd

from . import changepoint_model
from . import changepoint_preprocess
from .utils import EphysData

MODEL_SAVE_DIR = '/media/bigdata/firing_space_plot/changepoint_mcmc/saved_models'
MODEL_DATABASE_PATH = os.path.join(MODEL_SAVE_DIR, 'model_database.csv')


class FitHandler():
    """Class to handle pipeline of model fitting including:
    1) Loading data
    2) Preprocessing loaded arrays
    3) Fitting model
    4) Writing out fitted parameters to pkl file

    """

    def __init__(self,
                 data_dir,
                 taste_num,
                 region_name,
                 experiment_name=None,
                 model_params_path=None,
                 preprocess_params_path=None,
                 ):
        """Initialize FitHandler class

        Args:
            data_dir (str): Path to directory containing HDF5 file
            taste_num (int): Index of taste to perform fit on
                    (Corresponds to INDEX of taste in spike array, not actual dig_ins)
            region_name (str): Region on which to perform fit on
                    (must match regions in .info file)
            experiment_name (str, optional): Name given to fitted batch
                    (for metedata). Defaults to None.
            model_params_path (str, optional): Path to json file
                    containing model parameters. Defaults to None.
            preprocess_params_path (str, optional): Path to json file
                    containing preprocessing parameters. Defaults to None.

        Raises:
            Exception: If "experiment_name" is None
            Exception: If "taste_num" is not integer or "all"
        """

        # =============== Check for exceptions ===============
        if experiment_name is None:
            raise Exception('Please specify an experiment name')
        if not isinstance(taste_num, (int, str)):
            raise Exception('taste_num must be an integer or "all"')

        # =============== Save relevant arguments ===============
        self.data_dir = data_dir
        self.EphysData = EphysData(self.data_dir)
        #self.data = self.EphysData.get_spikes({"bla","gc","all"})

        self.taste_num = taste_num
        self.region_name = region_name
        self.experiment_name = experiment_name

        data_handler_init_kwargs = dict(zip(
            ['data_dir', 'experiment_name', 'taste_num', 'region_name'],
            [data_dir, experiment_name, taste_num, region_name]))
        self.database_handler = DatabaseHandler()
        self.database_handler.set_run_params(**data_handler_init_kwargs)

        if model_params_path is None:
            print('MODEL_PARAMS will have to be set')
        else:
            self.set_model_params(file_path=model_params_path)

        if preprocess_params_path is None:
            print('PREPROCESS_PARAMS will have to be set')
        else:
            self.set_preprocess_params(file_path=preprocess_params_path)

        # Attributes to be set later
        self.preprocessor = None
        self.model_template = None
        self.inference_func = None
        self.data = None
        self.preprocessed_data = None
        self.model = None
        self.inference_outs = None


    ########################################
    # SET PARAMS
    ########################################

    def set_preprocess_params(self,
                              time_lims,
                              bin_width,
                              data_transform,
                              file_path=None):
        """Load given params as "preprocess_params" attribute

        Args:
            time_lims (array/tuple/list): Start and end of where to cut
                    spike train array
            bin_width (int): Bin width for binning spikes to counts
            data_transform (str): Indicator for which transformation to
                    use (refer to changepoint_preprocess)
            file_path (str, optional): Path to json file containing preprocess
                    parameters. Defaults to None.
        """

        if file_path is None:
            self.preprocess_params = \
                dict(zip(['time_lims', 'bin_width', 'data_transform'],
                         [time_lims, bin_width, data_transform]))
        else:
            # Load json and save dict
            pass

    def set_model_params(self,
                         states,
                         fit,
                         samples,
                         file_path=None):
        """Load given params as "model_params" attribute

        Args:
            states (int): Number of states to use in model
            fit (int): Iterations to use for model fitting (given ADVI fit)
            samples (int): Number of samples to return from fitten model
            file_path (str, optional): Path to json file containing
                    preprocess parameters. Defaults to None.
        """

        if file_path is None:
            self.model_params = \
                dict(zip(['states', 'fit', 'samples'], [states, fit, samples]))
        else:
            # Load json and save dict
            pass

    ########################################
    # SET PIPELINE FUNCS
    ########################################

    def set_preprocessor(self, preprocessing_func):
        """Manually set preprocessor for data e.g.

        FitHandler.set_preprocessor(
                    changepoint_preprocess.preprocess_single_taste)

        Args:
            preprocessing_func (func):
                    Function to preprocess data (refer to changepoint_preprocess)
        """
        self.preprocessor = preprocessing_func

    def preprocess_selector(self):
        """Function to return preprocess function based off of input flag

        Preprocessing can be set manually but it is preferred to
        go through preprocess selector

        Raises:
            Exception: If self.taste_num is neither int nor str

        """

        if isinstance(self.taste_num, int):
            self.set_preprocessor(
                changepoint_preprocess.preprocess_single_taste)
        if isinstance(self.taste_num, str):
            self.set_preprocessor(changepoint_preprocess.preprocess_all_taste)
        raise Exception("Something went wrong")

    def set_model_template(self, model_template):
        """Manually set model_template for data e.g.

        FitHandler.set_model(changepoint_model.single_taste_poisson)

        Args:
            model_template (func): Function to generate model template for data]
        """
        self.model_template = model_template

    def model_template_selector(self):
        """Function to set model based off of input flag

        Models can be set manually but it is preferred to go through model selector

        Raises:
            Exception: If self.taste_num is neither int nor str

        """
        if isinstance(self.taste_num, int):
            self.set_model_template(changepoint_model.single_taste_poisson)
        if isinstance(self.taste_num, str):
            self.set_model_template(changepoint_model.all_taste_poisson)
        raise Exception("Something went wrong")

    def set_inference(self, inference_func):
        """Manually set inference function for model fit e.g.

        FitHandler.set_inference(changepoint_model.advi_fit)

        Args:
            inference_func (func): Function to use for fitting model
        """
        self.inference_func = changepoint_model.advi_fit

    def inference_func_selector(self):
        """Function to return model based off of input flag

        Currently hard-coded to use "advi_fit"
        """
        self.set_inference(changepoint_model.advi_fit)

    ########################################
    # PIPELINE FUNCS
    ########################################

    def load_spike_trains(self):
        """Helper function to load spike trains from data_dir using EphysData module
        """
        full_spike_array = self.EphysData.return_region_spikes(
            self.region_name)
        if isinstance(self.taste_num, int):
            self.data = full_spike_array[self.taste_num]
        if isinstance(self.taste_num, str):
            self.data = full_spike_array
        print(f'Loading spike trains from {self.database_handler.data_basename}, '
              f'dig_in {self.taste_num}')

    def preprocess_data(self):
        """Perform data preprocessing

        Will check for and complete:
        1) Raw data loaded
        2) Preprocessor selected
        """
        if 'data' not in dir(self):
            self.load_spike_trains()
        if 'preprocessor' not in dir(self):
            self.preprocess_selector()
        print('Preprocessing spike trains, '
              f'preprocessing func: <{self.preprocessor.__name__}>')
        self.preprocessed_data = \
            self.preprocessor(self.data, **self.preprocess_params)

    def create_model(self):
        """Create model and save as attribute

        Will check for and complete:
        1) Data preprocessed
        2) Model template selected
        """
        if 'preprocessed_data' not in dir(self):
            self.preprocess_data()
        if 'model_template' not in dir(self):
            self.model_template_selector()

        # In future iterations, before fitting model,
        # check that a similar entry doesn't exist

        print(
            f'Generating Model, model func: <{self.model_template.__name__}>')
        self.model = self.model_template(self.preprocessed_data,
                                         self.model_params['states'])

    def run_inference(self):
        """Perform inference on data

        Will check for and complete:
        1) Model created
        2) Inference function selected
        """
        if 'model' not in dir(self):
            self.create_model()
        if 'inference_func' not in dir(self):
            self.inference_func_selector()

        print('Running inference, inference func: '
              f'<{self.inference_func.__name__}>')
        temp_outs = self.inference_func(self.model,
                                        self.model_params['fit'],
                                        self.model_params['samples'])
        varnames = ['model', 'approx', 'lambda', 'tau', 'data']
        self.inference_outs = dict(zip(varnames, temp_outs))

    def _gen_fit_metadata(self):
        """Generate metadata for fit

        Generate metadat by compiling:
        1) Preprocess parameters given as input
        2) Model parameters given as input
        3) Functions used in inference pipeline for : preprocessing,
                model generation, fitting

        Returns:
            dict: Dictionary containing compiled metadata for different
                    parts of inference pipeline
        """
        pre_params = self.preprocess_params
        model_params = self.model_params
        pre_params['preprocessor_name'] = self.preprocessor.__name__
        model_params['model_template_name'] = self.model_template.__name__
        model_params['inference_func_name'] = self.inference_func.__name__
        fin_dict = dict(zip(['preprocess', 'model'], [pre_params, model_params]))
        return fin_dict

    def _pass_metadata_to_handler(self):
        """Function to coordinate transfer of metadata to DatabaseHandler
        """
        self.database_handler.ingest_fit_data(self._gen_fit_metadata())

    def _return_fit_output(self):
        """Compile data, model, fit, and metadata to save output

        Returns:
            dict: Dictionary containing fitted model data and metadata
        """
        self._pass_metadata_to_handler()
        agg_metadata = self.database_handler.aggregate_metadata()
        return {'model_data': self.inference_outs, 'metadata': agg_metadata}

    def save_fit_output(self):
        """Save fit output (fitted data + metadata) to pkl file
        """
        if 'inference_outs' not in dir(self):
            self.run_inference()
        out_dict = self._return_fit_output()
        with open(self.database_handler.model_save_path + '.pkl', 'wb') as buff:
            pickle.dump(out_dict, buff)

        json_file_name = os.path.join(
            self.database_handler.model_save_path + '.info')
        with open(json_file_name, 'w') as file:
            json.dump(out_dict['metadata'], file, indent=4)

        self.database_handler.write_to_database()

        print('Saving inference output to '
              f'{self.database_handler.model_save_dir}')


class DatabaseHandler():
    """Class to handle transactions with model database
    """

    def __init__(self):
        """Initialize DatabaseHandler class
        """
        self.unique_cols = ['exp.model_id', 'exp.save_path', 'exp.fit_date']
        self.model_database_path = MODEL_DATABASE_PATH
        self.model_save_base_dir = MODEL_SAVE_DIR

        if os.path.exists(self.model_database_path):
            self.fit_database = pd.read_csv(self.model_database_path,
                                            index_col=0)
            all_na = [all(x) for num, x in self.fit_database.isna().iterrows()]
            if all_na:
                print(f'{sum(all_na)} rows found with all NA, removing...')
                self.fit_database = self.fit_database.dropna(how='all')
        else:
            print('Fit database does not exist yet')

    def show_duplicates(self, keep='first'):
        """Find duplicates in database

        Args:
            keep (str, optional): Which duplicate to keep
                    (refer to pandas duplicated). Defaults to 'first'.

        Returns:
            pandas dataframe: Dataframe containing duplicated rows
            pandas series : Indices of duplicated rows
        """
        dup_inds = self.fit_database.drop(self.unique_cols, axis=1)\
            .duplicated(keep=keep)
        return self.fit_database.loc[dup_inds], dup_inds

    def drop_duplicates(self):
        """Remove duplicated rows from database
        """
        _, dup_inds = self.show_duplicates()
        print(f'Removing {sum(dup_inds)} duplicate rows')
        self.fit_database = self.fit_database.loc[~dup_inds]

    def check_mismatched_paths(self):
        """Check if there are any mismatched pkl files between database and directory

        Returns:
            pandas dataframe: Dataframe containing rows for which pkl file not present
            list: pkl files which cannot be matched to model in database
            list: all files in save directory
        """
        mismatch_from_database = [not os.path.exists(x + ".pkl")
                                  for x in self.fit_database['exp.save_path']]
        file_list = glob(os.path.join(self.model_save_base_dir, "*/*.pkl"))
        mismatch_from_file = [not
                              (x.split('.')[0] in list(
                                  self.fit_database['exp.save_path']))
                              for x in file_list]
        print(f"{sum(mismatch_from_database)} mismatches from database" + "\n"
              + f"{sum(mismatch_from_file)} mismatches from files")
        return mismatch_from_database, mismatch_from_file, file_list

    def clear_mismatched_paths(self):
        """Remove mismatched files and rows in database

        i.e. Remove
        1) Files for which no entry can be found in database
        2) Database entries for which no corresponding file can be found
        """
        mismatch_from_database, mismatch_from_file, file_list = \
            self.check_mismatched_paths()
        mismatch_from_file = np.array(mismatch_from_file)
        mismatch_from_database = np.array(mismatch_from_database)
        self.fit_database = self.fit_database.loc[~mismatch_from_database]
        mismatched_files = [x for x, y in zip(file_list, mismatch_from_file) if y]
        for x in mismatched_files:
            os.remove(x)
        print('==== Clearing Completed ====')

    def write_updated_database(self):
        """Can be called following clear_mismatched_entries to update current database
        """
        database_backup_dir = os.path.join(
            self.model_save_base_dir, '.database_backups')
        if not os.path.exists(database_backup_dir):
            os.makedirs(database_backup_dir)
        #current_date = date.today().strftime("%m-%d-%y")
        current_date = str(datetime.now()).replace(" ", "_")
        shutil.copy(self.model_database_path,
                    os.path.join(database_backup_dir,
                                 f"database_backup_{current_date}"))
        self.fit_database.to_csv(self.model_database_path, mode='w')

    def set_run_params(self, data_dir, experiment_name, taste_num, region_name):
        """Store metadata related to inference run

        Args:
            data_dir (str): Path to directory containing HDF5 file
            experiment_name (str): Name given to fitted batch
                    (for metedata). Defaults to None.
            taste_num (int): Index of taste to perform fit on (Corresponds to
                    INDEX of taste in spike array, not actual dig_ins)
            region_name (str): Region on which to perform fit on
                    (must match regions in .info file)
        """
        self.data_dir = data_dir
        self.data_basename = os.path.basename(self.data_dir)
        self.animal_name = self.data_basename.split("_")[0]
        self.session_date = self.data_basename.split("_")[-1]

        self.experiment_name = experiment_name
        self.model_save_dir = os.path.join(self.model_save_base_dir,
                                           experiment_name)

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        self.model_id = str(uuid.uuid4()).split('-')[0]
        self.model_save_path = os.path.join(self.model_save_dir,
                                            self.experiment_name +
                                            "_" + self.model_id)
        self.fit_date = date.today().strftime("%m-%d-%y")

        self.taste_num = taste_num
        self.region_name = region_name

        self.fit_exists = None

    def ingest_fit_data(self, met_dict):
        """Load external metadata

        Args:
            met_dict (dict): Dictionary of metadata from FitHandler class
        """
        self.external_metadata = met_dict

    def aggregate_metadata(self):
        """Collects information regarding data and current "experiment"

        Raises:
            Exception: If 'external_metadata' has not been ingested, that needs to be done first

        Returns:
            dict: Dictionary of metadata given to FitHandler class
        """
        if 'external_metadata' not in dir(self):
            raise Exception('Fit run metdata needs to be ingested '
                            'into data_handler first')

        data_details = dict(zip(
            ['data_dir',
             'basename',
             'animal_name',
             'session_date',
             'taste_num',
             'region_name'],
            [self.data_dir,
             self.data_basename,
             self.animal_name,
             self.session_date,
             self.taste_num,
             self.region_name]))

        exp_details = dict(zip(
            ['exp_name', 
             'model_id',
             'save_path',
             'fit_date'],
            [self.experiment_name,
             self.model_id,
             self.model_save_path,
             self.fit_date]))

        temp_ext_met = self.external_metadata
        temp_ext_met['data'] = data_details
        temp_ext_met['exp'] = exp_details

        return temp_ext_met

    def write_to_database(self):
        """Write out metadata to database
        """
        agg_metadata = self.aggregate_metadata()
        flat_metadata = pd.json_normalize(agg_metadata)
        if not os.path.isfile(self.model_database_path):
            flat_metadata.to_csv(self.model_database_path, mode='a')
        else:
            flat_metadata.to_csv(self.model_database_path,
                                 mode='a', header=False)

    def check_exists(self):
        """Check if the given fit already exists in database

        Returns:
            bool: Boolean for whether fit already exists or not
        """
        if self.fit_exists is not None:
            return self.fit_exists
