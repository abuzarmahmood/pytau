"""
Pipeline to handle model fitting from data extraction to saving results
"""

import os
import uuid
import pickle
import pandas as pd
import json
from datetime import date, datetime
from time import time
import shutil
from glob import glob
import numpy as np
import changepoint_preprocess
import changepoint_model
from ephys_data import ephys_data


MODEL_SAVE_DIR = '/media/bigdata/firing_space_plot/changepoint_mcmc/saved_models'
MODEL_DATABASE_PATH = os.path.join(MODEL_SAVE_DIR, 'model_database.csv')

class fit_handler():

    def __init__(self,
                data_dir,
                taste_num,
                region_name,
                experiment_name = None,
                model_params_path = None,
                preprocess_params_path = None,
                ):
        """
        taste_num: integer value, or 'all'
                - There should be a way to cross-reference whether
                    the model will accept a particular array type
                - Corresponds to INDEX of taste in spike array, not actual dig_ins
        """

        # =============== Check for exceptions ===============
        if experiment_name is None:
            raise Exception('Please specify an experiment name')
        if not (isinstance(taste_num,int) or isinstance(taste_num,str)):
            raise Exception('taste_num must be an integer or "all"')

        # =============== Save relevant arguments ===============
        self.data_dir = data_dir

        self.taste_num = taste_num
        self.region_name = region_name
        self.experiment_name = experiment_name

        data_handler_init_kwargs = dict(zip(
                        ['data_dir','experiment_name','taste_num','region_name'],
                        [data_dir, experiment_name, taste_num, region_name]))
        self.database_handler = database_handler()
        self.database_handler.set_run_params(**data_handler_init_kwargs)

        if model_params_path is None:
            print('MODEL_PARAMS will have to be set')
        else: 
            self.set_model_params(file_path = model_params_path)

        if preprocess_params_path is None:
            print('PREPROCESS_PARAMS will have to be set')
        else: 
            self.set_preprocess_params(file_path = preprocess_params_path)


    ########################################
    ## SET PARAMS
    ########################################

    def set_preprocess_params(self, 
                            time_lims, 
                            bin_width, 
                            data_transform,
                            file_path = None): 

        if file_path is None:
            self.preprocess_params = \
                    dict(zip(['time_lims','bin_width','data_transform'], 
                        [time_lims, bin_width, data_transform]))
        else:
            # Load json and save dict
            pass

    def set_model_params(self, 
                        states, 
                        fit, 
                        samples, 
                        file_path = None): 

        if file_path is None:
            self.model_params = \
                    dict(zip(['states','fit','samples'], [states,fit,samples]))
        else:
            # Load json and save dict
            pass


    ########################################
    ## SET PIPELINE FUNCS
    ########################################

    def set_preprocessor(self, preprocessing_func):
        """
        fit_handler.set_preprocessor(
                    changepoint_preprocess.preprocess_single_taste)
        """
        self.preprocessor = preprocessing_func

    def preprocess_selector(self):
        """
        Preprocessing can be set manually but it is preferred to go 
            through preprocess selector
        Function to return preprocess function based off of input flag 
        """
        if isinstance(self.taste_num,int):
            self.set_preprocessor(changepoint_preprocess.preprocess_single_taste)
        elif isinstance(self.taste_num,str):
            self.set_preprocessor(changepoint_preprocess.preprocess_all_taste)
        else:
            raise Exception("Something went wrong")
        # self.set_preprocessor(...)
        #pass

    def set_model_template(self, model_template):
        """
        Models can be set manually but it is preferred to go through model selector
        fit_handler.set_model(changepoint_model.single_taste_poisson)
        """
        self.model_template = model_template

    def model_template_selector(self):
        """
        Function to return model based off of input flag 
        """
        if isinstance(self.taste_num,int):
            self.set_model_template(changepoint_model.single_taste_poisson)
        elif isinstance(self.taste_num,str):
            self.set_model_template(changepoint_model.all_taste_poisson)
        else:
            raise Exception("Something went wrong")
        # self.set_model(...)
        #pass

    def set_inference(self, inference_func):
        """
        fit_handler.set_inference(changepoint_model.advi_fit)
        """
        self.inference_func = changepoint_model.advi_fit

    def inference_func_selector(self):
        """
        Function to return model based off of input flag 
        """
        self.set_inference(changepoint_model.advi_fit)

    ########################################
    ## PIPELINE FUNCS 
    ########################################

    def load_spike_trains(self):
        self.ephys_data = ephys_data(self.data_dir)
        #self.data = self.ephys_data.get_spikes({"bla","gc","all"})
        full_spike_array = self.ephys_data.return_region_spikes(self.region_name)
        if isinstance(self.taste_num,int):
            self.data = full_spike_array[self.taste_num] 
        elif isinstance(self.taste_num,str):
            self.data = full_spike_array 
        print(f'Loading spike trains from {self.database_handler.data_basename}, '
                f'dig_in {self.taste_num}')
        pass

    def preprocess_data(self):
        if 'data' not in dir(self):
            self.load_spike_trains()
        if 'preprocessor' not in dir(self):
            self.preprocess_selector()
        print('Preprocessing spike trains, '
                f'preprocessing func: <{self.preprocessor.__name__}>')
        self.preprocessed_data = \
                self.preprocessor(self.data, **self.preprocess_params)

    def create_model(self):
        if 'preprocessed_data' not in dir(self):
            self.preprocess_data()
        if 'model_template' not in dir(self):
            self.model_template_selector()

        # Before fitting model, check that a similar entry doesn't exist

        print(f'Generating Model, model func: <{self.model_template.__name__}>')
        self.model = self.model_template(self.preprocessed_data,
                        self.model_params['states'])

    def run_inference(self):
        if 'model' not in dir(self):
            self.create_model()
        if 'inference_func' not in dir(self):
            self.inference_func_selector()

        print('Running inference, inference func: '
                    f'<{self.inference_func.__name__}>')
        temp_outs = self.inference_func(self.model,
                        self.model_params['fit'], self.model_params['samples'])
        varnames = ['model','approx','lambda','tau','data']
        self.inference_outs = dict(zip(varnames, temp_outs)) 

    #def fit_pipeline(self):
    #    self.save_fit_output()

    def _gen_fit_metadata(self):
        pre_params = self.preprocess_params
        model_params = self.model_params
        pre_params['preprocessor_name'] = self.preprocessor.__name__
        model_params['model_template_name'] = self.model_template.__name__
        model_params['inference_func_name'] = self.inference_func.__name__
        fin_dict = dict(zip(['preprocess','model'], [pre_params, model_params]))
        return fin_dict 

    def _pass_metadata_to_handler(self):
        self.database_handler.ingest_fit_data(self._gen_fit_metadata())

    def _return_fit_output(self):
        """
        Compile data, model, fit, and metadata to save output
        """
        self._pass_metadata_to_handler()
        agg_metadata = self.database_handler.aggregate_metadata()
        return {'model_data': self.inference_outs, 'metadata' : agg_metadata} 

    def save_fit_output(self):
        if 'inference_outs' not in dir(self):
            self.run_inference()
        out_dict = self._return_fit_output()
        with open(self.database_handler.model_save_path + '.pkl', 'wb') as buff:
            pickle.dump(out_dict, buff)

        json_file_name = os.path.join(
                self.database_handler.model_save_path + '.info')
        with open(json_file_name,'w') as file:
            json.dump(out_dict['metadata'], file, indent = 4)

        self.database_handler.write_to_database()

        print('Saving inference output to '
                f'{self.database_handler.model_save_dir}')
        
class database_handler():
    
    def __init__(self):
        self.unique_cols = ['exp.model_id','exp.save_path','exp.fit_date']
        self.model_database_path = MODEL_DATABASE_PATH
        self.model_save_base_dir = MODEL_SAVE_DIR

        if os.path.exists(self.model_database_path):
            self.fit_database = pd.read_csv(self.model_database_path,
                                                index_col = 0)
            all_na = [all(x) for num,x in self.fit_database.isna().iterrows()]
            if all_na:
                print(f'{sum(all_na)} rows found with all NA, removing...')
                self.fit_database = self.fit_database.dropna(how='all')
        else:
            print('Fit database does not exist yet')

    def show_duplicates(self, keep = 'first'):
        dup_inds = self.fit_database.drop(self.unique_cols,axis=1)\
                .duplicated(keep=keep)
        return self.fit_database.loc[dup_inds], dup_inds

    def drop_duplicates(self):
        _, dup_inds = self.show_duplicates()
        print(f'Removing {sum(dup_inds)} duplicate rows')
        self.fit_database = self.fit_database.loc[~dup_inds]

    def check_mismatched_paths(self):
        mismatch_from_database = [not os.path.exists(x + ".pkl") \
                for x in self.fit_database['exp.save_path']]
        file_list = glob(os.path.join(self.model_save_base_dir, "*/*.pkl"))
        mismatch_from_file = [not \
                (x.split('.')[0] in list(self.fit_database['exp.save_path'])) \
                for x in file_list]
        print(f"{sum(mismatch_from_database)} mismatches from database" + "\n" \
                + f"{sum(mismatch_from_file)} mismatches from files")
        return mismatch_from_database, mismatch_from_file, file_list

    def clear_mismatched_paths(self):
        mismatch_from_database, mismatch_from_file, file_list = \
                self.check_mismatched_paths()
        mismatch_from_file = np.array(mismatch_from_file)
        mismatch_from_database = np.array(mismatch_from_database)
        self.fit_database = self.fit_database.loc[~mismatch_from_database]
        mismatched_files = [x for x,y in zip(file_list, mismatch_from_file) if y]
        for x in mismatched_files:
            os.remove(x)
        print('==== Clearing Completed ====')

    def write_updated_database(self):
        database_backup_dir = os.path.join(
            self.model_save_base_dir, '.database_backups')
        if not os.path.exists(database_backup_dir):
            os.makedirs(database_backup_dir)
        #current_date = date.today().strftime("%m-%d-%y")
        current_date = str(datetime.now()).replace(" ","_")
        shutil.copy(self.model_database_path,
            os.path.join(database_backup_dir, f"database_backup_{current_date}"))
        self.fit_database.to_csv(self.model_database_path, mode = 'w')

    def set_run_params(self, data_dir, experiment_name, taste_num, region_name):
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
                    self.experiment_name + "_" + self.model_id)
        self.fit_date = date.today().strftime("%m-%d-%y")

        self.taste_num = taste_num
        self.region_name = region_name

        self.fit_exists = None

    def ingest_fit_data(self, met_dict):
        """
        Load external metadata
        """
        self.external_metadata = met_dict

    def aggregate_metadata(self):
        if 'external_metadata' not in dir(self):
            raise Exception('Fit run metdata needs to be ingested '\
                    'into data_handler first')

        data_details = dict(zip(
            ['data_dir','basename','animal_name','session_date',
                'taste_num','region_name'],
            [self.data_dir, self.data_basename, self.animal_name,
                        self.session_date, self.taste_num, self.region_name]))

        exp_details = dict(zip(
            ['exp_name','model_id','save_path','fit_date'],
            [self.experiment_name, self.model_id, self.model_save_path,
                        self.fit_date]))

        temp_ext_met = self.external_metadata
        temp_ext_met['data'] = data_details
        temp_ext_met['exp'] = exp_details

        return temp_ext_met 

    def write_to_database(self):
        agg_metadata = self.aggregate_metadata()
        flat_metadata = pd.json_normalize(agg_metadata)
        if not os.path.isfile(self.model_database_path):
            flat_metadata.to_csv(self.model_database_path, mode='a')
        else:
            flat_metadata.to_csv(self.model_database_path, 
                    mode='a', header = False)

    def check_exists():
        if self.fit_exists is None:
            pass
        else:
            return self.fit_exists


