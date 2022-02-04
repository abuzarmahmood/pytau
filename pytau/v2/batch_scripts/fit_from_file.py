import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
sys.path.append('/media/bigdata/pytau/pytau/v2')
from ephys_data import ephys_data
from changepoint_io import fit_handler
import itertools as it
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import ast
from joblib import Parallel, delayed, cpu_count
import argparse
import sys

parser = argparse.ArgumentParser(description = 'Run single fit with parameters from file')
parser.add_argument('param_path_file',  help = 'JSON file containing parameters for fit')
args = parser.parse_args()
#parallel_temp_path = '/media/bigdata/pytau/pytau/v2/parallel_temp'

#job_file_path = args.param_path_file
job_file_path = sys.argv[1] 
input_params = pd.read_json(f'{job_file_path}').T
print(f'Reading file {job_file_path}')

print(input_params.T)

def fit_func(input_params):
    """
    Corresponds to this_iter in below loop
    """
    #for key,val in zip(param_dict.keys(), input_params):
    #    locals()[key] = val

    # Define arguments/parameters
    model_parameters_keys = ['states','fit','samples']
    preprocess_parameters_keys = ['time_lims','bin_width','data_transform']
    fit_handler_kwargs_keys = ['data_dir','taste_num','region_name','experiment_name']

    model_parameters = dict(zip(model_parameters_keys, 
                            input_params[model_parameters_keys].iloc[0]))
    preprocess_parameters = dict(zip(preprocess_parameters_keys, 
                            input_params[preprocess_parameters_keys].iloc[0]))
    fit_handler_kwargs = dict(zip(fit_handler_kwargs_keys, 
                            input_params[fit_handler_kwargs_keys].iloc[0]))

    #model_parameters = dict(zip(,
    #                                [states,fit,samples]))
    #preprocess_parameters = dict(zip(,
    #                                [time_lims,bin_width, data_transform]))
    #fit_handler_kwargs = {'data_dir' : data_dir,
    #                    'taste_num' : taste_num,
    #                    'region_name' : region_name,
    #                    'experiment_name' : experiment_name}

    ## Initialize handler, and feed paramters
    handler = fit_handler(**fit_handler_kwargs)
    handler.set_model_params(**model_parameters)
    handler.set_preprocess_params(**preprocess_parameters)

    error_file_path = os.path.join(
            handler.database_handler.model_save_dir,
            'error_log_file.txt')

    try:
        handler.run_inference()
        handler.save_fit_output()
        return 0
    except:
        # Only print out unique values
        # Assuming if there is something wrong in a few iterations,
        # it is likelt identifiable on the unique values
        this_unique_vals = [globals()[x] for x in unique_keys]
        # If file is not present, write keys to header
        if not os.path.isfile(error_file_path):
            with open(error_file_path,'a') as error_file:
                error_file.write(str(unique_keys) + "\n")
                error_file.write(str(this_unique_vals) + "\n")
        else:
            with open(error_file_path,'a') as error_file:
                error_file.write(str(this_unique_vals) + "\n")
        return 1

exit_code = fit_func(input_params)
#fit_func(input_params)
#
if exit_code == 0:
    os.remove(job_file_path)
