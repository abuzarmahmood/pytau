import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
sys.path.append('/media/bigdata/firing_space_plot/changepoint_mcmc/v2')
from ephys_data import ephys_data
from changepoint_io import fit_handler
import itertools as it
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import ast

def entry_checker(msg, check_func, fail_response):
    check_bool = False
    continue_bool = True
    exit_str = '"x" to exit :: '
    while not check_bool:
        msg_input = input(msg.join([' ',exit_str]))
        if msg_input == 'x':
            continue_bool = False
            break
        check_bool = check_func(msg_input)
        if not check_bool:
            print(fail_response)
    return msg_input, continue_bool

########################################
## Specify Things HERE
########################################

# Define all iterators
dir_list_path = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/all_tastes/inter_region_dirs.txt'
dir_list = [x.strip() for x in open(dir_list_path,'r').readlines()]


# Exp params
# NOTE :: EVERYTHING needs to be a list
# Except time_lims which has to be a list of lists
exp_model_parameters = {'states' :      [2,3,4],
                        'fit' :         [40000],
                        'samples' :     [20000]
                        }

exp_preprocess_parameters = {   
                        'time_lims' :   [ [2000,4000] ],
                        'bin_width' :   [50],
                        'data_transform' : ['None','simulated']
                        }

exp_fit_handler_kwargs = {
                        'data_dir' :    dir_list,
                        'taste_num' :   [0,1,2,3],
                        'region_name' : ['bla'],
                        'experiment_name' : ['test']
                        }

########################################
########################################

param_dict = {**exp_model_parameters, **exp_preprocess_parameters, 
                        **exp_fit_handler_kwargs}
iter_list = list(it.product(*list(param_dict.values())))
iter_frame = pd.DataFrame(iter_list)

unique_vals = []
unique_keys = []
for col in iter_frame:
    unique_vals.append(np.unique(iter_frame[col]))

for name,vals in zip(param_dict.keys(), unique_vals):
    if len(vals) > 1:
        print(str({name:vals}) + "\n")
        unique_keys.append(name)

_, continue_bool = entry_checker(\
        msg = 'Everything looks good? (y) :: ',
        check_func = lambda x: x in ['y'],
        fail_response = 'Please enter (y) :: ')
if continue_bool:
    pass
else:
    raise Exception('Things were stopped because you said so')

########################################
########################################
for num,this_iter in iter_frame.iterrows():
    for key,val in zip(param_dict.keys(), this_iter):
        globals()[key] = val

    # Define arguments/parameters
    model_parameters = dict(zip(['states','fit','samples'],
                                    [states,fit,samples]))
    preprocess_parameters = dict(zip(['time_lims','bin_width','data_transform'],
                                    [time_lims,bin_width, data_transform]))
    fit_handler_kwargs = {'data_dir' : data_dir,
                        'taste_num' : taste_num,
                        'region_name' : region_name,
                        'experiment_name' : experiment_name}

    # Initialize handler, and feed paramters
    handler = fit_handler(**fit_handler_kwargs)
    handler.set_model_params(**model_parameters)
    handler.set_preprocess_params(**preprocess_parameters)

    error_file_path = os.path.join(
            handler.database_handler.model_save_dir,
            'error_log_file.txt')

    try:
        handler.run_inference()
        handler.save_fit_output()
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
