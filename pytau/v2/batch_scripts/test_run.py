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

parallel_temp_path = '/media/bigdata/pytau/pytau/v2/parallel_temp'
########################################
## Specify Things HERE
########################################

# Define all iterators
dir_list_path = '/media/bigdata/firing_space_plot/changepoint_mcmc/v2/bla_dirs.txt'
dir_list = [x.strip() for x in open(dir_list_path,'r').readlines()]
#dir_list = [
#        '/media/bigdata/Abuzar_Data/bla_only/AM28/AM28_4Tastes_201006_095803']


# Exp params
# NOTE :: EVERYTHING needs to be a list
# Except time_lims which has to be a list of lists
exp_model_parameters = {'states' :      [i for i in range(8,11)],
                        'fit' :         [40000],
                        'samples' :     [20000]
                        }

exp_preprocess_parameters = {   
                        'time_lims' :   [ [2000,4000] ],
                        'bin_width' :   [50],
                        'data_transform' : ['None','spike_shuffled',
                                            'trial_shuffled', 'simulated']
                        }

exp_fit_handler_kwargs = {
                        'data_dir' :    dir_list,
                        'taste_num' :   ['all'],
                        'region_name' : ['bla'],
                        'experiment_name' : ['bla_transition_strength']
                        }

########################################
########################################

param_dict = {**exp_model_parameters, **exp_preprocess_parameters, 
                        **exp_fit_handler_kwargs}
iter_list = list(it.product(*list(param_dict.values())))
iter_list = [dict(zip(param_dict.keys(), x)) for x in iter_list]
iter_frame = pd.DataFrame(iter_list)

unique_vals = []
unique_keys = []
for col in iter_frame:
    unique_vals.append(np.unique(iter_frame[col]))

for name,vals in zip(param_dict.keys(), unique_vals):
    if len(vals) > 1:
        print(f"Total iterations : {iter_frame.shape[0]}" + "\n")
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

for num, this_row in iter_frame.iterrows():
    temp_frame = pd.DataFrame(this_row)
    temp_frame.to_json(os.path.join(parallel_temp_path, f'job{num:04}.json'))

########################################

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
                            input_params[model_parameters_keys]))
    preprocess_parameters = dict(zip(preprocess_parameters_keys, 
                            input_params[preprocess_parameters_keys]))
    fit_handler_kwargs = dict(zip(fit_handler_kwargs_keys, 
                            input_params[fit_handler_kwargs_keys]))

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

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

input_params_list = [x[1] for x in iter_frame.iterrows()] 
parallelize(fit_func, input_params_list) 

#=================================================

#for num,this_iter in tqmd(iter_frame.iterrows()):
#    #num, this_iter = next(iter_frame.iterrows())
#
#    for key,val in zip(param_dict.keys(), this_iter):
#        globals()[key] = val
#
#    # Define arguments/parameters
#    model_parameters = dict(zip(['states','fit','samples'],
#                                    [states,fit,samples]))
#    preprocess_parameters = dict(zip(['time_lims','bin_width','data_transform'],
#                                    [time_lims,bin_width, data_transform]))
#    fit_handler_kwargs = {'data_dir' : data_dir,
#                        'taste_num' : taste_num,
#                        'region_name' : region_name,
#                        'experiment_name' : experiment_name}
#
#    # Initialize handler, and feed paramters
#    handler = fit_handler(**fit_handler_kwargs)
#    handler.set_model_params(**model_parameters)
#    handler.set_preprocess_params(**preprocess_parameters)
#
#    error_file_path = os.path.join(
#            handler.database_handler.model_save_dir,
#            'error_log_file.txt')
#
#    try:
#        handler.run_inference()
#        handler.save_fit_output()
#    except:
#        # Only print out unique values
#        # Assuming if there is something wrong in a few iterations,
#        # it is likelt identifiable on the unique values
#        this_unique_vals = [globals()[x] for x in unique_keys]
#        # If file is not present, write keys to header
#        if not os.path.isfile(error_file_path):
#            with open(error_file_path,'a') as error_file:
#                error_file.write(str(unique_keys) + "\n")
#                error_file.write(str(this_unique_vals) + "\n")
#        else:
#            with open(error_file_path,'a') as error_file:
#                error_file.write(str(this_unique_vals) + "\n")
