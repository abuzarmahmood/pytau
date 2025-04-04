import os
import sys

base_dir = "/media/bigdata/projects/pytau"
# sys.path.append(os.path.join(base_dir, 'utils'))
sys.path.append(base_dir)
import argparse

import numpy as np
import pandas as pd

# from ephys_data import EphysData
from pytau.changepoint_io import FitHandler

parser = argparse.ArgumentParser(description="Run single fit with parameters from file")
parser.add_argument("param_path_file", help="JSON file containing parameters for fit")
args = parser.parse_args()
# parallel_temp_path = '/media/bigdata/pytau/pytau/v2/parallel_temp'

# job_file_path = '/media/bigdata/projects/pytau/pytau/utils/batch_utils/parallel_temp/job0015.json'
job_file_path = args.param_path_file
input_params = pd.read_json(f"{job_file_path}").T
# print(f'Reading file {job_file_path}')
# print(input_params.T)


def fit_func(input_params):
    """
    Corresponds to this_iter in below loop
    """

    # Define arguments/parameters
    model_parameters_keys = ["states", "fit", "samples", "model_kwargs"]
    preprocess_parameters_keys = ["time_lims", "bin_width", "data_transform"]
    FitHandler_kwargs_keys = [
        "data_dir",
        "taste_num",
        "region_name",
        "laser_type",
        "experiment_name",
    ]

    model_parameters = dict(
        zip(model_parameters_keys, input_params[model_parameters_keys].iloc[0])
    )
    preprocess_parameters = dict(
        zip(
            preprocess_parameters_keys, input_params[preprocess_parameters_keys].iloc[0]
        )
    )
    FitHandler_kwargs = dict(
        zip(FitHandler_kwargs_keys, input_params[FitHandler_kwargs_keys].iloc[0])
    )

    ## Initialize handler, and feed paramters
    handler = FitHandler(**FitHandler_kwargs)
    handler.set_model_params(**model_parameters)
    handler.set_preprocess_params(**preprocess_parameters)

    error_file_path = os.path.join(
        handler.database_handler.model_save_dir, "error_log_file.txt"
    )

    try:
        handler.run_inference()
        handler.save_fit_output()
        return 0
    except:
        # Only print out unique values
        # Assuming if there is something wrong in a few iterations,
        # it is likelt identifiable on the unique values
        # If file is not present, write keys to header
        if not os.path.isfile(error_file_path):
            # with open(error_file_path,'a') as error_file:
            #    error_file.write(str(unique_keys) + "\n")
            #    error_file.write(str(this_unique_vals) + "\n")
            input_params.to_csv(error_file_path)
        else:
            # with open(error_file_path,'a') as error_file:
            #    error_file.write(str(this_unique_vals) + "\n")
            input_params.to_csv(error_file_path, mode="a", header=False)
        return 1


exit_code = fit_func(input_params)
# fit_func(input_params)
#
if exit_code == 0:
    os.remove(job_file_path)
