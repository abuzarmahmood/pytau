import itertools as it
import os

import numpy as np
import pandas as pd


def entry_checker(msg, check_func, fail_response):
    check_bool = False
    continue_bool = True
    exit_str = '"x" to exit :: '
    while not check_bool:
        msg_input = input(msg.join([" ", exit_str]))
        if msg_input == "x":
            continue_bool = False
            break
        check_bool = check_func(msg_input)
        if not check_bool:
            print(fail_response)
    return msg_input, continue_bool


parallel_temp_path = "/media/bigdata/projects/pytau/pytau/utils" "/batch_utils/parallel_temp"
# dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
# dir_list_path = '/media/bigdata/projects/pytau/pytau/data/all_gc_dirs.txt'
# dir_list_path = '/media/bigdata/projects/pytau/pytau/data/bla_dirs.txt'
dir_list_path = "/media/bigdata/firing_space_plot/NM_gape_analysis/fin_NM_emg_dat.txt"
########################################
# Specify Things HERE
########################################

# Define all iterators
dir_list = [x.strip() for x in open(dir_list_path, "r").readlines()]
# dir_list = [
#        '/media/bigdata/Abuzar_Data/bla_only/AM28/AM28_4Tastes_201006_095803']


# Exp params
# NOTE :: EVERYTHING needs to be a list
# Except time_lims which has to be a list of lists
exp_model_parameters = {
    "states": [4],
    "fit": [80000],
    "samples": [20000],
    "model_kwargs": [{"None": None}],
}

exp_preprocess_parameters = {
    "time_lims": [[2000, 4000]],
    "bin_width": [50],
    "data_transform": ["None"],  # ,
    # 'spike_shuffled',
    # 'trial_shuffled']
}

exp_fit_handler_kwargs = {
    "data_dir": dir_list,
    "taste_num": [0, 1, 2, 3],
    "region_name": ["gc"],
    "laser_type": ["off"],
    "experiment_name": ["NM_EMG"],
}

########################################
########################################

param_dict = {
    **exp_model_parameters,
    **exp_preprocess_parameters,
    **exp_fit_handler_kwargs,
}
iter_list = list(it.product(*list(param_dict.values())))
iter_list = [dict(zip(param_dict.keys(), x)) for x in iter_list]
iter_frame = pd.DataFrame(iter_list)

unique_vals = []
unique_keys = []
for col in iter_frame:
    unique_vals.append(np.unique(iter_frame[col].astype("str")))

print(f"Total iterations : {iter_frame.shape[0]}" + "\n")
for name, vals in zip(param_dict.keys(), unique_vals):
    if len(vals) > 1:
        print(str({name: vals}) + "\n")
        unique_keys.append(name)

_, continue_bool = entry_checker(
    msg="Everything looks good? (y) :: ",
    check_func=lambda x: x in ["y"],
    fail_response="Please enter (y) :: ",
)
if continue_bool:
    pass
else:
    raise Exception("Things were stopped because you said so")

for num, this_row in iter_frame.iterrows():
    temp_frame = pd.DataFrame(this_row)
    temp_frame.to_json(os.path.join(parallel_temp_path,
                       f"job{num:04}.json"), indent=4)
