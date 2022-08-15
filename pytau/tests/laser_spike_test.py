"""
Test to make sure that the pipeline properly extracts laser vs non-laser trials
"""

import sys
import os
base_dir = '/media/bigdata/projects/pytau'
#sys.path.append(os.path.join(base_dir, 'utils'))
sys.path.append(base_dir)
#from ephys_data import EphysData
from pytau.changepoint_io import FitHandler
import pandas as pd
import numpy as np
import tables
from glob import glob

data_dir = '/media/fastdata/NM_sorted_data/NM51/NM51_500ms_161029_132459'
h5_path = glob(os.path.join(data_dir, "*.h5"))[0]

model_parameters_keys = ['states','fit','samples', 'model_kwargs']
model_parameters = [4, 40000, 20000, None]
preprocess_parameters_keys = ['time_lims','bin_width','data_transform']
preprocess_parameters = [[2000,4000], 50, None]
FitHandler_kwargs_keys = ['data_dir','taste_num',
                        'region_name','experiment_name', 'laser_type']
FitHandler_kwargs = [data_dir, 0, 'gc', 'EMG_analysis', 'on']

model_parameters = dict(zip(model_parameters_keys, model_parameters))
preprocess_parameters = dict(zip(preprocess_parameters_keys, preprocess_parameters))
FitHandler_kwargs = dict(zip(FitHandler_kwargs_keys, FitHandler_kwargs))

## Initialize handler, and feed paramters
handler = FitHandler(**FitHandler_kwargs)
handler.set_model_params(**model_parameters)
handler.set_preprocess_params(**preprocess_parameters)

handler.load_spike_trains()
loaded_on = handler.data

########################################
FitHandler_kwargs = [data_dir, 0, 'gc', 'EMG_analysis', None]
FitHandler_kwargs = dict(zip(FitHandler_kwargs_keys, FitHandler_kwargs))
## Initialize handler, and feed paramters
handler = FitHandler(**FitHandler_kwargs)
handler.set_model_params(**model_parameters)
handler.set_preprocess_params(**preprocess_parameters)

handler.load_spike_trains()
loaded_off = handler.data

############################################################
h5 = tables.open_file(h5_path,'r')
taste_ind = FitHandler_kwargs['taste_num']
spike_base_path = f'/spike_trains/dig_in_{taste_ind}'

laser_durations = h5.get_node(spike_base_path, 'laser_durations')[:]
spike_train = h5.get_node(spike_base_path, 'spike_array')[:]

on_spikes = spike_train[laser_durations > 0]
off_spikes = spike_train[laser_durations == 0]
############################################################

all((loaded_on == on_spikes).flatten())
all((loaded_off == off_spikes).flatten())
