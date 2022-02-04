import sys
import os
import pandas as pd
import numpy as np
import pickle as pkl
from scipy.stats import mode
from ephys_data import ephys_data

def get_state_firing(spike_array,tau_array):
    """
    spike_array : trials x nrns x bins
    tau_array : trials x switchpoints
    """
    states = tau_array.shape[-1] + 1
    # Get mean firing rate for each STATE using model
    state_inds = np.hstack([np.zeros((tau_array.shape[0],1)),
                            tau_array,
                            np.ones((tau_array.shape[0],1))*spike_array.shape[-1]])
    state_lims = np.array([state_inds[:,x:x+2] for x in range(states)])
    state_lims = np.vectorize(np.int)(state_lims)
    state_lims = np.swapaxes(state_lims,0,1)

    state_firing = \
            np.array([[np.mean(trial_dat[:,start:end],axis=-1) \
            for start, end in trial_lims] \
            for trial_dat, trial_lims in zip(spike_array,state_lims)])

    state_firing = np.nan_to_num(state_firing)
    return state_firing

class _firing():
    def __init__(self, tau_instance, processed_spikes, metadata):
        self.tau = tau_instance
        self.processed_spikes = processed_spikes
        self.metadata = metadata
        self._ephys_data = ephys_data(self.metadata['data']['data_dir'])
        temp_spikes = self._ephys_data.return_region_spikes(
                        self.metadata['data']['region_name'])
        self.raw_spikes = temp_spikes[self.metadata['data']['taste_num']]
        self.state_firing = get_state_firing(self.processed_spikes,
                                            self.tau.raw_mode_tau)

class _tau():
    def __init__(self, tau_array, metadata):
        self.raw_tau = tau_array
        #self.metadata = metadata

        #def process_tau(self):
        #time_lims = self.metadata['preprocess']['time_lims']
        #bin_width = self.metadata['preprocess']['bin_width']
        time_lims = metadata['preprocess']['time_lims']
        bin_width = metadata['preprocess']['bin_width']

        self.raw_int_tau = np.vectorize(np.int)(self.raw_tau)
        self.raw_mode_tau = mode(self.raw_int_tau)[0][0]

        self.scaled_tau = (self.raw_tau * bin_width) + time_lims[0]
        self.scaled_int_tau = np.vectorize(np.int)(self.scaled_tau)
        self.scaled_mode_tau = np.squeeze(mode(self.scaled_int_tau)[0])

class pkl_handler():
    def __init__(self, file_path):
        self.dir_name = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        self.file_name_base = file_name.split('.')[0]
        self.pkl_file_path = \
                os.path.join(self.dir_name, self.file_name_base + ".pkl")
        with open(self.pkl_file_path, 'rb') as f:
                self.data = pkl.load(f)

        model_keys = ['model', 'approx', 'lambda', 'tau', 'data']
        key_savenames = ['_model_structure','_fit_model',
                            'lambda_array','tau_array','processed_spikes']
        data_map = dict(zip(model_keys, key_savenames))

        for key, var_name in data_map.items():
            setattr(self, var_name, self.data['model_data'][key])

        self.metadata = self.data['metadata']
        self.pretty_metadata = pd.json_normalize(self.data['metadata']).T

        #self.tau = _tau(self.tau_array, self.metadata)
        #self.firing = _firing(self.tau, self.processed_spikes, self.metadata)

