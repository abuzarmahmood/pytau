"""
Helper classes and functions to perform analysis on fitted models
"""
import os
import pickle as pkl

import numpy as np
import pandas as pd
from scipy.stats import mode

from .utils import EphysData


def get_state_firing(spike_array, tau_array):
    """Calculate firing rates within states given changepoint positions on data

    Args:
        spike_array (3D Numpy array): trials x nrns x bins
        tau_array (2D Numpy array): trials x switchpoints

    Returns:
        Numpy array: Average firing given state bounds
    """

    states = tau_array.shape[-1] + 1
    # Get mean firing rate for each STATE using model
    state_inds = np.hstack([np.zeros((tau_array.shape[0], 1)),
                            tau_array,
                            np.ones((tau_array.shape[0], 1))*spike_array.shape[-1]])
    state_lims = np.array([state_inds[:, x:x+2] for x in range(states)])
    state_lims = np.vectorize(np.int)(state_lims)
    state_lims = np.swapaxes(state_lims, 0, 1)

    state_firing = \
        np.array([[np.mean(trial_dat[:, start:end], axis=-1)
                   for start, end in trial_lims]
                  for trial_dat, trial_lims in zip(spike_array, state_lims)])

    state_firing = np.nan_to_num(state_firing)
    return state_firing


class _firing():
    """Helper class to handle processing for firing rate using "EphysData"
    """

    def __init__(self, tau_instance, processed_spikes, metadata):
        """Initialize firing class

        Args:
            tau_instance (Class): Tau class containing metadata and relevant variables
            processed_spikes (Numpy array): Numpy array containing processed spiking data
            metadata (Dict): Dict containing metadata on fit
        """
        self.tau = tau_instance
        self.processed_spikes = processed_spikes
        self.metadata = metadata
        self._EphysData = EphysData(self.metadata['data']['data_dir'])
        temp_spikes = self._EphysData.return_region_spikes(
            self.metadata['data']['region_name'])
        self.raw_spikes = temp_spikes[self.metadata['data']['taste_num']]
        self.state_firing = get_state_firing(self.processed_spikes,
                                             self.tau.raw_mode_tau)


class _tau():
    """Tau class to keep track of metadata and perform useful transformations
    """

    def __init__(self, tau_array, metadata):
        """Initialize tau class

        Args:
            tau_array ([type]): Array of samples from fitted model
            metadata (Dict): Dict containing metadata on fit
        """
        self.raw_tau = tau_array
        #self.metadata = metadata

        # def process_tau(self):
        #time_lims = self.metadata['preprocess']['time_lims']
        #bin_width = self.metadata['preprocess']['bin_width']
        time_lims = metadata['preprocess']['time_lims']
        bin_width = metadata['preprocess']['bin_width']

        self.raw_int_tau = np.vectorize(np.int)(self.raw_tau)
        self.raw_mode_tau = mode(self.raw_int_tau)[0][0]

        self.scaled_tau = (self.raw_tau * bin_width) + time_lims[0]
        self.scaled_int_tau = np.vectorize(np.int)(self.scaled_tau)
        self.scaled_mode_tau = np.squeeze(mode(self.scaled_int_tau)[0])


class PklHandler():
    """Helper class to handle metadata and fit data from pkl file
    """

    def __init__(self, file_path):
        """Initialize PklHandler class

        Args:
            file_path (str): Path to pkl file
        """
        self.dir_name = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        self.file_name_base = file_name.split('.')[0]
        self.pkl_file_path = \
            os.path.join(self.dir_name, self.file_name_base + ".pkl")
        with open(self.pkl_file_path, 'rb') as this_file:
            self.data = pkl.load(this_file)

        model_keys = ['model', 'approx', 'lambda', 'tau', 'data']
        key_savenames = ['_model_structure', '_fit_model',
                         'lambda_array', 'tau_array', 'processed_spikes']
        data_map = dict(zip(model_keys, key_savenames))

        for key, var_name in data_map.items():
            setattr(self, var_name, self.data['model_data'][key])

        self.metadata = self.data['metadata']
        self.pretty_metadata = pd.json_normalize(self.data['metadata']).T

        self.tau = _tau(self.tau_array, self.metadata)
        self.firing = _firing(self.tau, self.processed_spikes, self.metadata)


# ###########################################################################
# ## LOAD DATA
# ###########################################################################

# FIT_PKL = '/media/bigdata/firing_space_plot/changepoint_mcmc/'\
#         'saved_models/natasha_gc_binary/natasha_gc_binary_0296f33c.info'

# model_dat = PklHandler(FIT_PKL)
# print(model_dat.tau.scaled_mode_tau.shape)
# print(model_dat.firing.state_firing.shape)
