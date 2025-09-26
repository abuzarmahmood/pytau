"""
Helper classes and functions to perform analysis on fitted models
"""

import os
import pickle as pkl

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, mode, ttest_rel

from .utils import EphysData


def get_transition_snips(spike_array, tau_array, window_radius=300):
    """Get snippets of activty around changepoints for each trial

    Args:
        spike_array (3D Numpy array): trials x nrns x bins
        tau_array (2D Numpy array): trials x switchpoints

    Returns:
        Numpy array: Transition snippets : trials x nrns x bins x transitions

    Make sure none of the snippets are outside the bounds of the data
    """
    # Get snippets of activity around changepoints for each trial
    n_trials, n_neurons, n_bins = spike_array.shape
    n_transitions = tau_array.shape[1]
    transition_snips = np.zeros(
        (n_trials, n_neurons, 2 * window_radius, n_transitions))
    window_lims = np.stack(
        [tau_array - window_radius, tau_array + window_radius], axis=-1)

    # Make sure no lims are outside the bounds of the data
    if (window_lims < 0).sum(axis=None) or (window_lims > n_bins).sum(axis=None):
        raise ValueError("Transition window extends outside data bounds")

    # Pull out snippets
    for trial in range(n_trials):
        for transition in range(n_transitions):
            transition_snips[trial, :, :, transition] = spike_array[
                trial,
                :,
                window_lims[trial, transition, 0]: window_lims[trial, transition, 1],
            ]
    return transition_snips


def get_state_firing(spike_array, tau_array):
    """Calculate firing rates within states given changepoint positions on data

    Args:
        spike_array (3D Numpy array): trials x nrns x bins
        tau_array (2D Numpy array): trials x switchpoints

    Returns:
        state_firing (3D Numpy array): trials x states x nrns
    """

    states = tau_array.shape[-1] + 1
    # Get mean firing rate for each STATE using model
    state_inds = np.hstack(
        [
            np.zeros((tau_array.shape[0], 1)),
            tau_array,
            np.ones((tau_array.shape[0], 1)) * spike_array.shape[-1],
        ]
    )
    state_lims = np.array([state_inds[:, x: x + 2] for x in range(states)])
    state_lims = np.vectorize(int)(state_lims)
    state_lims = np.swapaxes(state_lims, 0, 1)

    state_firing = np.array(
        [
            [np.mean(trial_dat[:, start:end], axis=-1)
             for start, end in trial_lims]
            for trial_dat, trial_lims in zip(spike_array, state_lims)
        ]
    )

    state_firing = np.nan_to_num(state_firing)
    return state_firing


def calc_significant_neurons_firing(state_firing, p_val=0.05):
    """Calculate significant changes in firing rate between states
    Iterate ANOVA over neurons for all states
    With Bonferroni correction

    Args
        state_firing (3D Numpy array): trials x states x nrns
        p_val (float, optional): p-value to use for significance. Defaults to 0.05.

    Returns:
        anova_p_val_array (1D Numpy array): p-values for each neuron
        anova_sig_neurons (1D Numpy array): indices of significant neurons
    """
    n_neurons = state_firing.shape[-1]
    # Calculate ANOVA p-values for each neuron
    anova_p_val_array = np.zeros(state_firing.shape[-1])
    for neuron in range(state_firing.shape[-1]):
        anova_p_val_array[neuron] = f_oneway(*state_firing[:, :, neuron].T)[1]
    anova_sig_neurons = np.where(anova_p_val_array < p_val / n_neurons)[0]

    return anova_p_val_array, anova_sig_neurons


def calc_significant_neurons_snippets(transition_snips, p_val=0.05):
    """Calculate pairwise t-tests to detect differences between each transition
    With Bonferroni correction

    Args
        transition_snips (4D Numpy array): trials x nrns x bins x transitions
        p_val (float, optional): p-value to use for significance. Defaults to 0.05.

    Returns:
        anova_p_val_array (neurons, transition): p-values for each neuron
        anova_sig_neurons (neurons, transition): indices of significant neurons
    """
    # Calculate pairwise t-tests for each transition
    # shape : [before, after] x trials x neurons x transitions
    mean_transition_snips = np.stack(np.array_split(
        transition_snips, 2, axis=2)).mean(axis=3)
    pairwise_p_val_array = np.zeros(mean_transition_snips.shape[2:])
    n_neuron, n_transitions = pairwise_p_val_array.shape
    for neuron in range(n_neuron):
        for transition in range(n_transitions):
            pairwise_p_val_array[neuron, transition] = ttest_rel(
                *mean_transition_snips[:, :, neuron, transition]
            )[1]
    pairwise_sig_neurons = pairwise_p_val_array < p_val  # /n_neuron
    return pairwise_p_val_array, pairwise_sig_neurons


class _firing:
    """Helper class to handle processing for firing rate using "EphysData" """

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
        self._EphysData = EphysData(self.metadata["data"]["data_dir"])
        temp_spikes = self._EphysData.return_region_spikes(
            self.metadata["data"]["region_name"])
        taste_num = self.metadata["data"]["taste_num"]
        if taste_num != "all":
            self.raw_spikes = temp_spikes[taste_num]
        else:
            self.raw_spikes = temp_spikes
        self.state_firing = get_state_firing(
            self.processed_spikes, self.tau.raw_mode_tau)
        self.transition_snips = get_transition_snips(
            self.raw_spikes, self.tau.scaled_mode_tau)
        (
            self.anova_p_val_array,
            self.anova_significant_neurons,
        ) = calc_significant_neurons_firing(self.state_firing)
        (
            self.pairwise_p_val_array,
            self.pairwise_significant_neurons,
        ) = calc_significant_neurons_snippets(self.transition_snips)


class _tau:
    """Tau class to keep track of metadata and perform useful transformations"""

    def __init__(self, tau_array, metadata, n_trials=None):
        """Initialize tau class

        Args:
            tau_array ([type]): Array of samples from fitted model
            metadata (Dict): Dict containing metadata on fit
        """
        self.raw_tau = tau_array

        time_lims = metadata["preprocess"]["time_lims"]
        bin_width = metadata["preprocess"]["bin_width"]

        self.raw_int_tau = np.vectorize(int)(self.raw_tau)
        self.raw_mode_tau = mode(self.raw_int_tau)[0][0]

        self.scaled_tau = (self.raw_tau * bin_width) + time_lims[0]
        self.scaled_int_tau = np.vectorize(int)(self.scaled_tau)
        mode_result = np.squeeze(mode(self.scaled_int_tau)[0])
        # Ensure mode_result is 1D array of changepoints
        if mode_result.ndim == 0:
            mode_result = np.array([mode_result])
        
        # If n_trials is provided, replicate changepoints for each trial
        # This is needed for plotting functions that expect (n_trials, n_changepoints)
        if n_trials is not None:
            self.scaled_mode_tau = np.tile(mode_result, (n_trials, 1))
        else:
            self.scaled_mode_tau = mode_result


class PklHandler:
    """Helper class to handle metadata and fit data from pkl file"""

    def __init__(self, file_path):
        """Initialize PklHandler class

        Args:
            file_path (str): Path to pkl file
        """
        self.dir_name = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        self.file_name_base = file_name.split(".")[0]
        self.pkl_file_path = os.path.join(
            self.dir_name, self.file_name_base + ".pkl")
        with open(self.pkl_file_path, "rb") as this_file:
            self.data = pkl.load(this_file)

        model_keys = ["model", "approx", "lambda", "tau", "data"]
        key_savenames = [
            "_model_structure",
            "_fit_model",
            "lambda_array",
            "tau_array",
            "processed_spikes",
        ]
        data_map = dict(zip(model_keys, key_savenames))

        for key, var_name in data_map.items():
            setattr(self, var_name, self.data["model_data"][key])

        self.metadata = self.data["metadata"]
        self.pretty_metadata = pd.json_normalize(self.data["metadata"]).T

        # Get number of trials from processed_spikes for proper tau formatting
        n_trials = self.processed_spikes.shape[0] if hasattr(self.processed_spikes, 'shape') else None
        self.tau = _tau(self.tau_array, self.metadata, n_trials)
        self.firing = _firing(self.tau, self.processed_spikes, self.metadata)
