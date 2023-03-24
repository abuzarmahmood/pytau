"""
Plotting functionality for PyTau output.
"""

import matplotlib.pyplot as plt
import numpy as np
from ..changepoint_analysis import get_transition_snips


def plot_elbo_history(fit_model, n_fit, final_window=0.05):
    """
    Plot the ELBO history of a fitted model.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(-fit_model.hist, alpha=.3)
    ax[0].set_ylabel('ELBO')
    ax[0].set_xlabel('Iteration');

    ind = int(n_fit - n_fit*0.05)
    last_iters = fit_model.hist[ind:]
    ax[1].plot(-last_iters, alpha=.3)
    x = np.arange(len(last_iters))
    binned_iters = np.reshape(last_iters, (-1, 100)).mean(axis=-1)
    binned_x = x[::100]
    ax[1].plot(binned_x, -binned_iters)
    ax[1].set_title('Final 5% of iterations')
    ax[1].set_ylabel('ELBO')
    ax[1].set_xlabel('iteration');

    return fig, ax


def raster(spike_array, ax=None, color='k', alpha=0.5):
    """
    spike_array : array of shape (n_neurons, n_bins)
    ax : axis to plot on
    color : color of the spikes
    alpha : alpha value of the spikes

    Plot spikes using scatter plot
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    else:
        fig = None
    n_neurons, n_bins = spike_array.shape
    spike_inds = np.where(spike_array)
    ax.scatter(*spike_inds[::-1], marker='|', color=color, alpha=alpha)
    ax.set_xlim([0, n_bins])
    ax.set_ylim([0, n_neurons])
    return fig, ax

def plot_changepoint_raster(spike_array, tau, plot_lims=None):
    """
    spike_array : array of shape (n_trials, n_neurons, n_bins)
    tau : array of shape (n_trials, n_transitions) with changepoint values

    Plot a raster for all neurons with changepoint lines for every trial
    and colored patched indicating duration of each state.

    Output figure with number of axes = n_trials
    Plot spikes using scatter plot
    Print trial number next to each raster
    """
    cmap = plt.cm.get_cmap('tab10')
    n_trials, n_neurons, n_bins=spike_array.shape
    n_states=tau.shape[1] + 1
    fig, ax=plt.subplots(n_trials, 1, figsize=(15, 5*n_trials),
                         sharex=True, sharey=True)
    for trial in range(n_trials):
        raster(spike_array[trial], ax=ax[trial])
        if trial == n_trials-1:
            ax[trial].set_xlabel('Time')
            ax[trial].set_ylabel('Nrn #')
        if plot_lims is not None:
            ax[trial].set_xlim(plot_lims)
            ax[trial].text(plot_lims[1]+10, 0.5*n_neurons, 'Trial {}'.format(trial))
        else:
            ax[trial].text(n_bins+10, 0.5*n_neurons, 'Trial {}'.format(trial))
        for change in range(n_states-1): 
            ax[trial].axvline(tau[trial, change], color='r',
                              linestyle='--', linewidth=2)
        for state in range(n_states):
            if state == 0:
                ax[trial].axvspan(0, tau[trial, state], 
                                  color=cmap(state), alpha=0.2)
            elif state == n_states - 1:
                ax[trial].axvspan(tau[trial, state-1], 
                                  n_bins, color=cmap(state), alpha=0.2)
            else:
                ax[trial].axvspan(tau[trial, state-1], tau[trial, state], 
                                  color=cmap(state), alpha=0.2)
    return fig, ax

def plot_changepoint_overview(tau, plot_lims):
    """
    tau: array of shape (n_trials, n_transitions) with changepoint values
    2 rows of subplots:
        1. Convert changepoints to state durations and plot heatmap of state durations
        2. Histogram of changepoint values for each changepoint
    """
    n_trials, n_states = tau.shape
    state_durations = np.zeros((n_trials, plot_lims[1])) 
    for trial in range(n_trials):
        for state in range(n_states):
            if state == 0:
                state_durations[trial, plot_lims[0]:tau[trial, state]] = state
            elif state == n_states - 1:
                state_durations[trial, tau[trial, state-1]:plot_lims[1]] = state
            else:
                state_durations[trial, tau[trial, state-1]:tau[trial, state]] = state

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    ax[0].pcolormesh(np.arange(plot_lims[1]), np.arange(n_trials), state_durations)
    ax[0].set_xlim(plot_lims)
    ax[0].set_ylabel('Trial')
    ax[0].set_xlabel('State')
    ax[0].set_title('State durations')

    for state in range(n_states):
        ax[1].hist(tau[:, state], bins=10, alpha=0.5, label='State {}'.format(state))
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Count')
    ax[1].legend()
    return fig, ax
    

def plot_aligned_state_firing(spike_array, tau, window_radius=300):
    """
    spike_array : array of shape (n_trials, n_neurons, n_bins)
    tau : array of shape (n_trials, n_transitions) with changepoint values

    Plot the average firing rate for each neuron across trials for 'window_radius'
    around each changepoint.
    Output figure with shape 1 x n_states
    """
    n_trials, n_neurons, n_bins=spike_array.shape
    n_transitions=tau.shape[1] 
    transitions_firing=get_transition_snips(spike_array, tau, window_radius)
    fig, ax=plt.subplots(1, n_transitions, figsize=(15, 5),
                         sharex=True, sharey=True)
    for transitions in range(n_transitions):
        raster(transitions_firing[...,transitions].sum(axis=0), ax=ax[transitions])
        ax[transitions].set_title('Transition {}'.format(transitions))
        ax[transitions].set_xlabel('Time')
        ax[transitions].set_ylabel('Neuron')
        ax[transitions].axvline(window_radius, color='r')
    return fig, ax
