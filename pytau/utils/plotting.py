"""
Plotting functionality for PyTau output.
"""

import matplotlib.pyplot as plt
import numpy as np

from ..changepoint_analysis import (
    calc_significant_neurons_firing,
    calc_significant_neurons_snippets,
    get_state_firing,
    get_transition_snips,
)


def plot_elbo_history(fit_model, final_window=0.05):
    """
    Plot the ELBO history of a fitted model.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(-fit_model.hist, alpha=0.3)
    ax[0].set_ylabel("ELBO")
    ax[0].set_xlabel("Iteration")

    n_fit = len(fit_model.hist)
    ind = int(n_fit - n_fit * 0.05)
    last_iters = fit_model.hist[ind:]
    ax[1].plot(-last_iters, alpha=0.3)
    x = np.arange(len(last_iters))
    binned_iters = np.reshape(last_iters, (-1, 100)).mean(axis=-1)
    binned_x = x[::100]
    ax[1].plot(binned_x, -binned_iters)
    ax[1].set_title("Final 5% of iterations")
    ax[1].set_ylabel("ELBO")
    ax[1].set_xlabel("iteration")

    return fig, ax


def raster(spike_array, ax=None, color="k", alpha=0.5):
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
    ax.scatter(*spike_inds[::-1], marker="|", color=color, alpha=alpha)
    ax.set_xlim([0, n_bins])
    ax.set_ylim([-0.5, n_neurons - 0.5])
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
    cmap = plt.cm.get_cmap("tab10")
    n_trials, n_neurons, n_bins = spike_array.shape
    n_states = tau.shape[1] + 1
    fig, ax = plt.subplots(n_trials, 1, figsize=(
        15, 5 * n_trials), sharex=True, sharey=True)
    for trial in range(n_trials):
        raster(spike_array[trial], ax=ax[trial])
        if trial == n_trials - 1:
            ax[trial].set_xlabel("Time")
            ax[trial].set_ylabel("Nrn #")
        if plot_lims is not None:
            ax[trial].set_xlim(plot_lims)
            ax[trial].text(plot_lims[1] + 10, 0.5 * n_neurons,
                           "Trial {}".format(trial))
        else:
            ax[trial].text(n_bins + 10, 0.5 * n_neurons,
                           "Trial {}".format(trial))
        for change in range(n_states - 1):
            ax[trial].axvline(tau[trial, change], color="r",
                              linestyle="--", linewidth=2)
        for state in range(n_states):
            if state == 0:
                ax[trial].axvspan(0, tau[trial, state],
                                  color=cmap(state), alpha=0.2)
            elif state == n_states - 1:
                ax[trial].axvspan(tau[trial, state - 1], n_bins,
                                  color=cmap(state), alpha=0.2)
            else:
                ax[trial].axvspan(
                    tau[trial, state - 1],
                    tau[trial, state],
                    color=cmap(state),
                    alpha=0.2,
                )
    return fig, ax


def plot_changepoint_overview(tau, plot_lims):
    """
    tau: array of shape (n_trials, n_transitions) with changepoint values
    2 rows of subplots:
        1. Convert changepoints to state durations and plot heatmap of state durations
        2. Histogram of changepoint values for each changepoint
    """
    n_trials, n_transitions = tau.shape
    n_states = n_transitions + 1
    state_durations = np.zeros((n_trials, plot_lims[1]))
    for trial in range(n_trials):
        for state in range(n_states):
            if state == 0:
                state_durations[trial, plot_lims[0]: tau[trial, state]] = state
            elif state == n_states - 1:
                state_durations[trial, tau[trial, state - 1]
                    : plot_lims[1]] = state
            else:
                state_durations[trial, tau[trial, state - 1]
                    : tau[trial, state]] = state

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    ax[0].pcolormesh(np.arange(plot_lims[1] + 1),
                     np.arange(n_trials + 1), state_durations)
    ax[0].set_xlim(plot_lims)
    ax[0].set_ylabel("Trial")
    ax[0].set_xlabel("State")
    ax[0].set_title("State durations")

    for transition in range(n_transitions):
        ax[1].hist(
            tau[:, transition],
            bins=10,
            alpha=0.5,
            label="Transition {}".format(transition),
        )
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Count")
    ax[1].legend()
    return fig, ax


def plot_aligned_state_firing(spike_array, tau, window_radius=300):
    """
    spike_array : array of shape (n_trials, n_neurons, n_bins)
    tau : array of shape (n_trials, n_transitions) with changepoint values

    Plot the average firing rate for each neuron across trials for 'window_radius'
    around each changepoint.
    Highlight significant neurons with yellow patch
    Output figure with shape 1 x n_states
    """
    n_trials, n_neurons, n_bins = spike_array.shape
    n_transitions = tau.shape[1]
    transitions_firing = get_transition_snips(spike_array, tau, window_radius)
    p_val_array, significant_neurons = calc_significant_neurons_snippets(
        transitions_firing)
    fig, ax = plt.subplots(1, n_transitions, figsize=(
        15, 5), sharex=True, sharey=True)
    for transitions in range(n_transitions):
        raster(transitions_firing[..., transitions].sum(
            axis=0), ax=ax[transitions])
        ax[transitions].set_title("Transition {}".format(transitions))
        ax[transitions].set_xlabel("Time")
        ax[transitions].set_ylabel("Neuron")
        ax[transitions].axvline(window_radius, color="r")
        for neuron in range(n_neurons):
            if significant_neurons[neuron, transitions]:
                ax[transitions].axhspan(
                    neuron - 0.5, neuron + 0.5, color="y", alpha=0.5, zorder=10)
    return fig, ax


def plot_state_firing_rates(spike_array, tau):
    """Stacked subplots of bar plots with error bars and significance markers
    for each neuron per state
    Also highlight significant neurons with yellow patch

    spike_array : array of shape (n_trials, n_neurons, n_bins)
    tau : array of shape (n_trials, n_transitions) with changepoint values
    """
    # shape (n_trials, n_states, n_neurons)
    state_firing_rates = get_state_firing(spike_array, tau)
    # shape (n_states, n_neurons)
    mean_state_firing, std_state_firing = state_firing_rates.mean(axis=0), state_firing_rates.std(
        axis=0
    )
    p_val_array, significant_neurons = calc_significant_neurons_firing(
        state_firing_rates)
    n_trials, n_states, n_neurons = state_firing_rates.shape
    fig, ax = plt.subplots(n_states, 1, figsize=(
        10, 3 * n_states), sharex=True, sharey=True)
    for state in range(n_states):
        ax[state].bar(
            np.arange(n_neurons),
            mean_state_firing[state],
            yerr=std_state_firing[state],
            capsize=5,
        )
        ax[state].set_ylabel("Firing rate")
        # Print state name on the right of each subplot (rotated vertically)
        ax[state].text(
            1.02,
            0.5,
            "State {}".format(state),
            rotation=270,
            verticalalignment="center",
            transform=ax[state].transAxes,
        )
        # ax[state].set_title('State {}'.format(state))
        ax[state].set_xticks(np.arange(n_neurons))
        ax[state].set_xticklabels(np.arange(n_neurons))
        for neuron in range(n_neurons):
            if neuron in significant_neurons:
                upper_lim = mean_state_firing[state,
                                              neuron] + std_state_firing[state, neuron]
                # ax[state].text(neuron, 1.1*upper_lim,
                #               '*', horizontalalignment='center')
                ax[state].axvspan(neuron - 0.5, neuron + 0.5,
                                  color="y", alpha=0.5, zorder=10)
    ax[n_states - 1].set_xlabel("Neuron")
    fig.suptitle(
        "Firing rate per state" + "\n" +
        "Highlight = Significance (with Bonf Correction, p < 0.05)"
    )
    plt.tight_layout()
    return fig, ax
