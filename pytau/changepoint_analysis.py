"""
Helper classes and functions to perform analysis on fitted models
"""

import os

import cloudpickle as pkl
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chisquare, entropy, f_oneway, mode, ttest_rel

from pytau.utils import EphysData


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


def get_state_snippets(spike_array, tau_array):
    """Extract neural activity snippets for each state and trial without averaging

    Returns raw neural activity for each state as a ragged array structure,
    where each state can have different durations across trials.

    Args:
        spike_array (np.ndarray): Neural activity data
            Shape: (n_trials, n_neurons, n_bins)
        tau_array (np.ndarray): Changepoint positions for each trial
            Shape: (n_trials, n_changepoints)

    Returns:
        list: Nested list structure organized as [state][trial]
            - Outer list length: n_states (n_changepoints + 1)
            - Inner list length: n_trials
            - Each element shape: (n_neurons, bins_in_state)
            Note: bins_in_state varies by trial and state

    Example:
        >>> spike_array.shape  # (5 trials, 3 neurons, 100 bins)
        (5, 3, 100)
        >>> tau_array.shape  # (5 trials, 2 changepoints)
        (5, 2)
        >>> snippets = get_state_snippets(spike_array, tau_array)
        >>> len(snippets)  # 3 states (2 changepoints + 1)
        3
        >>> len(snippets[0])  # 5 trials
        5
        >>> snippets[0][0].shape  # First trial, first state
        (3, 30)  # 3 neurons, 30 bins (from start to first changepoint)
    """
    states = tau_array.shape[-1] + 1
    # Get state boundaries for each trial
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

    # Extract snippets for each state and trial
    state_snippets = []
    for state_idx in range(states):
        trial_snippets = []
        for trial_idx, (trial_dat, trial_lims) in enumerate(zip(spike_array, state_lims)):
            start, end = trial_lims[state_idx]
            trial_snippets.append(trial_dat[:, start:end])
        state_snippets.append(trial_snippets)

    return state_snippets


def get_state_firing(spike_array, tau_array):
    """Calculate mean firing rates within states given changepoint positions

    Computes average neural activity for each state by calling get_state_snippets
    and averaging over time bins within each state.

    Args:
        spike_array (np.ndarray): Neural activity data
            Shape: (n_trials, n_neurons, n_bins)
        tau_array (np.ndarray): Changepoint positions for each trial
            Shape: (n_trials, n_changepoints)

    Returns:
        np.ndarray: Mean firing rates per state
            Shape: (n_trials, n_states, n_neurons)
            where n_states = n_changepoints + 1
            NaN values are replaced with 0

    Example:
        >>> spike_array.shape
        (5, 3, 100)  # 5 trials, 3 neurons, 100 bins
        >>> tau_array.shape
        (5, 2)  # 5 trials, 2 changepoints
        >>> firing = get_state_firing(spike_array, tau_array)
        >>> firing.shape
        (5, 3, 3)  # 5 trials, 3 states, 3 neurons
    """
    # Get state snippets
    state_snippets = get_state_snippets(spike_array, tau_array)

    # Calculate mean firing rate for each state and trial
    state_firing = np.array(
        [
            [np.mean(snippet, axis=-1) for snippet in trial_snippets]
            for trial_snippets in zip(*state_snippets)
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


##############################
# Data-quality diagnostics
##############################

def calc_firing_drift_anova(spike_array, n_trial_bins=4, p_val=0.05, chance_p_val=0.05):
    """Detect session-level drift by checking whether more neurons than
    expected by chance show a significant firing-rate change across the
    recording session (issue #37).

    Method:
        1. Split trials, in recording order, into `n_trial_bins` consecutive,
           roughly-equal blocks (a proxy for elapsed session time).
        2. For each neuron, compute per-trial mean firing rate (mean over
           time bins) and run a one-way ANOVA across the blocks.
        3. Count neurons individually significant at `p_val` (uncorrected --
           we want the raw hit-rate, not per-neuron confidence).
        4. Test whether that count exceeds a Binomial(n_neurons, p_val) null
           via a one-sided binomial test.

    Args:
        spike_array (3D Numpy array): trials x neurons x bins, in the order
            trials were recorded (e.g. processed_spikes).
        n_trial_bins (int, optional): number of consecutive trial-order
            blocks to compare. Defaults to 4.
        p_val (float, optional): per-neuron ANOVA significance threshold
            (uncorrected). Defaults to 0.05.
        chance_p_val (float, optional): significance threshold for the
            population-level binomial test. Defaults to 0.05.

    Returns:
        neuron_p_val_array (1D Numpy array, len=n_neurons): per-neuron ANOVA
            p-value; np.nan where a neuron/block had too few trials (<2) to
            test.
        drift_neurons (1D Numpy array): indices significant at p_val.
        population_p_val (float): binomial-test p-value that
            len(drift_neurons) is higher than chance. np.nan if no neuron
            could be tested.
        is_drift_detected (bool): population_p_val < chance_p_val.
    """
    n_trials, n_neurons, _ = spike_array.shape
    trial_rate = spike_array.mean(axis=-1)  # trials x neurons
    trial_blocks = [b for b in np.array_split(
        np.arange(n_trials), n_trial_bins) if len(b)]

    neuron_p_val_array = np.full(n_neurons, np.nan)
    for neuron in range(n_neurons):
        groups = [trial_rate[block, neuron] for block in trial_blocks]
        if len(groups) < 2 or any(len(g) < 2 for g in groups):
            continue
        neuron_p_val_array[neuron] = f_oneway(*groups)[1]

    valid = ~np.isnan(neuron_p_val_array)
    drift_neurons = np.where(valid & (neuron_p_val_array < p_val))[0]

    if valid.sum() == 0:
        return neuron_p_val_array, drift_neurons, np.nan, False

    population_p_val = binomtest(
        len(drift_neurons), int(valid.sum()), p_val, alternative="greater"
    ).pvalue
    is_drift_detected = bool(population_p_val < chance_p_val)
    return neuron_p_val_array, drift_neurons, population_p_val, is_drift_detected


def calc_state_trial_uniformity(tau_array, n_bins, n_trial_blocks=4,
                                min_state_frac=0.05, p_val=0.05):
    """Detect states whose presence is confined to a specific subset of
    trials (in recording order), which suggests the "state" is capturing a
    session-level drift event rather than a real, recurring behavioral
    state (issue #37).

    Method: for each state (interval between changepoints), mark it
    "present" in a trial if its duration exceeds `min_state_frac * n_bins`.
    Bin trials (in recording order) into `n_trial_blocks` blocks and
    chi-squared-test the observed per-block presence counts against a
    uniform-across-blocks null.

    Args:
        tau_array (2D Numpy array): trials x changepoints, in recording
            order (e.g. tau.raw_mode_tau).
        n_bins (int): total number of time bins per trial (upper bound of
            the state-duration range).
        n_trial_blocks (int, optional): number of trial-order blocks.
            Defaults to 4.
        min_state_frac (float, optional): minimum fraction of n_bins for a
            state to count as "present" in a trial. Defaults to 0.05.
        p_val (float, optional): significance threshold for the
            chi-squared test. Defaults to 0.05.

    Returns:
        state_p_val_array (1D Numpy array, len=n_states): chi-squared
            p-value per state; np.nan where too few "present" trials exist
            for a reliable test (any expected per-block count < 1) or where
            the state is present in 0 or all trials.
        nonuniform_states (1D Numpy array): indices of states with
            state_p_val_array < p_val -> suggests drift.
    """
    tau_array = np.asarray(tau_array)
    n_trials, n_changepoints = tau_array.shape
    n_states = n_changepoints + 1

    bounds = np.hstack([
        np.zeros((n_trials, 1)), tau_array, np.full((n_trials, 1), n_bins)
    ])
    durations = np.diff(bounds, axis=1)  # trials x states
    presence = durations > (min_state_frac * n_bins)

    trial_blocks = [b for b in np.array_split(
        np.arange(n_trials), n_trial_blocks) if len(b)]
    block_sizes = np.array([len(b) for b in trial_blocks])

    state_p_val_array = np.full(n_states, np.nan)
    for state in range(n_states):
        n_present = presence[:, state].sum()
        if n_present == 0 or n_present == n_trials:
            continue  # nothing to test / not this check's failure mode
        observed = np.array([presence[block, state].sum()
                             for block in trial_blocks])
        expected = n_present * (block_sizes / n_trials)
        if np.any(expected < 1):
            continue  # too few trials for a valid chi-squared test
        state_p_val_array[state] = chisquare(
            f_obs=observed, f_exp=expected)[1]

    valid = ~np.isnan(state_p_val_array)
    nonuniform_states = np.where(valid & (state_p_val_array < p_val))[0]
    return state_p_val_array, nonuniform_states


def calc_collapsed_transitions(tau_array, n_bins, edge_frac=0.05, gap_frac=0.05,
                               collapse_frac_thresh=0.1):
    """Detect changepoints collapsed to the edges of the trial window, or
    adjacent changepoints collapsed ("merged") together (issue #38).

    Args:
        tau_array (2D Numpy array): trials x changepoints (e.g.
            tau.raw_mode_tau).
        n_bins (int): total number of time bins per trial.
        edge_frac (float, optional): a changepoint within edge_frac * n_bins
            of 0 or n_bins is considered collapsed to that edge. Defaults
            to 0.05.
        gap_frac (float, optional): two (sorted) adjacent changepoints
            closer than gap_frac * n_bins are considered merged. Defaults
            to 0.05.
        collapse_frac_thresh (float, optional): fraction of trials that must
            show a collapse before a dataset-level warning string is
            produced. Defaults to 0.1.

    Returns:
        edge_collapse_mask (2D bool array, trials x changepoints): True
            where that changepoint sits in the edge zone.
        merged_mask (2D bool array, trials x (changepoints-1)): True where
            adjacent (sorted) changepoints are closer than gap_frac * n_bins.
            Empty along axis 1 if there is only one changepoint.
        warnings_list (list of str): populated only if the fraction of
            affected trials exceeds collapse_frac_thresh for either check.
    """
    tau_array = np.asarray(tau_array)
    n_trials, n_changepoints = tau_array.shape
    edge_buffer = edge_frac * n_bins
    gap_buffer = gap_frac * n_bins

    edge_collapse_mask = (tau_array < edge_buffer) | (
        tau_array > (n_bins - edge_buffer))

    if n_changepoints > 1:
        sorted_tau = np.sort(tau_array, axis=1)
        merged_mask = np.diff(sorted_tau, axis=1) < gap_buffer
    else:
        merged_mask = np.zeros((n_trials, 0), dtype=bool)

    warnings_list = []
    edge_frac_trials = edge_collapse_mask.any(axis=1).mean()
    if edge_frac_trials > collapse_frac_thresh:
        warnings_list.append(
            f"{edge_frac_trials:.1%} of trials have >=1 changepoint collapsed "
            f"to an edge (within {edge_frac:.0%} of the trial window)."
        )
    if merged_mask.size:
        merge_frac_trials = merged_mask.any(axis=1).mean()
        if merge_frac_trials > collapse_frac_thresh:
            warnings_list.append(
                f"{merge_frac_trials:.1%} of trials have >=2 changepoints "
                f"merged together (within {gap_frac:.0%} of the trial window)."
            )
    return edge_collapse_mask, merged_mask, warnings_list


def calc_transition_randomness(tau_array, n_bins, n_hist_bins=10, method="chisquare",
                               uniform_p_val=0.05, entropy_ratio_thresh=0.9):
    """Test whether tau values for each transition look like they were
    drawn from Uniform(0, n_bins), which would suggest the model isn't
    finding a consistent transition but scattering changepoints randomly
    across trials (issue #38).

    Note on directionality: the null hypothesis here IS uniformity, so for
    method='chisquare' a transition is flagged as "random" when we FAIL to
    reject the null, i.e. when p_val > uniform_p_val (the opposite
    direction from typical significance testing elsewhere in this module).

    Args:
        tau_array (2D Numpy array): trials x changepoints.
        n_bins (int): total number of time bins per trial (defines the
            [0, n_bins] range of the uniform null).
        n_hist_bins (int, optional): number of histogram bins used to
            discretize tau values per changepoint. Defaults to 10. Caller
            should ensure n_trials is comfortably larger than n_hist_bins
            for a valid chi-squared test; changepoints failing this return
            np.nan.
        method (str, optional): {'chisquare', 'entropy'}. Defaults to
            'chisquare' (maintainer's stated preference: "more simply").
        uniform_p_val (float, optional): threshold for method='chisquare'.
            Defaults to 0.05.
        entropy_ratio_thresh (float, optional): threshold for
            method='entropy'; ratio of observed Shannon entropy to the
            maximum entropy of a uniform distribution over n_hist_bins
            bins. Values close to 1 indicate near-uniform/random
            transitions. Defaults to 0.9.

    Returns:
        stat_array (1D Numpy array, len=n_changepoints): chi-squared
            p-values (method='chisquare') or entropy ratios
            (method='entropy'). np.nan for changepoints that couldn't be
            reliably tested.
        random_transitions (1D Numpy array): indices flagged as
            random/uninformative.
    """
    if method not in ("chisquare", "entropy"):
        raise ValueError("method must be one of {'chisquare', 'entropy'}")

    tau_array = np.asarray(tau_array)
    n_changepoints = tau_array.shape[1]
    bin_edges = np.linspace(0, n_bins, n_hist_bins + 1)

    stat_array = np.full(n_changepoints, np.nan)
    for c in range(n_changepoints):
        obs_counts, _ = np.histogram(tau_array[:, c], bins=bin_edges)
        total = obs_counts.sum()
        if total == 0:
            continue
        if method == "chisquare":
            expected = np.full(n_hist_bins, total / n_hist_bins)
            if np.any(expected < 1):
                continue
            stat_array[c] = chisquare(f_obs=obs_counts, f_exp=expected)[1]
        else:  # entropy
            probs = obs_counts[obs_counts > 0] / total
            stat_array[c] = entropy(probs) / np.log(n_hist_bins)

    valid = ~np.isnan(stat_array)
    if method == "chisquare":
        random_transitions = np.where(valid & (stat_array > uniform_p_val))[0]
    else:
        random_transitions = np.where(
            valid & (stat_array > entropy_ratio_thresh))[0]
    return stat_array, random_transitions


def get_time_binned_firing(spike_array, n_bins=10):
    """Bin the time axis of a spike array into n_bins equal consecutive
    chunks and sum spike counts within each chunk.

    Produces the same (trials, groups, neurons) shape convention as
    get_state_firing, but with "groups" = arbitrary equal time bins instead
    of model-fit states -- for use *before* a changepoint model has been
    fit (see calc_dynamic_neurons, issue #39).

    Args:
        spike_array (3D Numpy array): trials x neurons x bins.
        n_bins (int, optional): number of equal time chunks. Defaults to 10.

    Returns:
        binned_firing (3D Numpy array): trials x n_bins x neurons.

    Raises:
        ValueError: if spike_array.shape[-1] is not evenly divisible by
            n_bins.
    """
    n_trials, n_neurons, n_time = spike_array.shape
    if n_time % n_bins != 0:
        raise ValueError(
            f"n_bins ({n_bins}) must evenly divide the number of time bins "
            f"in spike_array ({n_time})"
        )
    binned = spike_array.reshape(n_trials, n_neurons, n_bins, -1).sum(axis=-1)
    return np.moveaxis(binned, 1, 2)  # trials x n_bins x neurons


def calc_dynamic_neurons(spike_array, n_bins=10, p_val=0.05, dynamic_p_val=0.05):
    """Select neurons whose firing rate changes across time within a trial
    ("dynamic" neurons), for use as a pre-fit screen -- e.g. deciding which
    neurons to include when fitting a changepoint model (issue #39).

    Design note: bins the raw spike array over arbitrary time chunks
    (get_time_binned_firing) and reuses the existing, unmodified
    calc_significant_neurons_firing as the statistical engine, rather than
    running the ANOVA over already-fit state_firing -- the latter would be
    circular for this issue's stated purpose (selecting neurons *before*
    fitting a model).

    Args:
        spike_array (3D Numpy array): trials x neurons x bins, raw or
            preprocessed spike counts (e.g. processed_spikes).
        n_bins (int, optional): number of equal time bins for the ANOVA
            groups. Defaults to 10.
        p_val (float, optional): Bonferroni-corrected p-value for the
            strict "significant" set (same semantics as
            calc_significant_neurons_firing). Defaults to 0.05.
        dynamic_p_val (float, optional): uncorrected, more lenient p-value
            defining the broader, practical "fit on these neurons" set.
            Defaults to 0.05.

    Returns:
        anova_p_val_array (1D Numpy array, len=n_neurons): per-neuron ANOVA
            p-value across the n_bins time bins.
        anova_sig_neurons (1D Numpy array): Bonferroni-significant indices.
        dynamic_neurons (1D Numpy array): indices with
            anova_p_val_array < dynamic_p_val.
    """
    binned_firing = get_time_binned_firing(spike_array, n_bins=n_bins)
    anova_p_val_array, anova_sig_neurons = calc_significant_neurons_firing(
        binned_firing, p_val=p_val
    )
    dynamic_neurons = np.where(anova_p_val_array < dynamic_p_val)[0]
    return anova_p_val_array, anova_sig_neurons, dynamic_neurons


def get_fit_diagnostics(spike_array, tau_array, n_bins, **kwargs):
    """Run all post-fit data-quality checks (#37 drift, #38 collapsed/random
    transitions) and return both the raw per-check outputs and a flat list
    of human-readable warning strings.

    This is a thin convenience wrapper; call the individual calc_* functions
    directly for programmatic access to p-values/masks/indices.
    calc_dynamic_neurons (#39) is deliberately excluded -- it's a pre-fit
    neuron-selection tool, not a post-fit diagnostic warning.

    Args:
        spike_array (3D Numpy array): trials x neurons x bins (e.g.
            processed_spikes).
        tau_array (2D Numpy array): trials x changepoints (e.g.
            tau.raw_mode_tau).
        n_bins (int): total number of time bins per trial.
        **kwargs: forwarded to individual checks via namespaced keys, e.g.
            drift_anova={"n_trial_bins": 6}, state_uniformity={"p_val": 0.01},
            collapsed_transitions={"edge_frac": 0.1},
            transition_randomness={"method": "entropy"}.

    Returns:
        warnings_list (list of str)
        diagnostics (dict): raw outputs keyed by check name, for users who
            want the underlying arrays without re-running the checks.
    """
    diagnostics = {}
    warnings_list = []

    neuron_p, drift_neurons, pop_p, is_drift = calc_firing_drift_anova(
        spike_array, **kwargs.get("drift_anova", {}))
    diagnostics["drift_anova"] = (neuron_p, drift_neurons, pop_p, is_drift)
    if is_drift:
        warnings_list.append(
            f"Possible session drift: {len(drift_neurons)} neurons show "
            f"significant firing-rate change across the session "
            f"(binomial p={pop_p:.3g})."
        )

    state_p, nonuniform_states = calc_state_trial_uniformity(
        tau_array, n_bins, **kwargs.get("state_uniformity", {}))
    diagnostics["state_trial_uniformity"] = (state_p, nonuniform_states)
    if len(nonuniform_states):
        warnings_list.append(
            f"Possible drift: state(s) {nonuniform_states.tolist()} appear "
            f"in a non-uniform, trial-order-dependent subset of trials."
        )

    edge_mask, merged_mask, collapse_warnings = calc_collapsed_transitions(
        tau_array, n_bins, **kwargs.get("collapsed_transitions", {}))
    diagnostics["collapsed_transitions"] = (edge_mask, merged_mask)
    warnings_list.extend(collapse_warnings)

    rand_stat, random_transitions = calc_transition_randomness(
        tau_array, n_bins, **kwargs.get("transition_randomness", {}))
    diagnostics["transition_randomness"] = (rand_stat, random_transitions)
    if len(random_transitions):
        warnings_list.append(
            f"Transition(s) {random_transitions.tolist()} are statistically "
            f"indistinguishable from a uniform/random distribution over trials."
        )

    return warnings_list, diagnostics


class _firing:
    """Helper class to handle processing for firing rate using "EphysData" """

    def __init__(self, tau_instance, processed_spikes, metadata):
        """Initialize firing class

        Args:
            tau_instance (Class): Tau class containing metadata and relevant variables
            processed_spikes (Numpy array): Numpy array containing processed spiking data
            metadata (Dict): Dict containing metadata on fit
        """
        # Check that inputs are valid, if not, raise error
        boilerplate_msg = "Error in _firing initialization:\n"
        assert isinstance(tau_instance, _tau), \
            boilerplate_msg + \
            f"tau_instance must be of type _tau, currently {type(tau_instance)}"
        assert isinstance(processed_spikes, np.ndarray), \
            boilerplate_msg + \
            f"processed_spikes must be a numpy array, currently {type(processed_spikes)}"
        assert isinstance(metadata, dict), \
            boilerplate_msg + \
            f"metadata must be a dict, currently {type(metadata)}"

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
        # Handle case where tau attributes are None (e.g., from fallback pickling)
        if self.tau.raw_mode_tau is not None and self.tau.scaled_mode_tau is not None:
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
        else:
            # Set to None if tau data is not available
            self.state_firing = None
            self.transition_snips = None
            self.anova_p_val_array = None
            self.anova_significant_neurons = None
            self.pairwise_p_val_array = None
            self.pairwise_significant_neurons = None


class _tau:
    """Tau class to keep track of metadata and perform useful transformations"""

    def __init__(self, tau_array, metadata, n_trials=None):
        """Initialize tau class

        Args:
            tau_array ([type]): Array of samples from fitted model
            metadata (Dict): Dict containing metadata on fit
        """

        # Check that inputs are valid, if not, raise error
        boilerplate_msg = "Error in _tau initialization:\n"
        assert isinstance(metadata, dict), \
            boilerplate_msg + \
            f"metadata must be a dict, currently {type(metadata)}"
        assert isinstance(tau_array, np.ndarray), \
            boilerplate_msg + \
            f"tau_array must be a numpy array, currently {type(tau_array)}"
        if n_trials is not None:
            assert isinstance(n_trials, int), \
                boilerplate_msg + \
                f"n_trials must be an int, currently {type(n_trials)}"

        self.raw_tau = tau_array

        # Handle case where tau_array is None (e.g., from fallback pickling)
        if tau_array is None:
            self.raw_int_tau = None
            self.raw_mode_tau = None
            self.scaled_tau = None
            self.scaled_int_tau = None
            self.scaled_mode_tau = None
            return

        time_lims = metadata["preprocess"]["time_lims"]
        bin_width = metadata["preprocess"]["bin_width"]

        self.raw_int_tau = np.vectorize(int)(self.raw_tau)
        raw_mode_result = np.squeeze(mode(self.raw_int_tau)[0])
        # Ensure raw_mode_result is 1D array of changepoints
        if raw_mode_result.ndim == 0:
            raw_mode_result = np.array([raw_mode_result])

        # If n_trials is provided, replicate changepoints for each trial
        # This is needed for functions that expect (n_trials, n_changepoints)
        if n_trials is not None:
            self.raw_mode_tau = np.tile(raw_mode_result, (n_trials, 1))
        else:
            self.raw_mode_tau = raw_mode_result

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
            if key in self.data["model_data"]:
                setattr(self, var_name, self.data["model_data"][key])
            else:
                # Set to None if key is missing (e.g., due to pickling fallback)
                setattr(self, var_name, None)

        self.metadata = self.data["metadata"]
        self.pretty_metadata = pd.json_normalize(self.data["metadata"]).T

        # Get number of trials from processed_spikes for proper tau formatting
        n_trials = self.processed_spikes.shape[0] if hasattr(
            self.processed_spikes, 'shape') else None
        self.tau = _tau(self.tau_array, self.metadata, n_trials)
        self.firing = _firing(self.tau, self.processed_spikes, self.metadata)

    def get_diagnostics(self, **kwargs):
        """Run post-fit data-quality diagnostics (issues #37/#38) on this
        fit and cache results on self.warnings (list[str]) and
        self.diagnostics (dict). Not run automatically in __init__ -- call
        explicitly, since some callers construct PklHandler in tight loops
        over many saved fits and shouldn't pay for these checks every time.

        Args:
            **kwargs: forwarded to get_fit_diagnostics (see its docstring
                for the per-check namespaced options).

        Returns:
            self.warnings (list of str)
        """
        n_bins = self.processed_spikes.shape[-1]
        self.warnings, self.diagnostics = get_fit_diagnostics(
            self.processed_spikes, self.tau.raw_mode_tau, n_bins, **kwargs
        )
        return self.warnings
