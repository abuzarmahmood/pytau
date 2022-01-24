"""
PyMC3 Blackbox Variational Inference implementation
of Poisson Likelihood Changepoint for spike trains.
"""

########################################
# ___                            _
# |_ _|_ __ ___  _ __   ___  _ __| |_
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_
# |___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|
########################################
import pymc3 as pm
import theano.tensor as tt
import numpy as np

############################################################
#  ____                _         __  __           _      _
# / ___|_ __ ___  __ _| |_ ___  |  \/  | ___   __| | ___| |
# | |   | '__/ _ \/ _` | __/ _ \ | |\/| |/ _ \ / _` |/ _ \ |
# | |___| | |  __/ (_| | ||  __/ | |  | | (_) | (_| |  __/ |
# \____|_|  \___|\__,_|\__\___| |_|  |_|\___/ \__,_|\___|_|
############################################################


def single_taste_poisson(
        spike_array,
        states):
    """Model for changepoint on single taste

    ** Largely taken from "non_hardcoded_changepoint_test_3d.ipynb"
    ** Note : This model does not have hierarchical structure for emissions

    Args:
        spike_array (3D Numpy array): trials x neurons x time
        states (int): Number of states to model

    Returns:
        pymc3 model: Model class containing graph to run inference on
    """

    mean_vals = np.array([np.mean(x, axis=-1)
                          for x in np.array_split(spike_array, states, axis=-1)]).T
    mean_vals = np.mean(mean_vals, axis=1)
    mean_vals += 0.01  # To avoid zero starting prob

    nrns = spike_array.shape[1]
    trials = spike_array.shape[0]
    idx = np.arange(spike_array.shape[-1])
    length = idx.max() + 1

    with pm.Model() as model:
        lambda_latent = pm.Exponential('lambda',
                                       1/mean_vals,
                                       shape=(nrns, states))

        a_tau = pm.HalfCauchy('a_tau', 3., shape=states - 1)
        b_tau = pm.HalfCauchy('b_tau', 3., shape=states - 1)

        even_switches = np.linspace(0, 1, states+1)[1:-1]
        tau_latent = pm.Beta('tau_latent', a_tau, b_tau,
                             testval=even_switches,
                             shape=(trials, states-1)).sort(axis=-1)

        tau = pm.Deterministic('tau',
                               idx.min() + (idx.max() - idx.min()) * tau_latent)

        weight_stack = tt.nnet.sigmoid(
            idx[np.newaxis, :]-tau[:, :, np.newaxis])
        weight_stack = tt.concatenate(
            [np.ones((trials, 1, length)), weight_stack], axis=1)
        inverse_stack = 1 - weight_stack[:, 1:]
        inverse_stack = tt.concatenate(
            [inverse_stack, np.ones((trials, 1, length))], axis=1)
        weight_stack = np.multiply(weight_stack, inverse_stack)

        lambda_ = tt.tensordot(weight_stack, lambda_latent, [1, 1]).swapaxes(1, 2)
        observation = pm.Poisson("obs", lambda_, observed=spike_array)

    return model


# def all_taste_poisson(
#         spike_array,
#         states):
#     pass

# def single_taste_poisson_biased_tau_priors(spike_array,states):
#     pass

# def single_taste_poisson_hard_padding_tau(spike_array,states):
#     pass

######################################################################
# |  _ \ _   _ _ __   |_ _|_ __  / _| ___ _ __ ___ _ __   ___ ___
# | |_) | | | | '_ \   | || '_ \| |_ / _ \ '__/ _ \ '_ \ / __/ _ \
# |  _ <| |_| | | | |  | || | | |  _|  __/ | |  __/ | | | (_|  __/
# |_| \_\\__,_|_| |_| |___|_| |_|_|  \___|_|  \___|_| |_|\___\___|
######################################################################

def advi_fit(model, fit, samples):
    """Convenience function to perform ADVI fit on model

    Args:
        model (pymc3 model): model object to run inference on
        fit (int): Number of iterationst to fit the model for
        samples (int): Number of samples to draw from fitted model

    Returns:
        model: original model on which inference was run,
        approx: fitted model,
        lambda_stack: array containing lambda (emission) values,
        tau_samples,: array containing samples from changepoint distribution
        model.obs.observations: processed array on which fit was run
    """

    with model:
        inference = pm.ADVI('full-rank')
        approx = pm.fit(n=fit, method=inference)
        trace = approx.sample(draws=samples)

    # Extract relevant variables from trace
    lambda_stack = trace['lambda'].swapaxes(0, 1)
    tau_samples = trace['tau']

    return model, approx, lambda_stack, tau_samples, model.obs.observations
