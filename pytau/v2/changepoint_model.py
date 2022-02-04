"""
PyMC3 Blackbox Variational Inference implementation
of Poisson Likelihood Changepoint for spike trains.
"""

########################################
# ___                            _   
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   
########################################
import pymc3 as pm
import theano.tensor as tt
import numpy as np

############################################################
#  ____                _         __  __           _      _ 
# / ___|_ __ ___  __ _| |_ ___  |  \/  | ___   __| | ___| |
#| |   | '__/ _ \/ _` | __/ _ \ | |\/| |/ _ \ / _` |/ _ \ |
#| |___| | |  __/ (_| | ||  __/ | |  | | (_) | (_| |  __/ |
# \____|_|  \___|\__,_|\__\___| |_|  |_|\___/ \__,_|\___|_|
############################################################

def single_taste_poisson(
        spike_array,
        states):

    """
    ** Model to fit changepoint to single taste **
    ** Largely taken from "non_hardcoded_changepoint_test_3d.ipynb"
    ** Note : This model does not have hierarchical structure for emissions

    spike_array :: Shape : trials, neurons, time_bins
    states :: number of states to include in the model 
    """

    mean_vals = np.array([np.mean(x,axis=-1) \
            for x in np.array_split(spike_array,states,axis=-1)]).T
    mean_vals = np.mean(mean_vals,axis=1)
    mean_vals += 0.01 # To avoid zero starting prob

    nrns = spike_array.shape[1]
    trials = spike_array.shape[0]
    idx = np.arange(spike_array.shape[-1])
    length = idx.max() + 1

    with pm.Model() as model:
        lambda_latent = pm.Exponential('lambda',
                                    1/mean_vals, 
                                    shape = (nrns,states))


        a = pm.HalfCauchy('a_tau', 3., shape = states - 1)
        b = pm.HalfCauchy('b_tau', 3., shape = states - 1)

        even_switches = np.linspace(0,1,states+1)[1:-1]
        tau_latent = pm.Beta('tau_latent', a, b, 
                testval = even_switches,
                shape = (trials,states-1)).sort(axis=-1)    

        tau = pm.Deterministic('tau', 
                idx.min() + (idx.max() - idx.min()) * tau_latent)

        weight_stack = tt.nnet.sigmoid(idx[np.newaxis,:]-tau[:,:,np.newaxis])
        weight_stack = tt.concatenate([np.ones((trials,1,length)),weight_stack],axis=1)
        inverse_stack = 1 - weight_stack[:,1:]
        inverse_stack = tt.concatenate([inverse_stack, np.ones((trials,1,length))],axis=1)
        weight_stack = np.multiply(weight_stack,inverse_stack)

        lambda_ = tt.tensordot(weight_stack,lambda_latent, [1,1]).swapaxes(1,2)
        observation = pm.Poisson("obs", lambda_, observed=spike_array)

    return model


def all_taste_poisson(
        spike_array,
        states):

    """
    ** Model to fit changepoint to single taste **
    ** Largely taken from "_v1/poisson_all_tastes_changepoint_model.py"

    spike_array :: Shape : tastes, trials, neurons, time_bins
    states :: number of states to include in the model 
    """

    # If model already doesn't exist, then create new one
    #spike_array = this_dat_binned
    # Unroll arrays along taste axis
    #spike_array_long = np.reshape(spike_array,(-1,*spike_array.shape[-2:]))
    spike_array_long = np.concatenate(spike_array, axis=0) 

    # Find mean firing for initial values
    tastes = spike_array.shape[0]
    length = spike_array.shape[-1]
    nrns = spike_array.shape[2]
    trials = spike_array.shape[1]

    split_list = np.array_split(spike_array,states,axis=-1)
    # Cut all to the same size
    min_val = min([x.shape[-1] for x in split_list])
    split_array = np.array([x[...,:min_val] for x in split_list])
    mean_vals = np.mean(split_array,axis=(2,-1)).swapaxes(0,1)
    mean_vals += 0.01 # To avoid zero starting prob
    mean_nrn_vals = np.mean(mean_vals,axis=(0,1))

    # Find evenly spaces switchpoints for initial values
    idx = np.arange(spike_array.shape[-1]) # Index
    array_idx = np.broadcast_to(idx, spike_array_long.shape)
    even_switches = np.linspace(0,idx.max(),states+1)
    even_switches_normal = even_switches/np.max(even_switches)

    taste_label = np.repeat(np.arange(spike_array.shape[0]), spike_array.shape[1])
    trial_num = array_idx.shape[0]

    # Being constructing model
    with pm.Model() as model:

        # Hierarchical firing rates
        # Refer to model diagram
        # Mean firing rate of neuron AT ALL TIMES
        lambda_nrn = pm.Exponential('lambda_nrn',
                                    1/mean_nrn_vals,
                                    shape=(mean_vals.shape[-1]))
        # Priors for each state, derived from each neuron
        # Mean firing rate of neuron IN EACH STATE (averaged across tastes)
        lambda_state = pm.Exponential('lambda_state',
                                      lambda_nrn,
                                      shape=(mean_vals.shape[1:]))
        # Mean firing rate of neuron PER STATE PER TASTE
        lambda_latent = pm.Exponential('lambda',
                                       lambda_state[np.newaxis, :, :],
                                       testval=mean_vals,
                                       shape=(mean_vals.shape))

        # Changepoint time variable
        # INDEPENDENT TAU FOR EVERY TRIAL
        a = pm.HalfNormal('a_tau', 3., shape=states - 1)
        b = pm.HalfNormal('b_tau', 3., shape=states - 1)

        # Stack produces states x trials --> That gets transposed
        # to trials x states and gets sorted along states (axis=-1)
        # Sort should work the same way as the Ordered transform -->
        # see rv_sort_test.ipynb
        tau_latent = pm.Beta('tau_latent', a, b,
                             shape=(trial_num, states-1),
                             testval=tt.tile(even_switches_normal[1:(states)],
                                             (array_idx.shape[0], 1))).sort(axis=-1)

        tau = pm.Deterministic('tau',
                               idx.min() + (idx.max() - idx.min()) * tau_latent)

        weight_stack = tt.nnet.sigmoid(
            idx[np.newaxis, :]-tau[:, :, np.newaxis])
        weight_stack = tt.concatenate(
            [np.ones((tastes*trials, 1, length)), weight_stack], axis=1)
        inverse_stack = 1 - weight_stack[:, 1:]
        inverse_stack = tt.concatenate([inverse_stack, np.ones(
            (tastes*trials, 1, length))], axis=1)
        weight_stack = weight_stack*inverse_stack
        weight_stack = tt.tile(weight_stack[:, :, None, :], (1, 1, nrns, 1))

        lambda_latent = lambda_latent.dimshuffle(2, 0, 1)
        lambda_latent = tt.repeat(lambda_latent, trials, axis=1)
        lambda_latent = tt.tile(
            lambda_latent[..., None], (1, 1, 1, length))
        lambda_latent = lambda_latent.dimshuffle(1, 2, 0, 3)
        lambda_ = tt.sum(lambda_latent * weight_stack, axis=1)

        observation = pm.Poisson("obs", lambda_, observed=spike_array_long)

    return model

def single_taste_poisson_biased_tau_priors(spike_array,states):
    pass

def single_taste_poisson_hard_padding_tau(spike_array,states):
    pass

######################################################################
#|  _ \ _   _ _ __   |_ _|_ __  / _| ___ _ __ ___ _ __   ___ ___ 
#| |_) | | | | '_ \   | || '_ \| |_ / _ \ '__/ _ \ '_ \ / __/ _ \
#|  _ <| |_| | | | |  | || | | |  _|  __/ | |  __/ | | | (_|  __/
#|_| \_\\__,_|_| |_| |___|_| |_|_|  \___|_|  \___|_| |_|\___\___|
######################################################################

def advi_fit(model,fit,samples):
    """
    model : PyMC3 changepoint model as defined above
    fit : number of iterations to fit
    samples : number of samples to generate from fitted model
    """

    with model:
        inference = pm.ADVI('full-rank')
        approx = pm.fit(n=fit, method=inference)
        trace = approx.sample(draws=samples)

    # Extract relevant variables from trace
    lambda_stack = trace['lambda'].swapaxes(0,1)
    tau_samples = trace['tau']

    return model, approx, lambda_stack, tau_samples, model.obs.observations
