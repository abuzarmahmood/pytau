"""
PyMC3 Blackbox Variational Inference implementation
of Poisson Likelihood Changepoint for spike trains.
"""

########################################
# Import
########################################
import pymc3 as pm
import theano
import theano.tensor as tt
import numpy as np
import os
import time

############################################################
# Functions
############################################################

def theano_lock_present():
    """
    Check if theano compilation lock is present
    """
    return os.path.exists(os.path.join(theano.config.compiledir, 'lock_dir'))

def compile_wait():
    """
    Function to allow waiting while a model is already fitting
    Wait twice because lock blips out between steps
    10 secs of waiting shouldn't be a problem for long fits (~mins)
    And wait a random time in the beginning to stagger fits
    """
    time.sleep(np.random.random()*10)
    while theano_lock_present():
        print('Lock present...waiting')
        time.sleep(10)
    while theano_lock_present():
        print('Lock present...waiting')
        time.sleep(10)

def gen_test_array(array_size, n_states, type = 'poisson'):
    """
    Generate test array for model fitting
    Last 2 dimensions consist of a single trial
    Time will always be last dimension

    Args:
        array_size (tuple): Size of array to generate
        n_states (int): Number of states to generate
        type (str): Type of data to generate
            - normal
            - poisson
    """
    assert array_size[-1] > n_states, 'Array too small for states'
    assert type in ['normal', 'poisson'], 'Invalid type, please use normal or poisson'

    # Generate transition times
    transition_times = np.random.random((*array_size[:-2], n_states))
    transition_times = np.cumsum(transition_times, axis=-1)
    transition_times = transition_times / transition_times.max(axis=-1, keepdims=True)
    transition_times *= array_size[-1]
    transition_times = np.vectorize(int)(transition_times)

    # Generate state bounds
    state_bounds = np.zeros((*array_size[:-2], n_states+1), dtype = int)
    state_bounds[..., 1:] = transition_times

    # Generate state rates
    lambda_vals = np.random.random((*array_size[:-1], n_states))

    # Generate array
    rate_array = np.zeros(array_size)
    inds = list(np.ndindex(lambda_vals.shape))
    for this_ind in inds:
        this_lambda = lambda_vals[this_ind[:-2]][:,this_ind[-1]]
        this_state_bounds = [
                state_bounds[(*this_ind[:-2], this_ind[-1])], 
                state_bounds[(*this_ind[:-2], this_ind[-1]+1)]]
        rate_array[this_ind[:-2]][:, slice(*this_state_bounds)] = this_lambda[:,None]

    if type == 'poisson':
        return np.random.poisson(rate_array)
    else:
        return np.random.normal(loc = rate_array, scale = 0.1)


def gaussian_changepoint_2d(data_array, n_states, **kwargs):
    """Model for gaussian data on 2D array

    Args:
        data_array (2D Numpy array): <dimension> x time
        n_states (int): Number of states to model

    Returns:
        pymc3 model: Model class containing graph to run inference on
    """
    mean_vals = np.array([np.mean(x, axis=-1)
                          for x in np.array_split(data_array, n_states, axis=-1)]).T
    mean_vals += 0.01  # To avoid zero starting prob

    y_dim = data_array.shape[0]
    idx = np.arange(data_array.shape[-1])
    length = idx.max() + 1

    with pm.Model() as model:
        mu = pm.Normal('mu', mu=mean_vals, sd=1, shape=(y_dim, n_states))
        sigma = pm.HalfCauchy('sigma', 3., shape=(y_dim, n_states))

        a_tau = pm.HalfCauchy('a_tau', 3., shape=n_states - 1)
        b_tau = pm.HalfCauchy('b_tau', 3., shape=n_states - 1)

        even_switches = np.linspace(0, 1, n_states+1)[1:-1]
        tau_latent = pm.Beta('tau_latent', a_tau, b_tau,
                             testval=even_switches,
                             shape=(n_states-1)).sort(axis=-1)

        tau = pm.Deterministic('tau',
                               idx.min() + (idx.max() - idx.min()) * tau_latent)

        weight_stack = tt.nnet.sigmoid(idx[np.newaxis,:]-tau[:,np.newaxis])
        weight_stack = tt.concatenate([np.ones((1,length)),weight_stack],axis=0)
        inverse_stack = 1 - weight_stack[1:]
        inverse_stack = tt.concatenate([inverse_stack, np.ones((1,length))],axis=0)
        weight_stack = np.multiply(weight_stack,inverse_stack)

        mu_latent = mu.dot(weight_stack)
        sigma_latent = sigma.dot(weight_stack)
        observation = pm.Normal("obs", mu = mu_latent, sd = sigma_latent, 
                                observed = data_array)

    return model


def single_taste_poisson(
        spike_array,
        states,
        **kwargs):
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

def var_sig_exp_tt(x,b):
    """
    x -->
    b -->
    """
    return 1/(1+tt.exp(-tt.exp(b)*x))

def var_sig_tt(x,b):
    """
    x -->
    b -->
    """
    return 1/(1+tt.exp(-b*x))

def single_taste_poisson_varsig(
        spike_array,
        states,
        **kwargs):
    """Model for changepoint on single taste
        **Uses variables sigmoid slope inferred from data

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

    lambda_test_vals = np.diff(mean_vals, axis=-1)
    even_switches = np.linspace(0,1,states+1)[1:-1]

    nrns = spike_array.shape[1]
    trials = spike_array.shape[0]
    idx = np.arange(spike_array.shape[-1])
    length = idx.max() + 1

    with pm.Model() as model:
        
        # Sigmoid slope
        sig_b = pm.Normal('sig_b', -1,2, shape = states-1)
        
        # Initial value
        s0 = pm.Exponential('state0', 
                            1/(np.mean(mean_vals)),
                            shape = nrns,
                            testval = mean_vals[:,0])
        
        # Changes to lambda
        lambda_diff = pm.Normal('lambda_diff', 
                                mu = 0, sigma = 10, 
                                shape = (nrns,states-1), 
                                testval = lambda_test_vals)

        # This is only here to be extracted at the end of sampling
        # NOT USED DIRECTLY IN MODEL
        lambda_fin = pm.Deterministic('lambda',
                        tt.concatenate([s0[:,np.newaxis],lambda_diff],axis=-1)
                        )
        
        # Changepoint positions
        a = pm.HalfCauchy('a_tau', 10, shape = states - 1)
        b = pm.HalfCauchy('b_tau', 10, shape = states - 1)
        
        tau_latent = pm.Beta('tau_latent', a, b, 
                            testval = even_switches,
                            shape = (trials, states-1)).sort(axis=-1)    
        tau = pm.Deterministic('tau', 
                idx.min() + (idx.max() - idx.min()) * tau_latent)
        
        # Mechanical manipulations to generate firing rates
        idx_temp = np.tile(idx[np.newaxis,np.newaxis,:], (trials, states-1,1))
        tau_temp = tt.tile(tau[:,:,np.newaxis], (1,1,len(idx)))
        sig_b_temp = tt.tile(sig_b[np.newaxis,:,np.newaxis], (trials,1,len(idx)))

        weight_stack = var_sig_exp_tt(idx_temp-tau_temp,sig_b_temp)
        weight_stack_temp = tt.tile(weight_stack[:,np.newaxis,:,:], (1,nrns,1,1))

        s0_temp = tt.tile(s0[np.newaxis,:,np.newaxis,np.newaxis], (trials,1,states-1, len(idx)))
        lambda_diff_temp = tt.tile(lambda_diff[np.newaxis,:,:,np.newaxis], (trials,1,1, len(idx)))

        # Calculate lambda
        lambda_ =  pm.Deterministic('lambda_',
                        tt.sum(s0_temp + (weight_stack_temp*lambda_diff_temp),axis=2))
        # Bound lambda to prevent the diffs from making it negative
        # Don't let it go down to zero otherwise we have trouble with probabilities
        lambda_bounded = pm.Deterministic("lambda_bounded", 
                tt.switch(lambda_>=0.01, lambda_, 0.01))
        
        # Add observations
        observation = pm.Poisson("obs", lambda_bounded, observed=spike_array)

    return model

def inds_to_b(x_span):
    return 5.8889/x_span

def single_taste_poisson_varsig_fixed(
        spike_array,
        states,
        inds_span = 1):
    """Model for changepoint on single taste
        **Uses sigmoid with given slope

    ** Largely taken from "non_hardcoded_changepoint_test_3d.ipynb"
    ** Note : This model does not have hierarchical structure for emissions

    Args:
        spike_array (3D Numpy array): trials x neurons x time
        states (int): Number of states to model
        inds_span(float) : Number of indices to cover 5-95% change in sigmoid

    Returns:
        pymc3 model: Model class containing graph to run inference on
    """

    mean_vals = np.array([np.mean(x, axis=-1)
                          for x in np.array_split(spike_array, states, axis=-1)]).T
    mean_vals = np.mean(mean_vals, axis=1)
    mean_vals += 0.01  # To avoid zero starting prob

    lambda_test_vals = np.diff(mean_vals, axis=-1)
    even_switches = np.linspace(0,1,states+1)[1:-1]

    nrns = spike_array.shape[1]
    trials = spike_array.shape[0]
    idx = np.arange(spike_array.shape[-1])
    length = idx.max() + 1

    # Define sigmoid with given sharpness
    sig_b = inds_to_b(inds_span)
    def sigmoid(x):
        b_temp = tt.tile(np.array(sig_b)[None,None,None], x.tag.test_value.shape)
        return 1/(1+tt.exp(-b_temp*x))

    with pm.Model() as model:
        
        # Initial value
        s0 = pm.Exponential('state0', 
                            1/(np.mean(mean_vals)),
                            shape = nrns,
                            testval = mean_vals[:,0])
        
        # Changes to lambda
        lambda_diff = pm.Normal('lambda_diff', 
                                mu = 0, sigma = 10, 
                                shape = (nrns,states-1), 
                                testval = lambda_test_vals)

        # This is only here to be extracted at the end of sampling
        # NOT USED DIRECTLY IN MODEL
        lambda_fin = pm.Deterministic('lambda',
                        tt.concatenate([s0[:,np.newaxis],lambda_diff],axis=-1)
                        )
        
        # Changepoint positions
        a = pm.HalfCauchy('a_tau', 10, shape = states - 1)
        b = pm.HalfCauchy('b_tau', 10, shape = states - 1)
        
        tau_latent = pm.Beta('tau_latent', a, b, 
                            testval = even_switches,
                            shape = (trials, states-1)).sort(axis=-1)    
        tau = pm.Deterministic('tau', 
                idx.min() + (idx.max() - idx.min()) * tau_latent)
        
        # Mechanical manipulations to generate firing rates
        idx_temp = np.tile(idx[np.newaxis,np.newaxis,:], (trials, states-1,1))
        tau_temp = tt.tile(tau[:,:,np.newaxis], (1,1,len(idx)))

        weight_stack = sigmoid(idx_temp-tau_temp)
        weight_stack_temp = tt.tile(weight_stack[:,np.newaxis,:,:], (1,nrns,1,1))

        s0_temp = tt.tile(s0[np.newaxis,:,np.newaxis,np.newaxis], (trials,1,states-1, len(idx)))
        lambda_diff_temp = tt.tile(lambda_diff[np.newaxis,:,:,np.newaxis], (trials,1,1, len(idx)))

        # Calculate lambda
        lambda_ =  pm.Deterministic('lambda_',
                        tt.sum(s0_temp + (weight_stack_temp*lambda_diff_temp),axis=2))
        # Bound lambda to prevent the diffs from making it negative
        # Don't let it go down to zero otherwise we have trouble with probabilities
        lambda_bounded = pm.Deterministic("lambda_bounded", 
                tt.switch(lambda_>=0.01, lambda_, 0.01))
        
        # Add observations
        observation = pm.Poisson("obs", lambda_bounded, observed=spike_array)

    return model

def all_taste_poisson(
        spike_array,
        states,
        **kwargs):

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

def all_taste_poisson_varsig_fixed(
        spike_array,
        states,
        inds_span = 1):

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

    # Define sigmoid with given sharpness
    sig_b = inds_to_b(inds_span)
    def sigmoid(x):
        b_temp = tt.tile(np.array(sig_b)[None,None,None], x.tag.test_value.shape)
        return 1/(1+tt.exp(-b_temp*x))

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

        weight_stack = sigmoid(
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

# def single_taste_poisson_biased_tau_priors(spike_array,states):
#     pass

# def single_taste_poisson_hard_padding_tau(spike_array,states):
#     pass


def single_taste_poisson_trial_switch(
        spike_array,
        switch_components,
        states):

    """
    Assuming only emissions change across trials
    Changepoint distribution remains constant

    spike_array :: trials x nrns x time
    states :: number of states to include in the model 
    """

    trial_num, nrn_num, time_bins = spike_array.shape

    with pm.Model() as model:
        
        # Define Emissions
        
        # nrns
        nrn_lambda = pm.Exponential('nrn_lambda', 10, shape = (nrn_num))
        
        # nrns x switch_comps
        trial_lambda = pm.Exponential('trial_lambda', 
                                    nrn_lambda.dimshuffle(0,'x'), 
                                    shape = (nrn_num, switch_components))
        
        # nrns x switch_comps x states
        state_lambda = pm.Exponential('state_lambda',
                                    trial_lambda.dimshuffle(0,1,'x'),
                                    shape = (nrn_num, switch_components, states))
        
        # Define Changepoints
        # Assuming distribution of changepoints remains
        # the same across all trials

        a = pm.HalfCauchy('a_tau', 3., shape = states - 1)
        b = pm.HalfCauchy('b_tau', 3., shape = states - 1)
        
        even_switches = np.linspace(0,1,states+1)[1:-1]
        tau_latent = pm.Beta('tau_latent', a, b, 
                            testval = even_switches,
                            shape = (trial_num,states-1)).sort(axis=-1)    
        
        # Trials x Changepoints
        tau = pm.Deterministic('tau', time_bins * tau_latent)
        
        # Define trial switches
        # Will have same structure as regular changepoints
        
        even_trial_switches = np.linspace(0,1,switch_components+1)[1:-1]
        tau_trial_latent = pm.Beta('tau_trial_latent', 1, 1, 
                            testval = even_trial_switches,
                            shape = (switch_components-1)).sort(axis=-1)    
        
        # Trial_changepoints
        tau_trial = pm.Deterministic('tau_trial', trial_num * tau_trial_latent)
        
        trial_idx = np.arange(trial_num)
        trial_selector = tt.nnet.sigmoid(trial_idx[np.newaxis,:] - tau_trial.dimshuffle(0,'x'))
        
        trial_selector = tt.concatenate([np.ones((1,trial_num)),trial_selector],axis=0)
        inverse_trial_selector = 1 - trial_selector[1:,:]
        inverse_trial_selector = tt.concatenate([inverse_trial_selector, 
                                                np.ones((1,trial_num))],axis=0)

        # First, we can "select" sets of emissions depending on trial_changepoints 
        # switch_comps x trials
        trial_selector = np.multiply(trial_selector,inverse_trial_selector)
        
        # state_lambda: nrns x switch_comps x states
        
        # selected_trial_lambda : nrns x states x trials
        selected_trial_lambda = pm.Deterministic('selected_trial_lambda',
                                tt.sum(
                                # "nrns" x switch_comps x "states" x trials
                                trial_selector.dimshuffle('x',0,'x',1) * state_lambda.dimshuffle(0,1,2,'x'),
                                axis=1)
                                                )
        
        # Then, we can select state_emissions for every trial
        idx = np.arange(time_bins)
        
        # tau : Trials x Changepoints
        weight_stack = tt.nnet.sigmoid(idx[np.newaxis,:]-tau[:,:,np.newaxis])
        weight_stack = tt.concatenate([np.ones((trial_num,1,time_bins)),weight_stack],axis=1)
        inverse_stack = 1 - weight_stack[:,1:]
        inverse_stack = tt.concatenate([inverse_stack, np.ones((trial_num,1,time_bins))],axis=1)

        # Trials x states x Time
        weight_stack = np.multiply(weight_stack,inverse_stack)
        
        # Convert selected_trial_lambda : nrns x trials x states x "time"
        
        # nrns x trials x time
        lambda_ = tt.sum(selected_trial_lambda.dimshuffle(0,2,1,'x') * weight_stack.dimshuffle('x',0,1,2),
                        axis = 2)
        
        # Convert to : trials x nrns x time
        lambda_ = lambda_.dimshuffle(1,0,2)
        
        # Add observations
        observation = pm.Poisson("obs", lambda_, observed=spike_array)
    
    return model


def all_taste_poisson_trial_switch(
        spike_array,
        switch_components,
        states):

    """
    Assuming only emissions change across trials
    Changepoint distribution remains constant

    spike_array :: Tastes x trials x nrns x time_bins
    states :: number of states to include in the model 
    """

    tastes, trial_num, nrn_num, time_bins = spike_array.shape

    with pm.Model() as model:
        
        # Define Emissions
        # =================================================
        
        # nrns
        nrn_lambda = pm.Exponential('nrn_lambda', 10, shape = (nrn_num))
        
        # tastes x nrns
        taste_lambda = pm.Exponential('taste_lambda', 
                                    nrn_lambda.dimshuffle('x',0),
                                    shape = (tastes, nrn_num))
        
        # tastes x nrns x switch_comps
        trial_lambda = pm.Exponential('trial_lambda', 
                                    taste_lambda.dimshuffle(0,1,'x'), 
                                    shape = (tastes, nrn_num, switch_components))
        
        # tastes x nrns x switch_comps x states
        state_lambda = pm.Exponential('state_lambda',
                                    trial_lambda.dimshuffle(0,1,2,'x'),
                                    shape = (tastes, nrn_num, switch_components, states))
        
        # Define Changepoints
        # =================================================
        # Assuming distribution of changepoints remains
        # the same across all trials

        a = pm.HalfCauchy('a_tau', 3., shape = states - 1)
        b = pm.HalfCauchy('b_tau', 3., shape = states - 1)
        
        even_switches = np.linspace(0,1,states+1)[1:-1]
        tau_latent = pm.Beta('tau_latent', a, b, 
                            testval = even_switches,
                            shape = (tastes, trial_num, states-1)).sort(axis=-1)    
        
        # Tasets x Trials x Changepoints
        tau = pm.Deterministic('tau', time_bins * tau_latent)
        
        # Define trial switches
        # Will have same structure as regular changepoints
        
        #a_trial = pm.HalfCauchy('a_trial', 3., shape = switch_components - 1)
        #b_trial = pm.HalfCauchy('b_trial', 3., shape = switch_components - 1)
        
        even_trial_switches = np.linspace(0,1,switch_components+1)[1:-1]
        tau_trial_latent = pm.Beta('tau_trial_latent', 1, 1, 
                            testval = even_trial_switches,
                            shape = (switch_components-1)).sort(axis=-1)    
        
        # Trial_changepoints
        # =================================================
        tau_trial = pm.Deterministic('tau_trial', trial_num * tau_trial_latent)
        
        trial_idx = np.arange(trial_num)
        trial_selector = tt.nnet.sigmoid(trial_idx[np.newaxis,:] - tau_trial.dimshuffle(0,'x'))
        
        trial_selector = tt.concatenate([np.ones((1,trial_num)),trial_selector],axis=0)
        inverse_trial_selector = 1 - trial_selector[1:,:]
        inverse_trial_selector = tt.concatenate([inverse_trial_selector, 
                                                np.ones((1,trial_num))],axis=0)

        # switch_comps x trials
        trial_selector = np.multiply(trial_selector,inverse_trial_selector)
        
        # state_lambda: tastes x nrns x switch_comps x states
        
        # selected_trial_lambda : tastes x nrns x states x trials
        selected_trial_lambda = pm.Deterministic('selected_trial_lambda',
                                tt.sum(
                                # "tastes" x "nrns" x switch_comps x "states" x trials
                                trial_selector.dimshuffle('x','x',0,'x',1) * state_lambda.dimshuffle(0,1,2,3,'x'),
                                axis=2)
                                                )
        
        # First, we can "select" sets of emissions depending on trial_changepoints 
        # =================================================
        trial_idx = np.arange(trial_num)
        trial_selector = tt.nnet.sigmoid(trial_idx[np.newaxis,:] - tau_trial.dimshuffle(0,'x'))
        
        trial_selector = tt.concatenate([np.ones((1,trial_num)),trial_selector],axis=0)
        inverse_trial_selector = 1 - trial_selector[1:,:]
        inverse_trial_selector = tt.concatenate([inverse_trial_selector, 
                                                np.ones((1,trial_num))],axis=0)

        # switch_comps x trials
        trial_selector = np.multiply(trial_selector,inverse_trial_selector)
        
        
        # Then, we can select state_emissions for every trial
        # =================================================
        
        idx = np.arange(time_bins)
        
        # tau : Tastes x Trials x Changepoints
        weight_stack = tt.nnet.sigmoid(idx[np.newaxis,:]-tau[:,:,:,np.newaxis])
        weight_stack = tt.concatenate([np.ones((tastes,trial_num,1,time_bins)),weight_stack],axis=2)
        inverse_stack = 1 - weight_stack[:,:,1:]
        inverse_stack = tt.concatenate([inverse_stack, np.ones((tastes,trial_num,1,time_bins))],axis=2)

        # Tastes x Trials x states x Time
        weight_stack = np.multiply(weight_stack,inverse_stack)
        
        # Putting everything together
        # =================================================
        
        # selected_trial_lambda :           tastes x nrns x states x trials
        # Convert selected_trial_lambda --> tastes x trials x nrns x states x "time"
        
        # weight_stack :           tastes x trials x states x time
        # Convert weight_stack --> tastes x trials x "nrns" x states x time
        
        # tastes x trials x nrns x time
        lambda_ = tt.sum(selected_trial_lambda.dimshuffle(0,3,1,2,'x') * weight_stack.dimshuffle(0,1,'x',2,3),
                        axis = 3)
        
        # Add observations
        observation = pm.Poisson("obs", lambda_, observed=spike_array)

    return model

######################################################################
# Run Inference
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
    tau_samples = trace['tau']
    if 'lambda' in trace.varnames:
        lambda_stack = trace['lambda'].swapaxes(0, 1)
        return model, approx, lambda_stack, tau_samples, model.obs.observations
    if 'mu' in trace.varnames:
        mu_stack = trace['mu'].swapaxes(0, 1)
        sigma_stack = trace['sigma'].swapaxes(0, 1)
        return model, approx, mu_stack, sigma_stack, tau_samples, model.obs.observations

def mcmc_fit(model, samples):
    """Convenience function to perform ADVI fit on model

    Args:
        model (pymc3 model): model object to run inference on
        samples (int): Number of samples to draw using MCMC 

    Returns:
        model: original model on which inference was run,
        trace:  samples drawn from MCMC,
        lambda_stack: array containing lambda (emission) values,
        tau_samples,: array containing samples from changepoint distribution
        model.obs.observations: processed array on which fit was run
    """

    with model:
        sampler_kwargs = {'cores' :1, 'chains':4}
        trace = pm.sample(draws=samples, **sampler_kwargs)
        trace = trace[::10]

    # Extract relevant variables from trace
    tau_samples = trace['tau']
    if 'lambda' in trace.varnames:
        lambda_stack = trace['lambda'].swapaxes(0, 1)
        return model, approx, lambda_stack, tau_samples, model.obs.observations
    if 'mu' in trace.varnames:
        mu_stack = trace['mu'].swapaxes(0, 1)
        sigma_stack = trace['sigma'].swapaxes(0, 1)
        return model, approx, mu_stack, sigma_stack, tau_samples, model.obs.observations
