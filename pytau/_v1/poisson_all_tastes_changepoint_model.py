"""
PyMC3 Blackbox Variational Inference implementation
of Poisson Likelihood Changepoint for spike trains.
- Changepoint distributions are shared across all tastes
- Each taste has it's own emission matrix

CODE TO GENERATE MODEL GIVEN SPIKE ARRAY
"""

########################################
# ___                            _   
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   
########################################
import os
import pickle
import scipy.stats as stats
import pymc3 as pm
import theano.tensor as tt
import numpy as np

########################################
# _   _                 _             
#| \ | | __ _ _ __ ___ (_)_ __   __ _ 
#|  \| |/ _` | '_ ` _ \| | '_ \ / _` |
#| |\  | (_| | | | | | | | | | | (_| |
#|_| \_|\__,_|_| |_| |_|_|_| |_|\__, |
#                               |___/ 
########################################
# Since naming is used to save and load models
# it must be consistent across files
# This section provides functions to name models
# and model_save_path's

def get_model_save_dir(data_dir, states):
    return os.path.join(data_dir,'saved_models',f'vi_{states}_states')

def get_model_name(states,fit,time_lims,bin_width, model_type):
    model_name = f'vi_{states}states_{fit}fit_'\
        f'{time_lims[0]}_{time_lims[1]}time_{bin_width}bin'
    if model_type == 'shuffle':
        model_name = 'shuffle_' + model_name
    if model_type == 'simulate':
        model_name = 'simulate_' + model_name
    return model_name

def get_model_dump_path(model_name, model_save_dir):
    model_path = os.path.join(model_save_dir,f'{model_name}.pkl')
    return model_path

############################################################
#  ____                _         __  __           _      _ 
# / ___|_ __ ___  __ _| |_ ___  |  \/  | ___   __| | ___| |
#| |   | '__/ _ \/ _` | __/ _ \ | |\/| |/ _ \ / _` |/ _ \ |
#| |___| | |  __/ (_| | ||  __/ | |  | | (_) | (_| |  __/ |
# \____|_|  \___|\__,_|\__\___| |_|  |_|\___/ \__,_|\___|_|
############################################################

def create_changepoint_model(
                        spike_array,
                        states,
                        fit,
                        samples):

    """
    spike_array :: Shape : tastes, trials, neurons, time_bins
    states :: number of states to include in the model 
    fit :: number of iterations to fit for
    samples :: number of samples to generate from the fit model
    """
    # If model already doesn't exist, then create new one
    #spike_array = this_dat_binned
    # Unroll arrays along taste axis
    spike_array_long = np.reshape(spike_array,(-1,*spike_array.shape[-2:]))

    # Find mean firing for initial values
    tastes = spike_array.shape[0]
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
    idx_range = idx.max() - idx.min()
    even_switches = np.linspace(0,idx.max(),states+1)
    even_switches_normal = even_switches/np.max(even_switches)

    taste_label = np.repeat([0,1,2,3],30)
    trial_num = array_idx.shape[0]

    # Being constructing model
    with pm.Model() as model:

        # Hierarchical firing rates
        # Refer to model diagram
        # Mean firing rate of neuron AT ALL TIMES
        lambda_nrn = pm.Exponential('lambda_nrn',
                                    1/mean_nrn_vals, 
                                    shape = (mean_vals.shape[-1]))
        # Priors for each state, derived from each neuron
        # Mean firing rate of neuron IN EACH STATE (averaged across tastes)
        lambda_state = pm.Exponential('lambda_state',
                                        lambda_nrn, 
                                        shape = (mean_vals.shape[1:]))
        # Mean firing rate of neuron PER STATE PER TASTE
        lambda_latent = pm.Exponential('lambda', 
                                        lambda_state[np.newaxis,:,:], 
                                        testval = mean_vals, 
                                        shape = (mean_vals.shape))

        # Changepoint time variable
        # INDEPENDENT TAU FOR EVERY TRIAL
        a = pm.HalfNormal('a_tau', 3., shape = states - 1)
        b = pm.HalfNormal('b_tau', 3., shape = states - 1)

        # Stack produces states x trials --> That gets transposed 
        # to trials x states and gets sorted along states (axis=-1)
        # Sort should work the same way as the Ordered transform --> 
        # see rv_sort_test.ipynb
        tau_latent = pm.Beta('tau_latent', a, b, 
                               shape = (trial_num, states-1),
                               testval = \
                                       tt.tile(even_switches_normal[1:(states)],
                                           (array_idx.shape[0],1))).sort(axis=-1)
               
        tau = pm.Deterministic('tau', 
                idx.min() + (idx.max() - idx.min()) * tau_latent)

        # Sigmoing to create transitions based off tau
        # Hardcoded 3-5 states
        weight_1_stack = tt.nnet.sigmoid(\
                array_idx - tau[:,0][...,np.newaxis,np.newaxis])
        weight_2_stack = tt.nnet.sigmoid(\
                array_idx - tau[:,1][...,np.newaxis,np.newaxis])
        if states > 3:
            weight_3_stack = tt.nnet.sigmoid(\
                    array_idx - tau[:,2][...,np.newaxis,np.newaxis])
        if states > 4:
            weight_4_stack = tt.nnet.sigmoid(\
                    array_idx - tau[:,3][...,np.newaxis,np.newaxis])

        # Generate firing rates from lambda and sigmoid weights
        if states == 3:
            # 3 states
            lambda_ = np.multiply(1 - weight_1_stack, 
                            lambda_latent[taste_label,0][:,:,np.newaxis]) + \
                    np.multiply(weight_1_stack * (1 - weight_2_stack), 
                            lambda_latent[taste_label][:,1][:,:,np.newaxis]) + \
                    np.multiply(weight_2_stack, 
                                lambda_latent[taste_label,2][:,:,np.newaxis])

        elif states == 4:
            # 4 states
            lambda_ = np.multiply(1 - weight_1_stack, 
                            lambda_latent[taste_label,0][:,:,np.newaxis]) + \
                    np.multiply(weight_1_stack * (1 - weight_2_stack), 
                            lambda_latent[taste_label][:,1][:,:,np.newaxis]) + \
                    np.multiply(weight_2_stack * (1 - weight_3_stack), 
                            lambda_latent[taste_label][:,2][:,:,np.newaxis]) + \
                    np.multiply(weight_3_stack, 
                                lambda_latent[taste_label,3][:,:,np.newaxis])

        elif states == 5:
            # 5 states
            lambda_ = np.multiply(1 - weight_1_stack, 
                            lambda_latent[taste_label,0][:,:,np.newaxis]) + \
                    np.multiply(weight_1_stack * (1 - weight_2_stack), 
                            lambda_latent[taste_label][:,1][:,:,np.newaxis]) + \
                    np.multiply(weight_2_stack * (1 - weight_3_stack), 
                            lambda_latent[taste_label][:,2][:,:,np.newaxis]) +\
                    np.multiply(weight_3_stack * (1 - weight_4_stack), 
                            lambda_latent[taste_label][:,3][:,:,np.newaxis])+ \
                    np.multiply(weight_4_stack, 
                            lambda_latent[taste_label,4][:,:,np.newaxis])
            
        # Add observations
        observation = pm.Poisson("obs", lambda_, observed=spike_array_long)

    return model

######################################################################
#|  _ \ _   _ _ __   |_ _|_ __  / _| ___ _ __ ___ _ __   ___ ___ 
#| |_) | | | | '_ \   | || '_ \| |_ / _ \ '__/ _ \ '_ \ / __/ _ \
#|  _ <| |_| | | | |  | || | | |  _|  __/ | |  __/ | | | (_|  __/
#|_| \_\\__,_|_| |_| |___|_| |_|_|  \___|_|  \___|_| |_|\___\___|
######################################################################
# If the unnecessarily detailed model name exists
# It will be loaded without running the inference
# Otherwise model will be fit and saved
def run_inference(model,fit,samples, model_save_dir, model_name):
    """
    model : PyMC3 changepoint model as defined above
    fit : number of iterations to fit
    samples : number of samples to generate from fitted model
    model_save_dir : parent directory of where to save model
    model_name : name for the model
    """

    #model_dump_path = os.path.join(model_save_dir,f'dump_{model_name}.pkl')
    model_dump_path = get_model_dump_path(model_name,model_save_dir) 
    #trace_dump_path = os.path.join(model_save_dir,f'traces_{model_name}.pkl')

    if os.path.exists(model_dump_path):
        print('Trace loaded from cache')
        with open(model_dump_path, 'rb') as buff:
            data = pickle.load(buff)
        model = data['model']
        approx = data['approx']
        # Remove pickled data to conserve memory
        del data
        # Recreate samples
        trace = approx.sample(draws=samples)
    else:
        with model:
            inference = pm.ADVI('full-rank')
            approx = pm.fit(n=fit, method=inference)
            trace = approx.sample(draws=samples)

        # Extract relevant variables from trace
        lambda_stack = trace['lambda'].swapaxes(0,1)
        tau_samples = trace['tau']

        print('Dumping trace to cache')
        with open(model_dump_path, 'wb') as buff:
            pickle.dump({'model' : model,
                        'approx' : approx,
                        'lambda' : lambda_stack,
                        'tau' : tau_samples,
                        'data' : model.obs.observations}, buff)
