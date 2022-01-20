"""
Code to preprocess spike trains before feeding into model
"""

########################################
# ___                            _   
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   
########################################
import numpy as np

def preprocess_single_taste(spike_array, time_lims, bin_width, data_transform):
    """
    spike_array : trials x neurons x time
    time_lims : Limits to cut spike_array according to
    bin_width: Width of bins to create spike counts
    data_transform : Data-type to return {actual, shuffled, simulated}

    ** Note, it may be useful to use x-arrays here to keep track of coordinates
    """

    accepted_transforms = ['shuffled','simulated','None',None]
    if data_transform not in accepted_transforms:
        raise Exception(f'data_transform must be of type {accepted_transforms}')

    ##################################################
    ## Create shuffled data
    ##################################################
    # Shuffle neurons across trials FOR SAME TASTE

    if data_transform == 'shuffled':
        transformed_dat = np.array([np.random.permutation(neuron) \
                    for neuron in np.swapaxes(spike_array,1,0)])
        transformed_dat = np.swapaxes(transformed_dat,0,1)

    ##################################################
    ## Create simulated data 
    ##################################################
    # Inhomogeneous poisson process using mean firing rates

    elif data_transform == 'simulated':
        mean_firing = np.mean(spike_array,axis=0)

        # Simulate spikes
        transformed_dat = np.array(\
                [np.random.random(mean_firing.shape) < mean_firing \
                for trial in range(spike_array.shape[0])])*1

    ##################################################
    ## Null Transform Case 
    ##################################################
    elif data_transform == None or data_transform == 'None':
        transformed_dat = spike_array

    ##################################################
    ## Bin Data 
    ##################################################
    spike_binned = \
            np.sum(transformed_dat[...,time_lims[0]:time_lims[1]].\
            reshape(*spike_array.shape[:-1],-1,bin_width),axis=-1)
    spike_binned = np.vectorize(np.int)(spike_binned)

    return spike_binned

def preprocess_all_taste():
    """
    spike_array : trials x neurons x time
    """
    accepted_transforms = ['shuffled','simulated','None',None]
    if data_transform not in accepted_transforms:
        raise Exception(f'data_transform must be of type {accepted_transforms}')

    ##################################################
    ## Create shuffled data
    ##################################################
    # Shuffle neurons across trials FOR SAME TASTE

    if data_transform == 'shuffled':
        transformed_dat = np.array([np.random.permutation(neuron) \
                    for neuron in np.swapaxes(spike_array,1,0)])
        transformed_dat = np.swapaxes(transformed_dat,0,1)

    ##################################################
    ## Create simulated data 
    ##################################################
    # Inhomogeneous poisson process using mean firing rates

    elif data_transform == 'simulated':
        mean_firing = np.mean(spike_array,axis=0)

        # Simulate spikes
        transformed_dat = np.array(\
                [np.random.random(mean_firing.shape) < mean_firing \
                for trial in range(spike_array.shape[0])])*1

    ##################################################
    ## Null Transform Case 
    ##################################################
    elif data_transform is None:
        transformed_dat = spike_array

    ##################################################
    ## Bin Data 
    ##################################################
    this_dat_binned = \
            np.sum(taste_dat[...,time_lims[0]:time_lims[1]].\
            reshape(*taste_dat.shape[:-1],-1,bin_width),axis=-1)
    this_dat_binned = np.vectorize(np.int)(this_dat_binned)

    return spike_binned
