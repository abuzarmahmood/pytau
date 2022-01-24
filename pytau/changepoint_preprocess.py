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
from numpy.random import permutation


def preprocess_single_taste(spike_array,
                            time_lims,
                            bin_width,
                            data_transform):
    """Preprocess array containing trials for all tastes (in blocks) concatenated

    ** Note, it may be useful to use x-arrays here to keep track of coordinates

    Args:
        spike_array (3D Numpy array): trials x neurons x time
        time_lims (List/Tuple/Numpy Array): 2-element object indicating limits of array
        bin_width (int): Width to use for binning
        data_transform (str): Data-type to return {actual, shuffled, simulated}

    Raises:
        Exception: If transforms do not belong to ['shuffled','simulated','None',None]

    Returns:
        (3D Numpy Array): Of processed data
    """

    accepted_transforms = ['shuffled', 'simulated', 'None', None]
    if data_transform not in accepted_transforms:
        raise Exception(
            f'data_transform must be of type {accepted_transforms}')

    ##################################################
    # Create shuffled data
    ##################################################
    # Shuffle neurons across trials FOR SAME TASTE

    if data_transform == 'shuffled':
        transformed_dat = np.array([permutation(neuron)
                                    for neuron in np.swapaxes(spike_array, 1, 0)])
        transformed_dat = np.swapaxes(transformed_dat, 0, 1)

    ##################################################
    # Create simulated data
    ##################################################
    # Inhomogeneous poisson process using mean firing rates

    elif data_transform == 'simulated':
        mean_firing = np.mean(spike_array, axis=0)

        # Simulate spikes
        transformed_dat = np.array(
            [np.random.random(mean_firing.shape) < mean_firing
             for trial in range(spike_array.shape[0])])*1

    ##################################################
    # Null Transform Case
    ##################################################
    elif data_transform in (None, 'None'):
        transformed_dat = spike_array

    ##################################################
    # Bin Data
    ##################################################
    spike_binned = \
        np.sum(transformed_dat[..., time_lims[0]:time_lims[1]].
               reshape(*spike_array.shape[:-1], -1, bin_width), axis=-1)
    spike_binned = np.vectorize(np.int)(spike_binned)

    return spike_binned


def preprocess_all_taste(spike_array,
                         time_lims,
                         bin_width,
                         data_transform):
    """Preprocess array containing trials for all tastes (in blocks) concatenated

    Args:
        spike_array (4D Numpy Array): Taste x Trials x Neurons x Time
        time_lims (List/Tuple/Numpy Array): 2-element object indicating limits of array
        bin_width (int): Width to use for binning
        data_transform (str): Data-type to return {actual, shuffled, simulated}

    Raises:
        Exception: If transforms do not belong to ['shuffled','simulated','None',None]

    Returns:
        (4D Numpy Array): Of processed data
    """

    accepted_transforms = ['shuffled', 'simulated', 'None', None]
    if data_transform not in accepted_transforms:
        raise Exception(
            f'data_transform must be of type {accepted_transforms}')

    ##################################################
    # Create shuffled data
    ##################################################
    # Shuffle neurons across trials FOR SAME TASTE

    if data_transform == 'shuffled':
        transformed_dat = np.array([permutation(neuron)
                                    for neuron in np.swapaxes(spike_array, 1, 0)])
        transformed_dat = np.swapaxes(transformed_dat, 0, 1)

    ##################################################
    # Create simulated data
    ##################################################
    # Inhomogeneous poisson process using mean firing rates

    elif data_transform == 'simulated':
        mean_firing = np.mean(spike_array, axis=0)

        # Simulate spikes
        transformed_dat = np.array(
            [np.random.random(mean_firing.shape) < mean_firing
             for trial in range(spike_array.shape[0])])*1

    ##################################################
    # Null Transform Case
    ##################################################
    elif data_transform is None:
        transformed_dat = spike_array

    ##################################################
    # Bin Data
    ##################################################
    this_dat_binned = \
        np.sum(transformed_dat[..., time_lims[0]:time_lims[1]].
               reshape(*transformed_dat.shape[:-1], -1, bin_width), axis=-1)
    this_dat_binned = np.transformed_dat(np.int)(this_dat_binned)

    return this_dat_binned
