"""
PyMC3 Blackbox Variational Inference implementation
of Poisson Likelihood Changepoint for spike trains.
- Changepoint distributions are shared across all tastes
- Each taste has it's own emission matrix
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
import sys
import pymc3 as pm
import theano.tensor as tt
import json
import tables

import numpy as np
import pickle
import argparse

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
sys.path.append('/media/bigdata/firing_space_plot/changepoint_mcmc')
from ephys_data import ephys_data
import visualize
import poisson_all_tastes_changepoint_model as changepoint 

############################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
############################################################

params_file_path = '/media/bigdata/firing_space_plot/changepoint_mcmc/fit_params.json'

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description = 'Script to fit changepoint model')
parser.add_argument('dir_name',  help = 'Directory containing data files')
parser.add_argument('states', type = int, help = 'Number of States to fit')
parser.add_argument("--good", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Use only good neurons")
parser.add_argument("--simulate", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Generate simulates AND shuffled fits")

args = parser.parse_args()
data_dir = args.dir_name 
good_nrn_bool = args.good
simulate_bool = args.simulate

#data_dir = '/media/NotVeryBig/for_you_to_play/file3'
#data_dir = '/media/bigdata/Abuzar_Data/AM35/AM35_4Tastes_201230_115322/'
#states = 4
#good_nrn_bool = True
#simulate_bool = True

dat = ephys_data(data_dir)

#dat.firing_rate_params = dat.default_firing_params

dat.get_unit_descriptors()
dat.get_spikes()
dat.check_laser()
#dat.get_firing_rates()
#dat.default_stft_params['max_freq'] = 50


if dat.laser_exists:
    dat.separate_laser_spikes()
    ##############################
    # ____    _    ____  
    #| __ )  / \  |  _ \ 
    #|  _ \ / _ \ | | | |
    #| |_) / ___ \| |_| |
    #|____/_/   \_\____/ 
    ##############################
                        
    taste_dat = np.array(dat.off_spikes)
else:
    taste_dat = dat.spikes

taste_dat = np.array(taste_dat)

if good_nrn_bool:
    print("Attempting to use good neurons")

    with tables.open_file(dat.hdf5_path,'r') as h5:
        good_nrn_bool_list = h5.get_node('/', 'selected_changepoint_nrns')[:]

    good_nrn_inds = np.where(good_nrn_bool_list)[0]
    taste_dat = taste_dat[:,:,good_nrn_inds]

##########
# PARAMS 
##########
states = int(args.states)

with open(params_file_path, 'r') as file:
    params_dict = json.load(file)

for key,val in params_dict.items():
    globals()[key] = val

#time_lims = [1500,4000]
#bin_width = 10
#fit = 40000
#samples = 20000

# Create dirs and names
model_save_dir = changepoint.get_model_save_dir(data_dir, states)
#model_save_dir = os.path.join(data_dir,'saved_models',f'vi_{states}_states')
#model_name = f'vi_{states}_states_{fit}fit_'\
#        f'time{time_lims[0]}_{time_lims[1]}_bin{bin_width}'

if good_nrn_bool:
    suffix = '_type_good'
else:
    suffix = '_type_reg'

model_name = changepoint.get_model_name(\
        states,fit,time_lims,bin_width, 'actual') + suffix
#model_dump_path = os.path.join(model_save_dir,f'dump_{model_name}.pkl')
model_dump_path = changepoint.get_model_dump_path(model_name,model_save_dir)

if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

##########
# Bin Data
##########
this_dat_binned = \
        np.sum(taste_dat[...,time_lims[0]:time_lims[1]].\
        reshape(*taste_dat.shape[:-1],-1,bin_width),axis=-1)
this_dat_binned = np.vectorize(np.int)(this_dat_binned)

########################################
# ___        __                              
#|_ _|_ __  / _| ___ _ __ ___ _ __   ___ ___ 
# | || '_ \| |_ / _ \ '__/ _ \ '_ \ / __/ _ \
# | || | | |  _|  __/ | |  __/ | | | (_|  __/
#|___|_| |_|_|  \___|_|  \___|_| |_|\___\___|
########################################
if not os.path.exists(model_dump_path):
    model = changepoint.create_changepoint_model(
                spike_array = this_dat_binned,
                states = states,
                fit = fit,
                samples = samples)
    
    # If the unnecessarily detailed model name exists
    # It will be loaded without running the inference
    # Otherwise model will be fit and saved

    changepoint.run_inference(model = model, 
                                fit = fit, 
                                samples = samples, 
                                #unbinned_array = taste_dat, 
                                model_save_dir = model_save_dir, 
                                model_name = model_name)

################################################################################
################################################################################

################################################
# ____  _                 _       _           _ 
#/ ___|(_)_ __ ___  _   _| | __ _| |_ ___  __| |
#\___ \| | '_ ` _ \| | | | |/ _` | __/ _ \/ _` |
# ___) | | | | | | | |_| | | (_| | ||  __/ (_| |
#|____/|_|_| |_| |_|\__,_|_|\__,_|\__\___|\__,_|
################################################                                               

if simulate_bool:

    print("Attempting simulated and shuffled fits")
    model_name_list = [changepoint.get_model_name(\
                            states,fit,time_lims,bin_width,this_type) \
                            for this_type in ['shuffle','simulate']]
    model_name_list = [x+suffix for x in model_name_list]

    model_dump_path_list =[\
            changepoint.get_model_dump_path(this_model_name,model_save_dir)
            for this_model_name in model_name_list]

    ##################################################
    ## Create shuffled data
    ##################################################
    # Shuffle neurons across trials FOR SAME TASTE
    shuffled_dat = np.array([np.random.permutation(neuron) \
                for neuron in np.swapaxes(taste_dat,2,0)])
    shuffled_dat = np.swapaxes(shuffled_dat,0,2)

    ##########
    # Bin Data
    ##########
    shuffled_dat_binned = \
            np.sum(shuffled_dat[...,time_lims[0]:time_lims[1]].\
            reshape(*shuffled_dat.shape[:-1],-1,bin_width),axis=-1)
    shuffled_dat_binned = np.vectorize(np.int)(shuffled_dat_binned)

    ##################################################
    ## Create simulated data 
    ##################################################
    # Inhomogeneous poisson process using mean firing rates

    mean_firing = np.mean(taste_dat,axis=1)

    # Simulate spikes
    simulated_spike_array = np.array(\
            [np.random.random(mean_firing.shape) < mean_firing \
            for trial in range(shuffled_dat_binned.shape[1])])*1
    simulated_spike_array = simulated_spike_array.swapaxes(0,1)
    simulated_dat_binned = \
            np.sum(simulated_spike_array[...,time_lims[0]:time_lims[1]].\
            reshape(*simulated_spike_array.shape[:-1],-1,bin_width),axis=-1)
    simulated_dat_binned = np.vectorize(np.int)(simulated_dat_binned)

    ########################################
    # ___        __                              
    #|_ _|_ __  / _| ___ _ __ ___ _ __   ___ ___ 
    # | || '_ \| |_ / _ \ '__/ _ \ '_ \ / __/ _ \
    # | || | | |  _|  __/ | |  __/ | | | (_|  __/
    #|___|_| |_|_|  \___|_|  \___|_| |_|\___\___|
    ########################################
    #if not all([os.path.exists(x) for x in model_dump_path_list]):

    model_kwargs = {'states':states,'fit':fit,'samples':samples}
    if not os.path.exists(model_dump_path_list[0]):
        shuffle_model = changepoint.create_changepoint_model(\
                        spike_array = shuffled_dat_binned, **model_kwargs)
        changepoint.run_inference(model = shuffle_model, 
                                    fit = fit, 
                                    samples = samples, 
                                    #unbinned_array = shuffled_dat, 
                                    model_save_dir = model_save_dir, 
                                    model_name = model_name_list[0])

    if not os.path.exists(model_dump_path_list[1]):
        simulate_model = changepoint.create_changepoint_model(\
                        spike_array = simulated_dat_binned, **model_kwargs)
        changepoint.run_inference(model = simulate_model, 
                                    fit = fit, 
                                    samples = samples, 
                                    #unbinned_array = simulated_spike_array,
                                    model_save_dir = model_save_dir, 
                                    model_name = model_name_list[1])
