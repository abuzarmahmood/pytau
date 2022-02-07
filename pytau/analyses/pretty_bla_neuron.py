import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
sys.path.append('/media/bigdata/firing_space_plot/changepoint_mcmc/v2')
from ephys_data import ephys_data
import visualize as vz
from changepoint_io import database_handler
from changepoint_analysis import pkl_handler
from tqdm import tqdm, trange
import pandas as pd
import matplotlib.pyplot as plt
import os
#from joblib import Parallel, cpu_count, delayed

fit_database = database_handler()
fit_database.drop_duplicates()
fit_database.clear_mismatched_paths()

wanted_exp_name = 'pretty_bla_change'
dframe = fit_database.fit_database
wanted_frame = dframe.loc[dframe['exp.exp_name'] == wanted_exp_name] 

file_list = list(wanted_frame['exp.save_path'])

########################################
## Extract spikes and state_firing
########################################
#def parallelize(func, iterator):
#    return Parallel(n_jobs = cpu_count()-2)\
#            (delayed(func)(this_iter) for this_iter in tqdm(iterator))
#
#def return_wanted_params(ind):
#    file_path = file_list[ind]
#    model_dat = pkl_handler(file_path)
#    spikes = model_dat.firing.raw_spikes
#    state_firing = model_dat.firing.state_firing.swapaxes(1,2)
#    scaled_tau = model_dat.tau.scaled_mode_tau
#    return spikes, state_firing, scaled_tau
#
#outs = parallelize(return_wanted_params, range(len(file_list)))

all_raw_spikes = [] # trials x neurons x time
all_state_firing = [] # trials x neurons x states
all_scaled_tau = []

#ind = 0
for ind in trange(len(file_list)):
    file_path = file_list[ind]
    model_dat = pkl_handler(file_path)
    all_raw_spikes.append(model_dat.firing.raw_spikes)
    all_state_firing.append(model_dat.firing.state_firing.swapaxes(1,2))
    all_scaled_tau.append(model_dat.tau.scaled_mode_tau)

# Separate out by neuron
nrn_raw_spikes = []
nrn_state_firing = []
nrn_scaled_tau = []
for spikes, firing, tau in \
        tqdm(zip(all_raw_spikes, all_state_firing, all_scaled_tau)):
    this_spikes = spikes.swapaxes(0,1)
    this_firing = firing.swapaxes(0,1)
    for nrn_spikes, nrn_firing in zip(this_spikes, this_firing):
        nrn_raw_spikes.append(nrn_spikes)
        nrn_state_firing.append(nrn_firing)
        nrn_scaled_tau.append(tau)

trials, states = zip(*[x.shape for x in nrn_state_firing])

nrn_frame = pd.DataFrame({
                'trials' : trials,
                'states' : states,
                'raw_spikes' : nrn_raw_spikes,
                'state_firing' : nrn_state_firing,
                'tau' : nrn_scaled_tau})

# Cleaning
nrn_frame = nrn_frame.loc[nrn_frame.trials == 30]

state_frame_list = [x[1] for x in list(nrn_frame.groupby('states'))]

this_state_frame = state_frame_list[0]
this_state_firing = np.stack(this_state_frame.state_firing)
this_state_diff = np.diff(this_state_firing)[...,0]
this_raw_spikes = np.stack(this_state_frame.raw_spikes)
this_scaled_tau = np.stack(this_state_frame.tau)

mean_state_diff = np.mean(this_state_diff, axis=1)
mean_scaled_tau = np.mean(this_scaled_tau, axis = 1)

plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/pretty_bla_firing'
plt.scatter(mean_scaled_tau, mean_state_diff)
plt.savefig(os.path.join(plot_dir, 'diff_tau_scatter'))
plt.close()

time_lims = [2600, 3000]
selected_inds = np.where(np.logical_and(mean_scaled_tau>=time_lims[0], 
                        mean_scaled_tau<=time_lims[1]))[0]
selected_mean_diff = mean_state_diff[selected_inds]
max_diff_inds = np.argsort(np.abs(selected_mean_diff))[-10:]

fin_inds = selected_inds[max_diff_inds]
fin_spikes = this_raw_spikes[fin_inds]
fin_tau = this_scaled_tau[fin_inds]

plot_lims = [1500,4500]
stim_t = 2000
for this_ind, this_spikes, this_tau in zip(fin_inds, fin_spikes, fin_tau):
    fig, ax = plt.subplots()
    ax = vz.raster(ax,this_spikes, marker = "|")
    ax.scatter(this_tau, np.arange(len(this_tau)), color = 'red', alpha = 0.7)
    ax.set_xlim(plot_lims)
    ax.axvline(stim_t, linewidth = 2, color = 'red', alpha = 0.7)
    ax.set_ylabel('Sorted Trials')
    ax.set_xlabel('Stim t = 2000 ms')
    fig.savefig(os.path.join(plot_dir, f'nrn{this_ind}_spike_train'))
    plt.close(fig)

    trial_order = np.argsort(this_tau)
    sorted_tau = this_tau[trial_order]
    sorted_spikes = this_spikes[trial_order]
    fig, ax = plt.subplots()
    ax = vz.raster(ax,sorted_spikes, marker = "|")
    ax.scatter(sorted_tau, np.arange(len(this_tau)), color = 'red', alpha = 0.7)
    ax.set_xlim(plot_lims)
    ax.axvline(stim_t, linewidth = 2, color = 'red', alpha = 0.7)
    ax.set_ylabel('Sorted Trials')
    ax.set_xlabel('Stim t = 2000 ms')
    fig.savefig(os.path.join(plot_dir, f'nrn{this_ind}_spike_train_sorted'))
    plt.close(fig)

########################################
## Increased step firing
########################################
max_increase_inds = np.argsort(mean_state_diff)[-10:]
inc_spikes = this_raw_spikes[max_increase_inds]
inc_tau =this_scaled_tau[max_increase_inds]

for this_ind, this_spikes, this_tau \
        in zip(max_increase_inds, inc_spikes, inc_tau):
    fig, ax = plt.subplots()
    ax = vz.raster(ax,this_spikes, marker = "|")
    ax.scatter(this_tau, np.arange(len(this_tau)), color = 'red', alpha = 0.7)
    ax.set_xlim(plot_lims)
    ax.axvline(stim_t, linewidth = 2, color = 'red', alpha = 0.7)
    ax.set_ylabel('Sorted Trials')
    ax.set_xlabel('Stim t = 2000 ms')
    fig.savefig(os.path.join(plot_dir, f'nrn{this_ind}_spike_train'))
    plt.close(fig)

    trial_order = np.argsort(this_tau)
    sorted_tau = this_tau[trial_order]
    sorted_spikes = this_spikes[trial_order]
    fig, ax = plt.subplots()
    ax = vz.raster(ax,sorted_spikes, marker = "|")
    ax.scatter(sorted_tau, np.arange(len(this_tau)), color = 'red', alpha = 0.7)
    ax.set_xlim(plot_lims)
    ax.axvline(stim_t, linewidth = 2, color = 'red', alpha = 0.7)
    ax.set_ylabel('Sorted Trials')
    ax.set_xlabel('Stim t = 2000 ms')
    fig.savefig(os.path.join(plot_dir, f'nrn{this_ind}_spike_train_sorted'))
    plt.close(fig)
