## Import modules
base_dir = '/media/bigdata/projects/pytau'
import sys
sys.path.append(base_dir)
from pytau.changepoint_io import FitHandler
import pylab as plt
from pytau.utils import plotting

# Specify params for fit
model_parameters = dict(
    states=4,
    fit=40000,
    samples=20000,
    model_kwargs={'None': None},
        )

preprocess_parameters = dict(
    time_lims=[2000, 4000],
    bin_width=50,
    data_transform='None',  # Can also be 'spike_shuffled','trial_shuffled'
    )

FitHandler_kwargs = dict(
    data_dir='/path/to/data/directory',
    taste_num=[0, 1, 2, 3],
    region_name=['CA1'],  # Should match specification in info file
    laser_type=['off'],
    experiment_name=['pytau_test'],
    )

## Initialize handler, and feed paramters
handler = FitHandler(**FitHandler_kwargs)
handler.set_model_params(**model_parameters)
handler.set_preprocess_params(**preprocess_parameters)

# Perform inference and save output to model database
handler.run_inference()
handler.save_fit_output()

# Access fit results
# Directly from handler
inference_outs = handler.inference_outs
# infernece_outs contains following attributes
# model : Model structure
# approx : Fitted model
# lambda : Inferred firing rates for each state
# tau : Inferred changepoints
# data : Data used for inference

# Can also get path to pkl file from model database
from pytau.changepoint_io import DatabaseHandler
fit_database = DatabaseHandler()
fit_database.drop_duplicates()
fit_database.clear_mismatched_paths()

# Get fits for a particular experiment
dframe = fit_database.fit_database
wanted_exp_name = 'pytau_test'
wanted_frame = dframe.loc[dframe['exp.exp_name'] == wanted_exp_name] 
# Pull out a single data_directory
pkl_path = wanted_frame['exp.save_path'].iloc[0]

## Information saved in model database 
# preprocess.time_lims
# preprocess.bin_width
# preprocess.data_transform
# preprocess.preprocessor_name
# model.states
# model.fit
# model.samples
# model.model_kwargs
# model.model_template_name
# model.inference_func_name
# data.data_dir
# data.basename
# data.animal_name
# data.session_date
# data.taste_num
# data.laser_type
# data.region_name
# exp.exp_name
# exp.model_id
# exp.save_path
# exp.fit_date
# module.pymc3_version
# module.theano_version

# From saved pkl file
from pytau.changepoint_analysis import PklHandler
this_handler = PklHandler(pkl_path)
# Can access following attributes
# Tau:
#   Raw Int tau : All tau samples in terms of indices of array given ==> this_handler.tau.raw_int_tau
#   Raw mode tau : Mode of samples in terms of indices of array given ==> this_handler.tau.raw_mode_tau
#   Scaled Tau : All tau samples scaled to stimulus delivery ==> this_handler.tau.scaled_tau
#   Int Scaled Tau : Integer values of "Scaled Tau" ==> this_handler.tau.scaled_int_tau
#   Mode Scale Tau : Mode of Int Scaled Tau ==> this_handler.tau.scaled_mode_tau
# Firing:
#   Raw spikes : Pulled using EphysData ==> this_handler.firing.raw_spikes 
#   Mean firing rate per state : this_handler.firing.state_firing
#   Snippets around each transition : this_handler.firing.transition_snips
#   Significance of changes in state firing : this_handler.firing.anova_p_val_array
#   Significance of changes in firing across transitions : this_handler.firing.pairwise_p_val_array
# Metadata
this_handler.pretty_metadata

# Plotting
fit_model = this_handler.data['model_data']['approx']
spike_train = this_handler.firing.raw_spikes
scaled_mode_tau = this_handler.tau.scaled_mode_tau

# Plot ELBO over iterations, should be flat by the end
fig, ax = plotting.plot_elbo_history(fit_model)
plt.show()

# Overlay raster plot with states
fig, ax = plotting.plot_changepoint_raster(
    spike_train, scaled_mode_tau, [1500, 4000])
plt.show()

# Overview of changepoint positions
fig, ax = plotting.plot_changepoint_overview(scaled_mode_tau, [1500, 4000])
plt.show()

# Aligned spiking
fig, ax = plotting.plot_aligned_state_firing(spike_train, scaled_mode_tau, 300)
plt.show()

# Plot mean firing rates per state
fig, ax = plotting.plot_state_firing_rates(spike_train, scaled_mode_tau)
plt.show()
