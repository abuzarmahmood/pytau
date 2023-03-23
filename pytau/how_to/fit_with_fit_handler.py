## Import modules
from pytau.changepoint_io import FitHandler
base_dir = '/media/bigdata/projects/pytau'
sys.path.append(base_dir)

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

# From saved pkl file
from changepoint_analysis import PklHandler
this_handler = PklHandler('/path/to/pkl/file')
# Can access following attributes
# Raw Int tau : All tau samples in terms of indices of array given ==> this_handler.tau.raw_int_tau
# Raw mode tau : Mode of samples in terms of indices of array given ==> this_handler.tau.raw_mode_tau
# Scaled Tau : All tau samples scaled to stimulus delivery ==> this_handler.tau.scaled_tau
# Int Scaled Tau : Integer values of "Scaled Tau" ==> this_handler.tau.scaled_int_tau
# Mode Scale Tau : Mode of Int Scaled Tau ==> this_handler.tau.scaled_mode_tau

# Can also get path to pkl file from model database
from pytau.changepoint_io import DatabaseHandler
fit_database = DatabaseHandler()
fit_database.drop_duplicates()
fit_database.clear_mismatched_paths()

# Get fits for a particular experiment
dframe = fit_database.fit_database
wanted_exp_name = 'bla_population_elbo_repeat'
wanted_frame = dframe.loc[dframe['exp.exp_name'] == wanted_exp_name] 
# Pull out a single data_directory
data_dir = wanted_frame['data.data_dir'].iloc[0]

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
