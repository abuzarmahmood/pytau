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
import seaborn as sns
from scipy.stats import zscore
import numpy as np

#def parallelize(func, iterator):
#    return Parallel(n_jobs = cpu_count()-2)\
#            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

fit_database = database_handler()
fit_database.drop_duplicates()
fit_database.clear_mismatched_paths()

dframe = fit_database.fit_database
wanted_exp_name = 'bla_transition_strength'
wanted_frame = dframe.loc[dframe['exp.exp_name'] == wanted_exp_name] 
#wanted_frame = wanted_frame.loc[wanted_frame['model.states'] == 4] 

file_list = list(wanted_frame['exp.save_path'])

#fin_elbo_list = []
#for ind in trange(len(file_list)):
#    file_path = file_list[ind]
#    model_dat = pkl_handler(file_path)
#    fin_elbo_list.append(-model_dat.data['model_data']['approx'].hist[-1])

def get_elbo(ind):
    file_path = file_list[ind]
    print(f'Reading file {file_path}')
    model_dat = pkl_handler(file_path)
    return -model_dat.data['model_data']['approx'].hist[-1]

#fin_elbo = parallelize(get_elbo, range(len(file_list)))
fin_elbo = [get_elbo(i) for i in trange(len(file_list))] 

wanted_frame['model_elbo'] = fin_elbo

# Analysis
wanted_columns = ['preprocess.data_transform', 'model.states', 
                'data.basename','data.animal_name',
                'data.session_date','model_elbo']
analysis_frame = wanted_frame[wanted_columns].reset_index(drop=True)
# Remove 2 Taste sessions
analysis_frame = analysis_frame[\
        ~analysis_frame['data.basename'].str.contains('2Tastes')]
# Remove simulated models
analysis_frame = analysis_frame[\
        ~analysis_frame['preprocess.data_transform'].str.contains('simulated')]

# Calculate elbo zscore and ranks
grouped_frame = list(analysis_frame.groupby('data.basename'))
for num, this_frame in grouped_frame:
    this_frame['zscore_elbo'] = zscore(this_frame['model_elbo'])
    this_frame['rank_elbo'] = np.argsort(this_frame['model_elbo'])
analysis_frame = pd.concat([x[1] for x in grouped_frame]).reset_index(drop=True)

# Plot all animals
g = sns.relplot(data = analysis_frame, 
            x = 'model.states', y = 'model_elbo', 
            hue = 'preprocess.data_transform', col = 'data.basename', 
            style = 'preprocess.data_transform',
            col_wrap = 4, kind = 'line', markers =True, 
            facet_kws={'sharey': False, 'sharex': True})
#plt.suptitle('ELBO per animal, per shuffle')
#plt.tight_layout()
plt.show()

# Scatter
g = sns.stripplot(data = analysis_frame,
            x = 'model.states', y = 'zscore_elbo',
            hue = 'preprocess.data_transform', 
            size = 10, alpha = 0.7) 
plt.suptitle('Zscored ELBO across datasets')
plt.show()

# Zscore average
g = sns.lineplot(data = analysis_frame,
            x = 'model.states', y = 'zscore_elbo', 
            style = 'preprocess.data_transform',
            hue = 'preprocess.data_transform', markers=True)
plt.suptitle('Zscored ELBO across datasets, mean +/- 95% CI')
plt.show()

#g = sns.lineplot(data = analysis_frame,
#            x = 'model.states', y = 'rank_elbo', 
#            style = 'preprocess.data_transform',
#            hue = 'preprocess.data_transform', markers=True)
#plt.show()
