import sys

from changepoint_io import database_handler
from ephys_data import ephys_data

sys.path.append("/media/bigdata/firing_space_plot/ephys_data")
sys.path.append("/media/bigdata/firing_space_plot/changepoint_mcmc/v2")

fit_database = database_handler()
fit_database.drop_duplicates()
fit_database.clear_mismatched_paths()
# fit_database.write_updated_database()
