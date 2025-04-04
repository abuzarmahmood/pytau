"""
Fit models to given spike array
"""

import json
import os
import pickle
import shutil
import sys
import uuid
from datetime import date, datetime
from glob import glob

import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from tqdm import tqdm, trange

base_dir = "/media/bigdata/projects/pytau"
sys.path.append(base_dir)

import pymc3
import theano

from pytau import changepoint_model, changepoint_preprocess
from pytau.utils import EphysData

for ind in trange(len(quin_clust_flat)):
    data = quin_clust_flat[ind]
    model_parameters_keys = ["states", "fit", "samples", "model_kwargs"]
    model_parameters_values = [4, 80000, 20000, {"None": None}]
    preprocess_parameters_keys = ["time_lims", "bin_width", "data_transform"]
    preprocess_parameters_values = [[2000, 4000], 50, None]

    model_params = dict(zip(model_parameters_keys, model_parameters_values))
    preprocess_params = dict(zip(preprocess_parameters_keys, preprocess_parameters_values))

    preprocessor = changepoint_preprocess.preprocess_single_taste
    model_template = changepoint_model.single_taste_poisson
    inference_func = changepoint_model.advi_fit

    preprocessed_data = preprocessor(data, **preprocess_params)

    model = model_template(
        preprocessed_data, model_params["states"], **model_params["model_kwargs"]
    )

    temp_outs = inference_func(model, model_params["fit"], model_params["samples"])
    varnames = ["model", "approx", "lambda", "tau", "data"]
    inference_outs = dict(zip(varnames, temp_outs))

    tau_array = inference_outs["tau"]
    tau_hist_array = convert_to_hist_array(tau_array, bins)
    np.save(os.path.join(save_path, id_frame["fin_name"][ind]), tau_hist_array)
