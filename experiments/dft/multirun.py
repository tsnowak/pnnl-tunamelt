import os
import sys
from copy import deepcopy
from typing import List, Dict, OrderedDict
import json
from datetime import datetime
from pathlib import Path
from pprint import pprint
from itertools import product
from turbx import REPO_PATH
from turbx.filter import common, dft
from turbx.run import run
from turbx.data import DataLoader, Dataset
from multiprocessing import pool

# estimated run time in seconds
# calculated experimentally with few batches
tox = 15 * 60

# create a list of parameters dictionaries from every combination of parameter values given in params.json
def generate_param_batches(param_list: Dict) -> List:
    # pull out list of values for each variable
    # create list of every combination of variables
    # assign each value in [filter{param{value}}, ...]
    all_params = []
    batch_key = OrderedDict()
    for k, v in param_list.items():
        batch_key[k] = OrderedDict()
        for v_k, v_v in v.items():
            # create reference dict to use in all_params_dict
            batch_key[k][v_k] = None
            all_params.append(v_v)

    # convert all_params from unlabeled List(List) to List(Dict) where Dict is of shape batch_key
    all_params = list(product(*all_params))
    all_params_dict = []
    iter = 0
    for batch in all_params:
        i = 0
        batch_dict = deepcopy(batch_key)
        for k, v in batch_key.items():
            for v_k, v_v in v.items():
                batch_dict[k][v_k] = batch[i]
                i += 1
        iter += 1
        out = deepcopy(batch_dict)
        all_params_dict.append(out)

    estimated_run_time = len(all_params) * tox
    print(
        f"Assuming each run takes {tox/60.} minutes, the estimated run time is {estimated_run_time/(60.*60.)} hours."
    )
    return all_params_dict


# load and generate params List(Dict)
params_path = f"{REPO_PATH}/experiments/dft/params.json"
with open(params_path, "r") as f:
    params_list = json.load(f)
param_batches = generate_param_batches(params_list)

# create output folder structure
date_time = datetime.now()
ymd = date_time.strftime("%Y-%m-%d")
hms = date_time.strftime("%H-%M-%S")
run_name = str(Path(__file__).parent.name)
run_path = f"{REPO_PATH}/experiments/{run_name}/outputs/{ymd}/{hms}"
os.makedirs(run_path, exist_ok=True)

# create the dataset - do this once, do dataloader.reset() each batch
data_path = f"{REPO_PATH}/data/mp4"
label_path = f"{REPO_PATH}/data/labels"
dataloader = DataLoader(Dataset(videos=data_path, labels=label_path))

# run filters on every video for each param combination
for itr, params in enumerate(param_batches):

    fps = 10
    frame_delay = 1.0 / fps

    # initialize filters given params
    mean_filter = common.MeanFilter(fps=fps, params=params["mean_filter"])
    turbine_filter = dft.DFTFilter(fps=fps, params=params["turbine_filter"])
    denoise_filter = common.GaussianBlurDenoiseFilter(
        fps=fps, params=params["denoise_filter"]
    )
    intensity_filter = common.IntensityFilter(
        fps=fps, params=params["intensity_filter"]
    )
    contour_filter = common.ContourFilter(params=params["contour_filter"])

    # define the name and order of filters to apply
    filters = OrderedDict(
        [
            ("original", None),
            ("mean_filter", mean_filter),
            ("turbine_filter", turbine_filter),
            ("denoise_filter", denoise_filter),
            ("intensity_filter", intensity_filter),
            ("contour_filter", contour_filter),
        ]
    )

    # set self_idx to 0 to restart runs through dataset
    dataloader.reset()

    # run the filters on every video in the dataset; or on N=<max_vid_itrs> videos (for testing)
    run(
        itr,
        filters,
        params,
        dataloader,
        run_path,
        max_vid_itrs=100000,
    )
