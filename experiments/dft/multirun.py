from typing import List, Dict
import json
from pprint import pprint
from itertools import product
from turbx import REPO_PATH
from multiprocessing import pool

# estimated run time in seconds
# calculated experimentally with few batches
tox = 5 * 60


def generate_param_batches(param_list: Dict) -> List:
    # pull out list of values for each variable
    # create list of every combination of variables
    # assign each value in [filter{param{value}}, ...]
    all_params = [v_v for _, v in param_list.items() for _, v_v in v.items()]
    all_params = list(product(*all_params))
    estimated_run_time = len(all_params) * tox
    print(
        f"Assuming each run takes {tox/60.} minutes, the estimated run time is {estimated_run_time/(60.*60.)} hours."
    )
    return []


# TODO: load params
params_path = f"{REPO_PATH}/experiments/dft/params.json"
with open(params_path, "r") as f:
    params_list = json.load(f)

param_batches = generate_param_batches(params_list)
