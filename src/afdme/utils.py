import argparse
import json
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, OrderedDict, Tuple, TypeVar, Union, Tuple, Set
from pprint import pprint

import cv2
import numpy as np

from afdme import REPO_PATH, log

Shape = TypeVar("Shape")
DType = TypeVar("DType")


def load_multirun_args(multirun_exp_path: Union[str, Path], multirun_args: Dict):
    runs = OrderedDict()
    for param in multirun_args["params"]:
        run_args_name = Path(param).stem
        run_args_file = list(Path(multirun_exp_path).glob(f"*-{run_args_name}.json"))
        if not len(run_args_file) == 1:
            raise ValueError(f"Multiple matching run files found.\n{run_args_file}")
        run_args_file = run_args_file[0]
        run_args_id = str(run_args_file.name).split("-")[0]
        runs[run_args_id] = {
            "params_file": run_args_file,
            "params_id": run_args_name,
            "exp_path": f"{run_args_file.parent}/{run_args_id}",
        }

    return runs


def load_results_json(results_path: Union[str, Path]) -> Tuple[List[Dict], Set[Path]]:
    if not isinstance(results_path, Path):
        results_path = Path(results_path)
    assert results_path.exists()

    inference_results = []
    param_files = set()
    for f_name in Path(results_path).glob("**/*.results.json"):
        param_files.add(f_name)
        with open(f_name, "r") as f:
            params = json.load(f)
            inference_results.append(params)
    log.info(f"Loaded {len(param_files)} results files:\n")
    pprint(f"{param_files}")

    return inference_results, param_files


def create_exp_dirs(results_path) -> Path:
    Path(results_path).mkdir(exist_ok=True)
    date_time = datetime.now()
    ymd = date_time.strftime("%Y-%m-%d")
    hms = date_time.strftime("%H-%M-%S")
    run_path = Path(f"{results_path}/{ymd}/{hms}")
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def load_params(run_path, params) -> List:
    params_list = []
    if isinstance(params, str):
        params = [params]
    for i, params_path in enumerate(params):
        params_path = f"{REPO_PATH}/{params_path}"
        # load and copy jsons to run path for reference
        with open(params_path, "r") as f:
            params_list.append(json.load(f))
            shutil.copy(params_path, f"{str(run_path)}/{i}-{Path(params_path).name}")
    return params_list


# create a list of parameters dictionaries from every combination of parameter values given in params.json
def generate_param_batches(params: Dict) -> List:
    if isinstance(params, dict):
        # pull out list of values for each variable
        # create list of every combination of variables
        # assign each value in [filter{param{value}}, ...]
        all_params = []
        batch_key = OrderedDict()
        for k, v in params.items():
            batch_key[k] = OrderedDict()
            for v_k, v_v in v.items():
                # create reference dict to use in all_params_dict
                batch_key[k][v_k] = None
                all_params.append(v_v)

        # convert all_params from unlabeled List(List) to List(Dict) where Dict is of shape batch_key
        try:
            all_params = list(product(*all_params))
        except TypeError:
            all_params = [all_params]
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
        return all_params_dict
    else:
        raise TypeError("Params must be a dictionary")


class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)


def generate_sinusoid(freq: int, fps: int, shape: Tuple, length=float):
    """
    Create a waveform
    """

    u, v = shape
    t = np.linspace(0, length, int(fps * length))
    # 8-bit-valued waveform
    waveform = (np.sin(freq * (2 * np.pi) * t) / 2.0) + 0.5

    log.debug(f"\nSinusoid shape: {waveform.shape}")

    waveform_video = np.tile(waveform, (u, v, 1))

    log.debug(f"\nSinusoid video shape: {waveform_video.shape}")

    assert waveform_video.shape == (u, v, len(t)), "Waveform video incorrectly shaped"

    return waveform_video


def generate_sinusoid_tile(freqs, element_shape, n_frames):
    """
    Returns numpy array representing a video of
    multi-frequency image tiles
    """
    # hold fps constant
    fps = 10
    length = n_frames / fps

    waveforms = []
    for freq in freqs:
        waveforms.append(
            generate_sinusoid(freq=freq, fps=fps, shape=element_shape, length=length)
        )

    output = np.concatenate(waveforms, axis=1)
    return output, fps, length


def crop_polygon(img, pts):
    """
    crop polygon of image, and return with black background
    """
    # (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = img[y : y + h, x : x + w].copy()

    # (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    # (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    return dst


def standard_parser():
    parser = DefaultHelpParser(description="Input path of video to filter")
    parser.add_argument(
        "--data_dir",
        nargs="?",
        required=True,
        type=str,
        help="Data directory that can be used to reference files without supplying a full path.",
    )
    # name = "2010-09-08_081500_HF_S021"
    # name = "2010-09-09_020001_HF_S013"
    parser.add_argument(
        "--file_name",
        nargs="?",
        default="2010-09-08_074500_HF_S002_S001",
        type=str,
        help="Name of video file on which to run experiments.",
    )
    args = parser.parse_args()

    assert args.data_dir is not None
    assert args.file_name is not None

    return args
