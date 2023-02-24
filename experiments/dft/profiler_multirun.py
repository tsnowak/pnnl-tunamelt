import os, shutil, sys, argparse
from copy import deepcopy
from typing import List, Dict, OrderedDict, Union
import json
from datetime import datetime
from pathlib import Path
from pprint import pprint
from itertools import product
from memory_profiler import profile


from turbx import REPO_PATH
from turbx.filter import common, dft
from turbx.run import run
from turbx.data import DataLoader, Dataset
from multiprocessing import pool


def calculate_runtime(ds_len, avg_video_len=300):
    runtime_file = f"{REPO_PATH}/experiments/runtime.txt"
    if not os.path.isfile(runtime_file):
        open(runtime_file, "x")
    with open(f"{REPO_PATH}/experiments/runtime.txt", "r") as f_runtime:
        avg_runtime = list(f_runtime.readlines())
        if len(avg_runtime) == 0:
            return None
        avg_runtime = [float(r) for r in avg_runtime]
        avg_runtime = sum(avg_runtime) / len(avg_runtime)

    print(
        f"Estimated run time is: {(ds_len*avg_video_len*avg_runtime)/(1000000000*60*60)} hours."
    )
    return avg_runtime


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


def create_paths(args):
    # parse params arg
    assert (args["params"] is not None) and (
        len(args["params"])
    ) > 0, "Path to one or more params json files must be passed to the params argument"

    # create output folder structure
    date_time = datetime.now()
    ymd = date_time.strftime("%Y-%m-%d")
    hms = date_time.strftime("%H-%M-%S")
    run_name = str(Path(__file__).parent.name)
    run_path = f"{REPO_PATH}/experiments/{run_name}/outputs/{ymd}/{hms}"
    os.makedirs(run_path, exist_ok=True)

    params_list = []
    for i, params_path in enumerate(args["params"]):
        params_path = f"{REPO_PATH}/{params_path}"
        # load and copy jsons to run path for reference
        with open(params_path, "r") as f:
            params_list.append(json.load(f))
            shutil.copy(params_path, f"{run_path}/{i}-{Path(params_path).name}")

    # save flag arguments to run path
    args_json = json.dumps(args, indent=4)
    with open(f"{run_path}/args.json", "w") as outfile:
        outfile.write(args_json)

    param_batches = []
    for params in params_list:
        param_batches += generate_param_batches(params)

    return run_path, param_batches


# TODO: there are some big memory leaks and inefficiencies in here
@profile
def multirun(args):

    run_path, param_batches = create_paths(args)

    # create the dataset - do this once, do dataloader.reset() each batch
    data_path = ""
    label_path = ""
    if args["dataset"] == "test":
        data_path = f"{args['path']}/mp4/batched_test"
        label_path = f"{args['path']}/labels/cvat-video-1.1/batched_test"
    else:
        data_path = f"{args['path']}/mp4/{args['dataset']}"
        label_path = f"{args['path']}/labels/cvat-video-1.1/{args['dataset']}"
    dataloader = DataLoader(
        Dataset(videos=data_path, labels=label_path), split=args["dataset"]
    )
    calculate_runtime(len(dataloader))

    # run filters on every video for each param combination
    for itr, params in enumerate(param_batches):

        fps = 10
        frame_delay = 1.0 / fps

        # reinitialize filters given params
        mean_filter = common.MeanFilter(fps=fps, params=params["mean_filter"])
        turbine_filter = dft.DFTFilter(fps=fps, params=params["turbine_filter"])
        denoise_filter = common.GaussianBlurDenoiseFilter(
            fps=fps, params=params["denoise_filter"]
        )
        intensity_filter = common.IntensityFilter(
            fps=fps, params=params["intensity_filter"]
        )
        contour_filter = common.ContourFilter(params=params["contour_filter"])
        tracklet_association = common.TrackletAssociation(
            params=params["tracklet_association"]
        )

        # automatically load filters at runtime and ensure required filters are present
        # ie {"original": None} and {"contour_filter": contour_filter}
        filters = OrderedDict()
        filters.update({"original": None})
        # NOTE: I think, but am not sure all these filters are order-agnostic
        for filter in args["filters"]:
            try:
                filters.update({filter: eval(filter)})
            except SyntaxError:
                raise SyntaxError(
                    "The only valid filters to pass to --filters are: "
                    + "[mean_filter, turbine_filter, "
                    + "denoise_filter, intensity_filter, "
                    + "tracklet_association"
                )
        # always end with contour_filter
        filters.update({"contour_filter": contour_filter})
        # unless tracklet_association present
        if "tracklet_association" in filters.keys():
            filters.move_to_end("tracklet_association")

        # hardcoded equivalent
        # filters = OrderedDict(
        #    [
        #        ("original", None),
        #        ("mean_filter", mean_filter),
        #        ("turbine_filter", turbine_filter),
        #        ("denoise_filter", denoise_filter),
        #        ("intensity_filter", intensity_filter),
        #        ("contour_filter", contour_filter),
        #        #            ("tracklet_association", tracklet_association),
        #    ]
        # )

        # set self_idx to 0 to restart runs through dataset
        dataloader.reset()

        # run the filters on every video in the dataset; or on N=<max_vid_itrs> videos (for testing)
        run(
            itr,
            filters,
            params,
            dataloader,
            run_path,
            max_vid_itrs=10000000,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: update path to data based on split used
    parser.add_argument(
        "-d", "--dataset", help="dataset to use: [train, test]", default="train"
    )
    parser.add_argument("-p", "--params", help="Params files to run on", nargs="*")
    parser.add_argument(
        "--path",
        help="path to the root of the data directory",
        default=f"{REPO_PATH}/data",
    )
    parser.add_argument(
        "-f",
        "--filters",
        help="filters to use in the pipeline. Used for ablation tests.",
        default=[
            "mean_filter",
            "turbine_filter",
            "denoise_filter",
            "intensity_filter",
            "tracklet_association",
        ],
        nargs="*",
    )
    args = vars(parser.parse_args())
    multirun(args)
