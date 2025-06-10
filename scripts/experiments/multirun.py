import argparse
import json
from typing import OrderedDict
from venv import create

from tunamelt import REPO_PATH, log
from tunamelt.data import DataLoader, Dataset
from tunamelt.filter import common, dft
from tunamelt.run import run, create_pipeline, create_dataloader, create_paths


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", help="dataset to use: [train, test]", default="train"
    )
    parser.add_argument(
        "-p",
        "--params",
        help="Params files to run on relative to the base repo directory",
        nargs="*",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        help="path to the root of the data directory",
        default=f"{REPO_PATH}/data/PNNL-TUNAMELT",
    )
    parser.add_argument(
        "--results_path",
        help="path at which to save experimental results",
        default=f"{REPO_PATH}/scripts/experiments/results",
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

    return parser


def multirun(args):
    run_path, param_batches = create_paths(args)

    dataloader = create_dataloader(args)

    # run filters on every video for each param combination
    for itr, params in enumerate(param_batches):
        fps = 10
        frame_delay = 1.0 / fps

        # create pipeline of filters
        filters = create_pipeline(args, params, fps)

        # set self_idx to 0 to restart runs through dataset
        dataloader.reset()

        # run over dataset using given params
        run(
            itr,
            filters,
            params,
            dataloader,
            run_path,
        )


if __name__ == "__main__":
    parser = create_parser()
    args = vars(parser.parse_args())
    multirun(args)
