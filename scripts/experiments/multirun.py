import argparse
from typing import OrderedDict
import json


from afdme import REPO_PATH, log
from afdme.filter import common, dft
from afdme.run import run
from afdme.data import DataLoader, Dataset
from afdme.utils import (
    create_exp_dirs,
    load_params,
    generate_param_batches,
    create_paths,
)


def multirun(args):
    run_path, param_batches = create_paths(args)

    # create the dataset - do this once, do dataloader.reset() each batch
    if args["dataset"] == "test":
        data_path = f"{args['data_path']}/mp4/batched_test"
        label_path = f"{args['data_path']}/labels/cvat-video-1.1/batched_test"
    else:
        data_path = f"{args['data_path']}/mp4/{args['dataset']}"
        label_path = f"{args['data_path']}/labels/cvat-video-1.1/{args['dataset']}"
    dataloader = DataLoader(
        Dataset(videos=data_path, labels=label_path), split=args["dataset"]
    )

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

        # set self_idx to 0 to restart runs through dataset
        dataloader.reset()

        # run over dataset using given params
        # TODO: pass params to viz_video_results at a higher level
        # aka from the params file?
        run(
            itr,
            filters,
            params,
            dataloader,
            run_path,
        )


if __name__ == "__main__":
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
        default=f"{REPO_PATH}/data/AFD-ME",
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
    args = vars(parser.parse_args())
    multirun(args)
