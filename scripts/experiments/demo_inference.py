import argparse
import json
import sys
from pathlib import Path
from typing import OrderedDict

import imageio.v3 as iio
import numpy as np

from afdme import REPO_PATH, log
from afdme.data import DataLoader, Dataset, numpy_to_cv2
from afdme.run import create_dataloader, create_pipeline, create_paths, run_pipeline
from afdme.filter import common, dft
from afdme.vis import viz_video_results


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", help="dataset to use: [train, test]", default="train"
    )
    parser.add_argument(
        "-p",
        "--params",
        help="Params files to run on relative to the base repo directory",
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
    parser.add_argument(
        "-s",
        "--show",
        help="If true, creates windows to display the video and masks",
        default=False,
    )
    parser.add_argument(
        "-i",
        "--id",
        help="ID of the video to run on. See labels .xml for video ids.",
        required=True,
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = vars(parser.parse_args())
    run_path, param_batches = create_paths(args)

    fps = 10
    frame_delay = 1.0 / fps

    # compose the dataloader
    dataloader = create_dataloader(args)

    # initialize filters
    params_path = f"{REPO_PATH}/{args['params']}"
    with open(params_path, "r") as f:
        params = json.load(f)

    # create the filter pipeline  and load the params file
    filters = create_pipeline(args, params, fps=fps)

    # get video, label

    # Smallest Target/Hardest to Detect: 20,19
    # Most Noise (least % frames removed): 13
    # index dataloader by video_id
    video, label = dataloader.get_vid_id(int(args["id"]))
    # video, label = dataloader[0]  # get idx from dataloader iterator
    log.info(f"Running on video {label['video_id']}...")

    # run the pipeline on the given video
    outputs = run_pipeline(filters, video)

    # get filter outputs in order
    display = OrderedDict()
    pred = None
    idx = 0
    filters_list = list(filters.items())
    for name, output in outputs.items():
        if name == "original":
            display[name] = numpy_to_cv2(video, "HSV", "BGR")
        elif filters[name].__class__.__name__ == "ContourFilter":
            contour_pred = output
        elif filters[name].__class__.__name__ == "TrackletAssociation":
            assoc_pred = output
        else:
            display[name] = numpy_to_cv2(output, filters_list[idx][1].out_format, "BGR")
        idx += 1

    log.info(f"Writing videos and masks to {run_path}")
    iio.imwrite(f"{run_path}/hsv.mp4", video[..., 2], format_hint=".mp4", fps=10)
    iio.imwrite(
        f"{run_path}/mean_filter_mask.mp4",
        filters["mean_filter"].mask.astype(np.uint8) * 255.0,
        format_hint=".mp4",
        fps=10,
    )
    iio.imwrite(
        f"{run_path}/turbine_filter_mask.png",
        filters["turbine_filter"].mask.astype(np.uint8) * 255,
        format_hint=".png",
    )
    iio.imwrite(
        f"{run_path}/intensity_filter_mask.mp4",
        filters["intensity_filter"].mask.astype(np.uint8) * 255.0,
        format_hint=".mp4",
        fps=10,
    )
    log.info("Done.")

    # view and save videos with results + save results
    log.info("Visualizing contour filter outputs...")
    viz_video_results(
        display,
        label,
        contour_pred,
        fps,
        params=params,
        show=args["show"],
        save=False,
        out_path=Path("contour_det"),
        video_type=".mp4",
    )

    log.info("Visualizing association filter outputs...")
    viz_video_results(
        display,
        label,
        assoc_pred,
        fps,
        params=params,
        show=args["show"],
        save=False,
        out_path=Path("association_det"),
        video_type=".mp4",
    )

    log.info("Visualizing filter outputs...")
    viz_video_results(
        display,
        label,
        None,
        fps,
        params=params,
        show=args["show"],
        save=False,
        out_path=Path("no_preds"),
        video_type=".mp4",
    )
