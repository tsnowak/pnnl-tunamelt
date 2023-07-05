import sys
import json
from pathlib import Path
from typing import OrderedDict
from afdme import REPO_PATH, log
from afdme.data import DataLoader, Dataset, numpy_to_cv2
from afdme.filter import common, dft
from afdme.vis import viz_video_results

import imageio.v3 as iio
import numpy as np

if __name__ == "__main__":

    file_path = f"{REPO_PATH}/data/mp4/train"
    labels = f"{REPO_PATH}/data/labels/cvat-video-1.1/train"

    dataloader = DataLoader(Dataset(videos=file_path, labels=labels), split="train")

    fps = 10
    frame_delay = 1.0 / fps

    # initialize filters
    params_path = f"{REPO_PATH}/experiments/88.json"
    with open(params_path, "r") as f:
        params = json.load(f)

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

    # define filter order
    filters = OrderedDict(
        [
            ("original", None),
            ("mean_filter", mean_filter),
            ("turbine_filter", turbine_filter),
            ("denoise_filter", denoise_filter),
            ("intensity_filter", intensity_filter),
            ("contour_filter", contour_filter),
            ("tracklet_association", tracklet_association),
        ]
    )

    # get video, label

    # Smallest Target/Hardest to Detect: 20,19
    # Most Noise (least % frames removed): 13
    # video, label = dataloader.get_vid_id(13)  # index dataloader by video_id
    video, label = dataloader[0]  # get idx from dataloader iterator
    log.info(f"Using video {label['video_id']}...")

    # run data through the filters in order
    log.info("Calculating filters...")
    outputs = OrderedDict()
    for idx, (filter_name, filter) in enumerate(filters.items()):
        log.info(f"\tCalculating {filter_name}...")
        if filter_name == "original":
            outputs["original"] = video[..., 2]
        else:
            tmp = list(outputs.items())[-1]
            outputs[filter_name] = filter.filter(tmp[1])

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

    log.info("writing hsv and masks")
    log.info(video[..., 2].shape)
    iio.imwrite("hsv.mp4", video[..., 2], format_hint=".mp4", fps=10)
    log.info(mean_filter.mask.dtype)
    iio.imwrite(
        "mean_filter_mask.mp4",
        mean_filter.mask.astype(np.uint8) * 255.0,
        format_hint=".mp4",
        fps=10,
    )
    log.info(turbine_filter.mask.shape)
    log.info(turbine_filter.mask.dtype)
    iio.imwrite(
        "turbine_filter_mask.png",
        turbine_filter.mask.astype(np.uint8) * 255,
        format_hint=".png",
    )
    log.info(intensity_filter.mask.dtype)
    iio.imwrite(
        "intensity_filter_mask.mp4",
        intensity_filter.mask.astype(np.uint8) * 255.0,
        format_hint=".mp4",
        fps=10,
    )
    log.info("done")

    # view and save videos with results + save results
    log.info("Visualizing contour filter outputs...")
    viz_video_results(
        display,
        label,
        contour_pred,
        fps,
        params=params,
        show=False,
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
        show=False,
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
        show=False,
        save=False,
        out_path=Path("no_preds"),
        video_type=".mp4",
    )
