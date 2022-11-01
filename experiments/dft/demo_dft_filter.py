import sys
import json
from pathlib import Path
from typing import OrderedDict
from turbx import REPO_PATH, log
from turbx.data import DataLoader, Dataset, numpy_to_cv2
from turbx.filter import common, dft
from turbx.vis import viz_video_results

if __name__ == "__main__":

    file_path = f"{REPO_PATH}/data/mp4"
    labels = f"{REPO_PATH}/data/labels"

    dataloader = DataLoader(Dataset(videos=file_path, labels=labels))

    fps = 10
    frame_delay = 1.0 / fps

    # initialize filters
    params_path = f"{REPO_PATH}/experiments/best_params.json"
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
            ("turbine_filter", turbine_filter),
            ("mean_filter", mean_filter),
            ("intensity_filter", intensity_filter),
            ("denoise_filter", denoise_filter),
            ("contour_filter", contour_filter),
            ("tracklet_association", tracklet_association),
        ]
    )

    # get video, label
    video, label = dataloader.get_vid_id(14)  # index dataloader by video_id
    # video, label = dataloader[14]             # get idx from dataloader iterator
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
            # pred = output
            continue
        elif filters[name].__class__.__name__ == "TrackletAssociation":
            pred = output
            # continue
        else:
            display[name] = numpy_to_cv2(output, filters_list[idx][1].out_format, "BGR")
        idx += 1

    # view and save videos with results + save results
    log.info("Visualizing filter outputs...")
    viz_video_results(
        display,
        label,
        pred,
        fps,
        params=params,
        show=True,
        save=False,
        out_path=Path(),
        video_type=".mp4",
    )
