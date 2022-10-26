import sys
from pathlib import Path
from typing import OrderedDict
from turbx import REPO_PATH, log
from turbx.data import DataLoader, Dataset, numpy_to_cv2
from turbx.filter import common, dft
from turbx.vis import viz_video_results

## SUPER SLOW
if __name__ == "__main__":

    file_path = f"{REPO_PATH}/data/mp4"
    labels = f"{REPO_PATH}/data/labels"

    dataloader = DataLoader(Dataset(videos=file_path, labels=labels))

    fps = 10
    frame_delay = 1.0 / fps

    # initialize filters
    params = {
        "mean_filter": {"std_devs": 2.5},
        "turbine_filter": {"freq_range": (1.5, 3.0), "mask_smoothing": 9},
        "denoise_filter": {"blur_size": 11},
        "intensity_filter": {"thresh": 11},
        "contour_filter": {"min_area": 200, "max_area": 6000},
    }
    mean_filter = common.MeanFilter(fps=fps, params=params["mean_filter"])
    turbine_filter = dft.DFTFilter(fps=fps, params=params["turbine_filter"])
    denoise_filter = common.GaussianBlurDenoiseFilter(
        fps=fps, params=params["denoise_filter"]
    )
    intensity_filter = common.IntensityFilter(
        fps=fps, params=params["intensity_filter"]
    )
    contour_filter = common.ContourFilter(params=params["contour_filter"])
    # tracklet_params = {"window": 3, "thresh": 100.0}
    # tracklet_association = common.TrackletAssociation()

    # define filter order
    filter_order = [
        "original",
        "turbine_filter",
        "mean_filter",
        "intensity_filter",
        "denoise_filter",
        "contour_filter",
        # tracklet_association,
    ]

    # get video, label
    video, label = dataloader[0]
    log.info(f"Using video {label['video_id']}...")

    # calculate filters in order
    log.info("Calculating filters...")
    outputs = OrderedDict()
    for idx, filter_name in enumerate(filter_order):
        log.info(f"\tCalculating {filter_name}...")
        if filter_name == "original":
            outputs["original"] = video[..., 2]
        else:
            tmp = list(outputs.items())[-1]
            # outputs[filter_name] = filter.filter(tmp[1])
            outputs[filter_name] = eval(filter_name).filter(tmp[1])

    # get filter outputs in order
    display = OrderedDict()
    pred = None
    idx = 0
    for name, output in outputs.items():
        if name == "original":
            display[name] = numpy_to_cv2(video, "HSV", "BGR")
        elif eval(name).__class__.__name__ == "ContourFilter":
            pred = output
            # continue
        elif eval(name).__class__.__name__ == "TrackletAssociation":
            # pred = output
            continue
        else:
            display[name] = numpy_to_cv2(
                output, eval(filter_order[idx]).out_format, "BGR"
            )
        idx += 1

    # display or save filters
    log.info("Visualizing filter output...")
    viz_video_results(
        display,
        label,
        pred,
        fps,
        params=params,
        show=True,
        save=False,
        out_path=Path(),
        video_type=".gif",
    )
