from pathlib import Path
from typing import OrderedDict
from turbx import REPO_PATH, log
from turbx.data import DataLoader, Dataset, numpy_to_cv2
from turbx.filter import common
from turbx.vis import view

## SUPER SLOW
if __name__ == "__main__":

    file_path = f"{REPO_PATH}/data/mp4"
    labels = f"{REPO_PATH}/data/labels"

    dataloader = DataLoader(Dataset(videos=file_path, labels=labels))

    fps = 10
    frame_delay = 1.0 / fps

    # initialize filters
    mean_filter = common.MeanFilter(fps=fps)
    fp_filter = common.FindPeaksFilter(fps=fps, method="lee")
    contour_filter = common.ContourFilter()

    # define filter order
    filter_order = [
        "original",
        mean_filter,
        fp_filter,
        contour_filter,
    ]

    # get video, label
    video, label = dataloader[3]
    log.info(f"Using video {label['video_id']}...")

    # calculate filters in order
    log.info("Calculating filters...")
    outputs = OrderedDict()
    for idx, filter in enumerate(filter_order):
        filter_name = filter if isinstance(filter, str) else filter.__class__.__name__
        log.info(f"\tCalculating {filter_name}...")
        if filter == "original":
            outputs["original"] = video[..., 2]
        else:
            tmp = list(outputs.items())[-1]
            outputs[filter.__class__.__name__] = filter.filter(tmp[1])

    # get filter outputs in order
    display = OrderedDict()
    pred = None
    idx = 0
    for name, output in outputs.items():
        if name == "original":
            display[name] = numpy_to_cv2(video, "HSV", "BGR")
        elif name == "ContourFilter":
            pred = output
        else:
            display[name] = numpy_to_cv2(output, filter_order[idx].out_format, "BGR")
        idx += 1

    # display or save filters
    log.info("Visualizing filter output...")
    view(
        display,
        label,
        pred,
        fps,
        show=True,
        save=False,
        out_path=Path(),
        video_type=".gif",
    )
