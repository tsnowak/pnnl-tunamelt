from pathlib import Path
import numpy as np
import cv2

from turbx import REPO_PATH, log
from turbx.data import DataLoader, Dataset, numpy_to_cv2
from turbx.filter import common, dft
from turbx.utils import standard_parser
from turbx.vis import view

# args = standard_parser()

if __name__ == "__main__":

    file_path = f"{REPO_PATH}/data/mp4"
    labels = f"{REPO_PATH}/data/labels"

    dataloader = DataLoader(Dataset(videos=file_path, labels=labels))

    # TODO can I get this from video file?
    fps = 10
    frame_delay = 1.0 / fps

    filter = common.MeanFilter(fps=fps)

    log.info("Calculating filter...")
    video, label = next(dataloader)
    filter.calculate(video, fps)
    out = filter.filter(video)

    video = numpy_to_cv2(video, "HSV", "BGR")
    out = numpy_to_cv2(out, "HSV", "RGB")

    log.info("Displaying output...")
    view(
        {"original": video, "mean_filtered": out},
        label,
        label,  # placeholder for predictions output
        fps,
        out_path=Path(),
        video_type=".gif",
    )
