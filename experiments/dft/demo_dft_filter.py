import os
from datetime import datetime
from pathlib import Path
import imageio as iio
from turbx import REPO_PATH, log
from turbx.data import DataLoader, Dataset, numpy_to_cv2
from turbx.filter import common, dft
from turbx.vis import view

# args = standard_parser()

if __name__ == "__main__":

    file_path = f"{REPO_PATH}/data/mp4"
    labels = f"{REPO_PATH}/data/labels"

    dataloader = DataLoader(Dataset(videos=file_path, labels=labels))

    # TODO can I get this from video file?
    fps = 10
    frame_delay = 1.0 / fps

    mean_filter = common.MeanFilter(fps=fps)
    turbine_filter = dft.DFTFilter(fps=fps)
    intensity_filter = common.IntensityFilter(fps=fps)
    # median
    de_filter = common.DilateErrode(fps=fps, dilation=4, erosion=4)
    contour_filter = common.ContourFilter()

    # get video, label
    video, label = dataloader[3]
    value_channel = video[..., 2]
    log.info("Calculating filter...")
    # mean filter
    mean = mean_filter.filter(value_channel)
    # turbine filter
    turbine = turbine_filter.filter(mean)
    # intensity filter
    # intensity = intensity_filter.filter(turbine)
    # dilation_erosion filter
    de = de_filter.filter(turbine)
    # contour filter
    pred = contour_filter.filter(de)

    video = numpy_to_cv2(video, "HSV", "BGR")
    # mean = numpy_to_cv2(mean, "HSV", "BGR")
    # turbine = numpy_to_cv2(turbine, "HSV", "BGR")
    # intensity = numpy_to_cv2(intensity, "HSV", "RGB")
    # de = numpy_to_cv2(de, "HSV", "RGB")

    log.info("Generating output...")
    view(
        {
            "original": video,
            "mean_filtered": mean,
            "turbine_filtered": turbine,
            "intensity_filtered": intensity,
            "dilated_eroded": de,
        },
        label,
        pred,  # placeholder for predictions output
        fps,
        show=True,
        save=False,
        out_path=Path(),
        video_type=".gif",
    )
