from pathlib import Path

from turbx import REPO_PATH, log
from turbx.data import DataLoader, Dataset, numpy_to_cv2
from turbx.filter import common, dft
from turbx.vis import view
from turbx.metrics import boxes_to_binary

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
    contour_filter = common.ContourFilter()

    # get video, label
    video, label = dataloader[0]
    log.info("Calculating filter...")
    # mean filter
    mean = mean_filter.filter(video)
    # intensity filter
    intensity = intensity_filter.filter(mean)
    # contour filter
    pred = contour_filter.filter(intensity)

    video = numpy_to_cv2(video, "HSV", "BGR")
    mean = numpy_to_cv2(mean, "HSV", "RGB")
    intensity = numpy_to_cv2(intensity, "HSV", "RGB")

    log.info("Displaying output...")
    view(
        {
            "original": video,
            "mean_filtered": mean,
            "intensity_filtered": intensity,
        },
        label,
        pred,  # placeholder for predictions output
        fps,
        save=False,
        out_path=Path(),
        video_type=".gif",
    )
