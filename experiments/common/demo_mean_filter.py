from pathlib import Path
from turbx import REPO_PATH, log
from turbx.data import DataLoader, Dataset, numpy_to_cv2
from turbx.filter import common
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
    video, label = dataloader[0]
    mean = filter.filter(video)

    video = numpy_to_cv2(video, "HSV", "BGR")
    mean = numpy_to_cv2(mean, "HSV", "RGB")

    log.info("Displaying output...")
    view(
        {
            "original": video,
            "mean_filtered": mean,
        },
        label,
        [],  # placeholder for predictions output
        fps,
        save=False,
        out_path=Path(),
        video_type=".gif",
    )
