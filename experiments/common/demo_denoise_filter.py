from pathlib import Path
import numpy as np
from turbx import REPO_PATH, log
from turbx.data import DataLoader, Dataset, numpy_to_cv2
from turbx.filter import common
from turbx.vis import view, label_to_per_frame_list, write_video

# args = standard_parser()

if __name__ == "__main__":

    file_path = f"{REPO_PATH}/data/mp4"
    labels = f"{REPO_PATH}/data/labels"

    dataloader = DataLoader(
        Dataset(videos=file_path, labels=labels, video_format="HSV")
    )

    # TODO can I get this from video file?
    fps = 10
    frame_delay = 1.0 / fps

    mean_filter = common.MeanFilter(fps=fps)
    denoise_filter = common.DeNoiseFilter(fps=fps)

    log.info("Calculating filter...")
    video, label = dataloader[0]
    value_channel = video[..., 2]
    mean = mean_filter.filter(value_channel)
    denoised = denoise_filter.filter(mean)

    video = numpy_to_cv2(video, "HSV", "BGR")
    # mean = numpy_to_cv2(mean, "HSV", "BGR")

    # mask = numpy_to_cv2((filter.mask).astype(np.uint8), "GRAY", "RGB")
    # write_video(mask, "mean_mask", fps=fps)

    log.info("Displaying output...")
    view(
        {
            "original": video,
            "mean_filtered": mean,
            "denoised": denoised,
        },
        label,
        None,  # placeholder for predictions output
        fps,
        show=True,
        save=False,
        out_path=Path(),
        video_type=".gif",
    )
