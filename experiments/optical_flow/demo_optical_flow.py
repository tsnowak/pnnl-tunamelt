from pathlib import Path
import numpy as np
import cv2
import imageio as iio

from turbx import REPO_PATH, log
from turbx.data import get_file_path, cap_to_nparray
from turbx.filter.lucas_kanade import lucas_kanade
from turbx.filter.common import mean_filter, intensity_filter

if __name__ == "__main__":

    # TODO - modify for generalized usage
    # load data
    data_paths = [
        "/home/bl33m/Desktop/windowsshare/fish_detection/",
        "/home/bl33m/Desktop/windowsshare/mp4/mp4",
    ]

    # name = "2010-09-08_074500_HF_S002_S001"
    name = "2010-09-08_103000_HF_S022"
    vid_path = get_file_path(f"{name}.mp4", data_paths, absolute=True)

    # define place to save outputs
    image_path = Path(REPO_PATH + "/experiments/optical_flow/outputs")
    Path(image_path).mkdir(exist_ok=True)

    fps = 10
    filter_freq_range = (1.25, 2.75)

    # create cv video
    log.info("Opening video...")
    cap = cv2.VideoCapture(str(vid_path))

    # convert to HSV
    video = cap_to_nparray(cap, format="HSV")
    n, h, w, c = video.shape
    s_channel = video[..., 2].squeeze()
    s_channel = np.expand_dims(s_channel, axis=-1)

    # generate the filter
    log.info("Cleaning video...")
    background_subtracted, avg_filter = mean_filter(s_channel)

    motion_mask = lucas_kanade(background_subtracted, fps, freq_range=filter_freq_range)

    for i in range(s_channel.shape[0]):
        background_subtracted[i, :, :, 0] = (
            background_subtracted[i, :, :, 0] * motion_mask
        )

    intensity_filter = intensity_filter(background_subtracted)

    log.info("Writing Demos to file")
    raw_writer = iio.get_writer(str(image_path) + "/demo_raw.gif", mode="I", fps=fps)
    filter_writer = iio.get_writer(
        str(image_path) + "/demo_filtered.gif", mode="I", fps=fps
    )

    for i in range(s_channel.shape[0]):
        raw_writer.append_data(s_channel[i, ...].astype(np.uint8))
        filter_writer.append_data(intensity_filter[i, ...].astype(np.uint8))

    raw_writer.close()
    filter_writer.close()
