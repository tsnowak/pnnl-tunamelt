from pathlib import Path
import numpy as np
import cv2
import imageio as iio

from turbx import REPO_PATH, log
from turbx.data import get_file_path, cap_to_nparray
from turbx.filter.piv import piv_filter
from turbx.filter.common import mean_filter, intensity_filter

if __name__ == "__main__":

    # TODO - modify for generalized usage
    # load data
    data_paths = [
        "/home/bl33m/Desktop/windowsshare/fish_detection/",
        "/home/bl33m/Desktop/windowsshare/mp4/mp4",
    ]

    name = "2010-09-08_074500_HF_S002_S001"
    vid_path = get_file_path(f"{name}.mp4", data_paths, absolute=True)

    # define place to save outputs
    image_path = Path(REPO_PATH + "/experiments/dft/outputs")
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

    # using background subtraction and intensity filtering to try and clean
    # footage before piv
    background_subtracted, avg_filter = mean_filter(s_channel)

    intensity_filtered = intensity_filter(background_subtracted)

    # generate piv filter
    log.info("Generating SVD filter...")
    piv_mask = piv_filter(intensity_filtered, fps, freq_range=filter_freq_range)

    # write to gifs
    log.info("Writing to file...")
