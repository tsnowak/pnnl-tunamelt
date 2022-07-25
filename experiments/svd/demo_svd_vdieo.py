from pathlib import Path
import numpy as np
import cv2
import imageio as iio

from turbx import REPO_PATH, log
from turbx.data import get_file_path, cap_to_nparray
from turbx.filter.svd import svd_filter

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

    # generate the filter
    log.info("Generating SVD filter...")
    fourier_pos = svd_filter(s_channel, fps, freq_range=filter_freq_range)

    # write to gifs
    log.info("Writing to file...")
#   raw_writer = iio.get_writer(str(image_path) + '/demo_raw_video.gif',
#                               mode='I', fps=fps)
#   filter_writer = iio.get_writer(str(image_path) + '/demo_dft_filtered_video.gif',
#                                 mode='I', fps=fps)
#   for i in range(n):
#        frame = s_channel[i, ...] * fourier_zero
#       raw_writer.append_data(s_channel[i, ...].astype(np.uint8))
#        filter_writer.append_data(frame.astype(np.uint8))
#   raw_writer.close()
#   filter_writer.close()
