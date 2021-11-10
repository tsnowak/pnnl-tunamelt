
from pathlib import Path
import numpy as np
import cv2
import imageio as iio

from fish import REPO_PATH, logger
from fish.data import get_file_path, cap_to_nparray
from fish.dft_filter import fourier_filter

if __name__ == "__main__":

    # TODO - modify for generalized usage
    # load data
    data_paths = [
        '/Users/nowa201/Data/fish_detector',
        '/data/nowa201/Projects/fish_detection/mp4'
    ]

    name = "2010-09-08_074500_HF_S002_S001"
    vid_path = get_file_path(f"{name}.mp4", data_paths, absolute=True)

    # define place to save images
    image_path = Path(REPO_PATH + '/experiments/dft/images')
    Path(image_path).mkdir(exist_ok=True)

    fps = 10
    filter_freq_range = (1.25, 2.75)

    # create cv video
    logger.info("Opening video...")
    cap = cv2.VideoCapture(str(vid_path))

    # convert to HSV
    video = cap_to_nparray(cap, format="HSV")
    n, h, w, c = video.shape
    s_channel = video[..., 2].squeeze()
    s_channel = np.expand_dims(s_channel, axis=-1)

    # generate the filter
    logger.info("Generating DFT filter...")
    fourier_pos = fourier_filter(s_channel, fps, freq_range=filter_freq_range)
    fourier_zero = np.abs(fourier_pos - 1.)

    # write to gifs
    logger.info("Writing to file...")
    raw_writer = iio.get_writer(str(image_path) + '/demo_raw_video.gif',
                                mode='I', fps=fps)
    filter_writer = iio.get_writer(str(image_path) + '/demo_dft_filtered_video.gif',
                                   mode='I', fps=fps)
    for i in range(n):
        frame = s_channel[i, ...] * fourier_zero
        raw_writer.append_data(s_channel[i, ...].astype(np.uint8))
        filter_writer.append_data(frame.astype(np.uint8))
    raw_writer.close()
    filter_writer.close()
