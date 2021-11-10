
from pathlib import Path
import numpy as np
import cv2
import imageio as iio

from fish import REPO_PATH, logger
from fish.data import get_file_path, cap_to_nparray
from fish.filter.dft import fourier_filter
from fish.filter.common import mean_filter, intensity_filter

if __name__ == "__main__":

    # TODO - modify for generalized usage
    # load data
    data_paths = [
        '/Users/nowa201/Data/fish_detector',
        '/data/nowa201/Projects/fish_detection/mp4'
    ]

    #name = "2010-09-08_074500_HF_S002_S001"
    #name = "2010-09-08_081500_HF_S021"
    name = "2010-09-09_020001_HF_S013"
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

    # generate the DFT filter
    logger.info("Generating DFT filter...")
    fourier_pos = fourier_filter(s_channel, fps, freq_range=filter_freq_range)
    fourier_zero = np.abs(fourier_pos - 1.)

    # apply the rolling average filter
    logger.info("Generating Rolling Average filter...")
    background_subtracted, avg_filter = mean_filter(s_channel)

    # apply the fourier filter
    fourier_filtered = np.multiply(background_subtracted, fourier_zero)

    # apply intensity filter
    intensity_filtered = intensity_filter(fourier_filtered)

    # write to gifs
    logger.info("Writing to file...")
    raw_writer = iio.get_writer(str(image_path) + '/demo_raw_video.gif',
                                mode='I', fps=fps)
    dft_writer = iio.get_writer(str(image_path) + '/demo_fourier_filtered_video.gif',
                                mode='I', fps=fps)
    bgf_writer = iio.get_writer(str(image_path) + '/demo_background-subtracted_video.gif',
                                mode='I', fps=fps)
    if_writer = iio.get_writer(str(image_path) + '/demo_intensity-filtered_video.gif',
                               mode='I', fps=fps)
    for i in range(n):
        raw_writer.append_data(s_channel[i, ...].astype(np.uint8))
        dft_writer.append_data(fourier_filtered[i, ...].astype(np.uint8))
        bgf_writer.append_data(background_subtracted[i, ...].astype(np.uint8))
        if_writer.append_data(intensity_filtered[i, ...].astype(np.uint8))
    raw_writer.close()
    dft_writer.close()
    bgf_writer.close()
    if_writer.close()
