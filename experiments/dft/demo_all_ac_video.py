from pathlib import Path
import numpy as np
import cv2
import imageio as iio
import argparse

from turbx import REPO_PATH, log
from turbx.data import prep_exp_data, cap_to_nparray
from turbx.filter.dft import DFTFilter
from turbx.filter.common import IntensityFilter, MeanFilter
from turbx.utils import standard_parser
from turbx.vis import write_video

args = standard_parser()

if __name__ == "__main__":

    data_dir = args.data_dir
    file_name = args.file_name

    vid_path, image_path = prep_exp_data(
        data_dir, file_name, "/experiments/dft/outputs/all_ac_video"
    )

    fps = 10
    filter_freq_range = (1.25, 2.75)

    log.info("Opening video...")

    cap = cv2.VideoCapture(str(vid_path))

    # convert to HSV
    video = cap_to_nparray(cap, format="HSV")

    # create cv video
    n, h, w, c = video.shape
    s_channel = video[..., 2].squeeze()
    s_channel = np.expand_dims(s_channel, axis=-1)

    # generate background subtraction
    log.info("Generating Rolling Average filter...")
    m_filter = MeanFilter(s_channel, fps)
    background_subtracted = m_filter.apply(s_channel, fps)

    # generate the DFT filter
    log.info("Generating DFT filter...")
    dft = DFTFilter(s_channel, fps, freq_range=filter_freq_range)
    dft_filtered = dft.apply(background_subtracted, fps)

    # apply intensity filter
    log.info("Generating Intensity filter...")
    i_filter = IntensityFilter(dft_filtered, fps, n=500)
    intensity_filtered = i_filter.apply(dft_filtered, fps)

    # write to gifs
    log.info("Writing to file...")
    videos = [s_channel, dft_filtered, background_subtracted, intensity_filtered]
    names = [
        "demo_raw_video.gif",
        "demo_dft_filtered_video.gif",
        "demo_background-subtracted_video.gif",
        "demo_intensity-filtered_video.gif",
    ]

    for video, name in zip(videos, names):
        write_video(video, name, image_path, fps, n)
