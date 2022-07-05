from pathlib import Path
import numpy as np
import cv2
import imageio as iio

from fish import REPO_PATH, log
from fish.data import prep_exp_data, cap_to_nparray
from fish.filter.dft import DFTFilter
from fish.utils import standard_parser
from fish.vis import write_video

args = standard_parser()

if __name__ == "__main__":

    data_dir = args.data_dir
    file_name = args.file_name

    vid_path, image_path = prep_exp_data(
        data_dir, file_name, "/experiments/dft/outputs/dft_ac_video"
    )
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
    log.info("Generating DFT filter...")
    dft = DFTFilter(s_channel, fps, freq_range=filter_freq_range)
    fourier_pos = dft.generate()
    fourier_zero = np.abs(fourier_pos - 1.0)

    # write to gifs
    log.info("Writing to file...")

    videos = [s_channel, s_channel * fourier_zero]
    names = ["demo_raw_video.gif", "demo_dft_filtered_video.gif"]
    for video, name in zip(videos, names):
        write_video(video, name, image_path, fps, n)
