
from pathlib import Path
import numpy as np
import cv2
import imageio as iio
import argparse

from fish import REPO_PATH, logger
from fish.data import prep_exp_data, cap_to_nparray
from fish.filter.common import MeanFilter
from fish.utils import standard_parser
from fish.vis import write_video

args = standard_parser()

if __name__ == "__main__":

    data_dir = args.data_dir
    file_name = args.file_name

    vid_path, image_path = prep_exp_data(data_dir, file_name,
                                         '/experiments/common/outputs/mean_filter')

    # TODO can I get this from video file?
    fps = 10

    # create cv video
    logger.info("Opening video...")
    cap = cv2.VideoCapture(str(vid_path))

    # convert to HSV
    video = cap_to_nparray(cap, format="HSV")
    n, h, w, c = video.shape
    s_channel = video[..., 2].squeeze()
    s_channel = np.expand_dims(s_channel, axis=-1)

    # generate background subtraction
    logger.info("Generating Rolling Average filter...")
    m_filter = MeanFilter(s_channel, fps)
    background_subtracted = m_filter.apply()

    logger.info("Writing to file...")
    videos = [s_channel, background_subtracted]
    names = ["demo_raw_video.gif", "demo_background-subtracted_video.gif"]
    for video, name in zip(videos, names):
        write_video(video, name, image_path, fps, n)
