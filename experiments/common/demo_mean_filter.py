
from pathlib import Path
import numpy as np
import cv2
import imageio as iio
import argparse

from fish import REPO_PATH, logger
from fish.data import prep_exp_data, cap_to_nparray
from fish.filter.common import MeanFilter
from fish.utils import DefaultHelpParser

parser = DefaultHelpParser(description="Input path of video to filter")
parser.add_argument(
    'data_dir',
    metavar='d',
    nargs='?',
    default="/Users/nowa201/Data/fish_detector/mp4",
    type=str,
    help="Data directory that can be used to reference files without supplying a full path."
)
#name = "2010-09-08_081500_HF_S021"
#name = "2010-09-09_020001_HF_S013"
parser.add_argument(
    'file_name',
    metavar='f',
    nargs='?',
    default="2010-09-08_074500_HF_S002_S001",
    type=str,
    help="Name of video file on which to run experiments."
)
args = parser.parse_args()

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
    raw_writer = iio.get_writer(str(image_path) + '/demo_raw_video.gif',
                                mode='I', fps=fps)
    bgf_writer = iio.get_writer(str(image_path) + '/demo_background-subtracted_video.gif',
                                mode='I', fps=fps)

    for i in range(n):
        raw_writer.append_data(s_channel[i, ...].astype(np.uint8))
        bgf_writer.append_data(background_subtracted[i, ...].astype(np.uint8))
    raw_writer.close()
    bgf_writer.close()
