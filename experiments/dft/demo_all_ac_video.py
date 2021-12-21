
from pathlib import Path
import numpy as np
import cv2
import imageio as iio
import argparse

from fish import REPO_PATH, logger
from fish.data import prep_exp_data, cap_to_nparray
from fish.filter.dft import DFTFilter
from fish.filter.common import IntensityFilter, MeanFilter
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
                                         '/experiments/dft/outputs/all_ac_video')

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

    # generate background subtraction
    logger.info("Generating Rolling Average filter...")
    m_filter = MeanFilter(s_channel, fps)
    background_subtracted = m_filter.apply()

    # generate the DFT filter
    logger.info("Generating DFT filter...")
    dft = DFTFilter(s_channel, fps, freq_range=filter_freq_range)
    dft_filtered = dft.apply(background_subtracted)

    # apply intensity filter
    logger.info("Generating Intensity filter...")
    i_filter = IntensityFilter(dft_filtered, fps, n=500)
    intensity_filtered = i_filter.apply()

    # write to gifs
    logger.info("Writing to file...")
    raw_writer = iio.get_writer(str(image_path) + '/demo_raw_video.gif',
                                mode='I', fps=fps)
    dft_writer = iio.get_writer(str(image_path) + '/demo_dft_filtered_video.gif',
                                mode='I', fps=fps)
    bgf_writer = iio.get_writer(str(image_path) + '/demo_background-subtracted_video.gif',
                                mode='I', fps=fps)
    if_writer = iio.get_writer(str(image_path) + '/demo_intensity-filtered_video.gif',
                               mode='I', fps=fps)
    for i in range(n):
        raw_writer.append_data(s_channel[i, ...].astype(np.uint8))
        dft_writer.append_data(dft_filtered[i, ...].astype(np.uint8))
        bgf_writer.append_data(background_subtracted[i, ...].astype(np.uint8))
        if_writer.append_data(intensity_filtered[i, ...].astype(np.uint8))
    raw_writer.close()
    dft_writer.close()
    bgf_writer.close()
    if_writer.close()
