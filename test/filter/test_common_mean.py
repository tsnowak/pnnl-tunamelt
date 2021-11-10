
from pathlib import Path
import numpy as np
import cv2
import imageio as iio

from fish import REPO_PATH, logger
from fish.data import get_file_path, cap_to_nparray
from fish.filter.common import mean_filter


def test_mean_filter():
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

    # generate the rolling average filter
    logger.info("Generating Rolling Average filter...")
    mf_s_channel, mf_filter = mean_filter(s_channel)

    cv2.namedWindow("Mean Filter Video", cv2.WINDOW_AUTOSIZE)

    # show the video until escape is pressed
    n_frames = mf_s_channel.shape[0]
    cntr = 0
    while True:
        frame = np.concatenate(
            (s_channel[cntr, ...], mf_s_channel[cntr, ...]), axis=1)
        cv2.imshow("Mean Filter Video", frame)

        cntr += 1
        if cntr == n_frames:
            cntr = 0

        k = cv2.waitKey(int(1000*(1/fps)))
        if k == 27:
            cv2.destroyAllWindows()
            break

    return None
