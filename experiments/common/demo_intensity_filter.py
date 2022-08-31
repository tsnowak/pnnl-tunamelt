from pathlib import Path
import numpy as np
import cv2

from turbx import REPO_PATH, log
from turbx.data import prep_exp_data, cap_to_nparray
from turbx.filter.common import IntensityFilter
from turbx.utils import standard_parser
from turbx.vis import write_video

args = standard_parser()

if __name__ == "__main__":

    data_dir = args.data_dir
    file_name = args.file_name

    vid_path, image_path = prep_exp_data(
        data_dir, file_name, "/experiments/common/outputs/intensity_filter"
    )

    # TODO can I get this from video file?
    fps = 10

    # create cv video
    log.info("Opening video...")
    cap = cv2.VideoCapture(str(vid_path))

    # convert to HSV
    video = cap_to_nparray(cap, format="HSV")
    n, h, w, c = video.shape
    s_channel = video[..., 2].squeeze()
    s_channel = np.expand_dims(s_channel, axis=-1)

    # generate background subtraction
    log.info("Generating Intensity filter...")
    i_filter = IntensityFilter(s_channel, fps)
    intensity_filtered = i_filter.apply()

    log.info("Writing to file...")
    videos = [s_channel, intensity_filtered]
    names = ["demo_raw_video.gif", "demo_intensity-filtered_video.gif"]
    for video, name in zip(videos, names):
        write_video(video, name, fps, video_length=n, out_path=image_path)
