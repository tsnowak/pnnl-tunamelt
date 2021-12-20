
from pathlib import Path
import numpy as np
import cv2
import imageio as iio

from fish import REPO_PATH, logger
from fish.data import get_file_path, cap_to_nparray
from fish.filter.dft import DFTFilter
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

    # TODO - modify for generalized usage
    # load data
    data_dir = args.data_dir
    file_name = args.file_name
    #name = "2010-09-08_081500_HF_S021"
    #name = "2010-09-09_020001_HF_S013"

    if Path(data_dir + "/" + file_name).is_file():
        file_name = file_name
        data_dir = data_dir
    elif Path(data_dir + "/" + file_name + ".mp4").is_file():
        file_name = file_name + ".mp4"
        data_dir = data_dir
    elif Path(file_name).is_file():
        data_dir = str(Path(file_name).parent)
        file_name = str(Path(file_name).name)
    else:
        raise ValueError(
            f"Invalid data_dir or file_name provided.\ndata_dir: {data_dir}\nfile_name: {file_name}")

    logger.info(f"Found file at: {data_dir}/{file_name}")
    vid_path = get_file_path(file_name, [data_dir], absolute=True)

    # define place to save outputs
    image_path = Path(REPO_PATH + '/experiments/dft/outputs/dft_ac_video')
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
    dft = DFTFilter(s_channel, fps, freq_range=filter_freq_range)
    fourier_pos = dft.generate()
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
