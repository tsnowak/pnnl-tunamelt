import sys
from pathlib import Path
import imageio as iio

from fish import REPO_PATH, log
from fish.data import DataLoader
from fish.filter.common import MeanFilter

# VERIFIED

# create test output directory
file_path = Path(__file__).absolute()
out_path = Path(f"{file_path.parent}/outputs/{str(file_path.name).split('.')[0]}")
out_path.mkdir(parents=True, exist_ok=True)

# test background (mean) filter on test video
def test_mean_filter():

    # open video with dataloader
    log.info("Opening video...")
    dl = DataLoader(
        f"{REPO_PATH}/data/mp4/2010-09-08_074500_HF_S002_S001.mp4", format="HSV"
    )
    dli = iter(dl)
    vid = next(dli)

    # set fps and video channel to apply filter to
    fps = 10
    s_channel = vid[..., 2]

    # TODO: make this into a list of filters to apply?
    # generate + apply filter
    log.info("Generating Rolling Average filter...")
    filter = MeanFilter()
    filtered_video = filter.filter(s_channel, fps)

    # save filtered video under test/filter/outputs/<test_name>/<test_name>.mp4
    iio.mimwrite(
        f"{out_path}/{sys._getframe().f_code.co_name}.mp4", filtered_video, fps=fps
    )
