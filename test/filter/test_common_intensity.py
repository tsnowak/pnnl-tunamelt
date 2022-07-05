import sys
from pathlib import Path
import numpy as np
import cv2
import imageio as iio

from fish import REPO_PATH, log
from fish.data import DataLoader
from fish.filter.dft import DFTFilter
from fish.filter.common import IntensityFilter, MeanFilter

# create test output directory
file_path = Path(__file__).absolute()
out_path = Path(f"{file_path.parent}/outputs/{str(file_path.name).split('.')[0]}")
out_path.mkdir(parents=True, exist_ok=True)


def test_intensity_filter():
    # load video
    log.info("Opening video...")
    dl = DataLoader(
        f"{REPO_PATH}/data/mp4/2010-09-08_074500_HF_S002_S001.mp4", format="HSV"
    )
    dli = iter(dl)
    vid = next(dli)

    s_channel = vid[..., 2]
    fps = 10
    filter_freq_range = (1.25, 2.75)

    dft_filter = DFTFilter(freq_range=filter_freq_range)
    mean_filter = MeanFilter()
    intensity_filter = IntensityFilter()

    filters = [dft_filter, mean_filter, intensity_filter]
    filtered_video = s_channel
    for filter in filters:
        filtered_video = filter.filter(filtered_video, fps)

    cv2.namedWindow("Intensity Filter Video", cv2.WINDOW_AUTOSIZE)

    # TODO: REFACTOR VIEWER TO BE CLEAN AND REUSABLE
    # show the video until escape is pressed
    n_frames = filtered_video.shape[0]
    cntr = 0
    while True:
        frame = np.concatenate(
            (
                s_channel[cntr, ...].astype(np.uint8),
                mf_s_channel[cntr, ...].astype(np.uint8),
                i_s_channel[cntr, ...].astype(np.uint8),
            ),
            axis=1,
        )
        cv2.imshow("Intensity Filter Video", frame)

        cntr += 1
        if cntr == n_frames:
            cntr = 0

        k = cv2.waitKey(int(1000 * (1 / fps)))
        if k == 27:
            cv2.destroyAllWindows()
            break

        # save filtered video under test/filter/outputs/<test_name>/<test_name>.mp4
        iio.mimwrite(
            f"{out_path}/{sys._getframe().f_code.co_name}.mp4", filtered_video, fps=fps
        )

    return None
