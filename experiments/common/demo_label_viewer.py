from pathlib import Path
import numpy as np
import cv2

from fish import REPO_PATH, log
from fish.data import DataLoader


def main():

    log.info("Opening video...")
    dl = DataLoader(
        f"{REPO_PATH}/data/mp4",
        labels_path=f"{REPO_PATH}/data/labels/cvat-video-1.1/annotations.xml",
        format="HSV",
    )

    files = [f.name for f in dl.path]
    log.info(files)

    for label in dl.labels[0]:

    # dli = iter(dl)
    # vid = next(dli)
    # s_channel = vid[..., 2]
    # fps = 10
    # filter_freq_range = (1.25, 2.75)


if __name__ == "__main__":
    main()
