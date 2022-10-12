from pathlib import Path
import numpy as np
import cv2
import imageio as iio
from openpiv import pyprocess, piv

from pathlib import Path
from turbx import REPO_PATH, log
from turbx.data import DataLoader, Dataset, numpy_to_cv2
from turbx.filter import common

# from turbx.filter.piv import piv_filter
from turbx.vis import view

if __name__ == "__main__":

    file_path = f"{REPO_PATH}/data/mp4"
    labels = f"{REPO_PATH}/data/labels"

    dataloader = DataLoader(Dataset(videos=file_path, labels=labels))

    # TODO can I get this from video file?
    fps = 10
    frame_delay = 1.0 / fps

    mean_filter = common.MeanFilter(fps=fps)

    # get video, label
    video, label = dataloader[0]
    log.info("Calculating filter...")
    # mean filter
    mean = mean_filter.filter(video)
    video = numpy_to_cv2(video, "HSV", "BGR")
    mean = numpy_to_cv2(mean, "HSV", "RGB")

    # generate piv filter
    # see https://openpiv.readthedocs.io/en/latest/_modules/openpiv/piv.html#simple_piv
    log.info("Generating PIV filter...")
    cv2.namedWindow("original")
    cv2.namedWindow("mean")
    cv2.namedWindow("piv")
    frame2 = None
    win_size = 16
    overlap = 8
    for idx, frame1 in enumerate(mean):
        oframe = video[idx, ...]
        cv2.imshow("original", oframe)
        cv2.imshow("mean", frame1)
        if frame2 is not None:
            f1 = frame1.sum(axis=2)
            f2 = frame2.sum(axis=2)
            u, v, s2n = pyprocess.extended_search_area_piv(
                f1.astype(np.int32),
                f2.astype(np.int32),
                window_size=win_size,
                overlap=overlap,
                search_area_size=win_size,
            )
            x, y = pyprocess.get_coordinates(
                image_size=f1.shape, search_area_size=win_size, overlap=overlap
            )
            valid = s2n > np.percentile(s2n, 5)
            for batch in zip(x[valid], y[valid], u[valid], v[valid]):
                a, b, c, d = batch
                cv2.arrowedLine(
                    frame2, (int(a), int(b)), (int(a + c), int(b - d)), (255, 0, 0), 3
                )
            cv2.imshow("piv", frame2)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        frame2 = frame1.copy()

    cv2.destroyAllWindows()
