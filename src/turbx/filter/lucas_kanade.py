import cv2
import numpy as np


"""
Implements a singular value decomposition (SVD) filter which is used to
remove the high/med. frequency components induced by the turbine's motion
in the given video block
"""

import numpy as np
import imageio as iio
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt

import cv2

from turbx import log
from turbx.utils import Array


def lucas_kanade(
    video: Array["N,H,W,C", np.uint8],
    fps: int,
    freq_range: Optional[Tuple] = (1.5, 3.0),
    thresh_func: Optional[Callable[[Tuple], Array["N,H,W,C", np.float32]]] = None,
) -> Array["H,W,C", np.uint8]:

    feature_params = dict(
        maxCorners=100, qualityLevel=0.00000001, minDistance=2, blockSize=2
    )

    lk_params = dict(
        winSize=(60, 80),
        maxLevel=8,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 0.03),
        minEigThreshold=1e-8,
    )

    color = np.random.randint(0, 255, (100, 3))

    old_gray = video[7, :, :, 0]
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    mask = np.zeros_like(
        np.dstack((video[0, :, :, :], np.zeros((video.shape[1], video.shape[2], 2)))),
        dtype=np.uint8,
    )
    motion_mask = mask

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray, video[8, :, :, 0], p0, None, **lk_params
    )

    motion_filter = iio.get_writer("motion_filter.gif", mode="I", fps=fps)
    i = 9
    while i < video.shape[0]:

        frame = np.dstack(
            (video[i, :, :, :], np.zeros((video.shape[1], video.shape[2], 2)))
        )
        # print(frame.shape)
        frame_gray = video[i, :, :, 0]
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        if p1 is None:
            i += 1
            continue
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), (255, 255, 255), 10)
        # print(mask.shape)
        img = mask + frame
        motion_filter.append_data(img)
        motion_mask += mask

        mask = np.zeros_like(
            np.dstack(
                (video[0, :, :, :], np.zeros((video.shape[1], video.shape[2], 2)))
            ),
            dtype=np.uint8,
        )
        p0 = good_new.reshape(-1, 1, 2)
        old_gray = video[i, :, :, 0].copy()
        i += 1

    plt.imsave("motion_mask.png", motion_mask)
    maskFactor = 1 - motion_mask[:, :, 0] / 255
    return maskFactor

    """
    for i in range(video.shape[0]):
        filtered = video[i,:,:,0] * maskFactor
        filtered_writer.append_data(filtered)
        demo_raw.append_data(video[i,:,:,0])
    
    
    log.debug(motion_mask.shape)
    plt.imsave("motion_mask.png", motion_mask[:,:,2])
    demo_writer.close()
    filtered_writer.close()
    demo_writer.close()
    """
