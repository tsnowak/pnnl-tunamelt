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


def svd_filter(
    video: Array["N,H,W,C", np.uint8],
    fps: int,
    freq_range: Optional[Tuple] = (1.5, 3.0),
    thresh_func: Optional[Callable[[Tuple], Array["N,H,W,C", np.float32]]] = None,
) -> Array["H,W,C", np.uint8]:

    log.debug(f"input video shape:{video.shape}")

    frame = video[0, :, :, 0]
    frame_line = frame.ravel()
    M = frame_line

    # create big M matrix for SVD capped at 100 because of ram limits
    for i in range(0, 100):
        frame = np.zeros((1792, 1032))
        frame = video[i, :, :, 0]
        frame_line = frame.ravel()
        M = np.vstack([M, frame_line])

    test_frame = M[49, :].reshape((1792, 1032)).T
    plt.imsave("test_frame.png", test_frame)
    # SVD vals
    u, s, v = np.linalg.svd(M.T, full_matrices=False)

    # fram lines from the big M matrix
    low_rank = np.zeros((len(u), len(v)))
    mid_rank = np.expand_dims(u[:, 47], 1) * s[47] * np.expand_dims(v[47, :], 0)
    high_rank = np.zeros((len(u), len(v)))

    # trying to take off vectores from either side
    for i in range(0, 10):
        low_rank += s[i] * np.outer(u.T[i], v[i])

    mask = np.zeros((video.shape[1], video.shape[2]))

    for i in range(0, 100):
        high_rank += np.expand_dims(u[:, i], 1) * s[i] * np.expand_dims(v[i, :], 0)
        mask += np.reshape(high_rank[:, i], (video.shape[1], video.shape[2]))

    for i in range(60, 40, -1):
        mid_rank += np.expand_dims(u[:, i], 1) * s[i] * np.expand_dims(v[i, :], 0)

    log.debug(f"input video shape:{low_rank.shape}")

    middleground = np.reshape(mid_rank[:, 49], (video.shape[1], video.shape[2]))
    background = np.reshape(low_rank[:, 49], (video.shape[1], video.shape[2]))
    foreground = np.reshape(high_rank[:, 49], (video.shape[1], video.shape[2]))

    plt.imsave("mask.png", mask, cmap="Greys")
    plt.imsave("background.png", background, cmap="Greys")
    plt.imsave("foreground.png", foreground, cmap="Greys")
    plt.imsave("middleground.png", middleground, cmap="Greys")

    filter_writer = iio.get_writer("demo_svd.gif", mode="I", fps=fps)
    raw_writer = iio.get_writer("raw_video.gif", mode="I", fps=fps)

    mask = 1 - mask / 255
    for i in range(video.shape[0] - 60):
        frame_buffer = video[i, :, :, 0] - mask
        filter_writer.append_data(frame_buffer.astype(np.uint8))
        raw_writer.append_data(video[i, :, :, 0])

    filter_writer.close()
    raw_writer.close()

    pass
