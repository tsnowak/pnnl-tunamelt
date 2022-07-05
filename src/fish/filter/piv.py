"""
Implements a particle image velocimetry (PIV) filter which is used to
detect circular motion in the video, detecting the turbine motion, and 
remove it from the video
"""

import numpy as np
import imageio as iio
from typing import Callable, Optional, Tuple
from openpiv import tools, pyprocess, scaling, filters, validation, preprocess

import matplotlib.pyplot as plt

import cv2

from fish import log
from fish.utils import Array


def piv_filter(
    video: Array["N,H,W,C", np.uint8],
    fps: int,
    freq_range: Optional[Tuple] = (1.5, 3.0),
    thresh_func: Optional[Callable[[Tuple], Array["N,H,W,C", np.float32]]] = None,
) -> Array["H,W,C", np.uint8]:
    pass

    log.debug(f"input video shape:{video.shape}")
    frame_a = video[0, :, :, 0]
    frame_b = video[1, :, :, 0]

    # Process the original cropped image and see the OpenPIV result:

    # typical parameters:
    window_size = 32  # pixels
    overlap = 16  # pixels
    search_area_size = 64  # pixels
    scaling_factor = 100  # micron/pixel

    # process again with the masked images, for comparison# process once with the original images
    u, v, sig2noise = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32),
        frame_b.astype(np.int32),
        window_size=window_size,
        overlap=overlap,
        dt=1.0 / 10,
        search_area_size=search_area_size,
        sig2noise_method="peak2peak",
    )
    x, y = pyprocess.get_coordinates(frame_a.shape, search_area_size, overlap)
    u, v, mask = validation.global_val(u, v, (-300.0, 300.0), (-300.0, 300.0))
    u, v, mask = validation.sig2noise_val(u, v, sig2noise, threshold=1.1)
    u, v = filters.replace_outliers(u, v, method="localmean", max_iter=3, kernel_size=3)
    x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor=scaling_factor)
    # save to a file
    x, y, u, v = tools.transform_coordinates(x, y, u, v)

    numFrames = 4

    for i in range(numFrames):
        frame_a = video[i, :, :, 0]
        frame_b = video[i + 1, :, :, 0]

        u_buff, v_buff, sig2noise_buff = pyprocess.extended_search_area_piv(
            frame_a.astype(np.int32),
            frame_b.astype(np.int32),
            window_size=window_size,
            overlap=overlap,
            dt=1.0 / 10,
            search_area_size=search_area_size,
            sig2noise_method="peak2peak",
        )

        u_buff, v_buff, mask_buff = validation.global_val(
            u_buff, v_buff, (-300.0, 300.0), (-300.0, 300.0)
        )
        u_buff, v_buff, mask_buff = validation.sig2noise_val(
            u_buff, v_buff, sig2noise, threshold=1.1
        )
        u_buff, v_buff = filters.replace_outliers(
            u_buff, v_buff, method="localmean", max_iter=3, kernel_size=3
        )
        x_buff, y_buff, u_buff, v_buff = scaling.uniform(
            x, y, u_buff, v_buff, scaling_factor=scaling_factor
        )
        x_buff, y_buff, u_buff, v_buff = tools.transform_coordinates(
            x, y, u_buff, v_buff
        )
        u += u_buff
        v += v_buff

    tools.save(x, y, u / numFrames, v / numFrames, mask, "test.txt")
    plt.imsave("frame_a.png", frame_a)
    fig, ax = plt.subplots(figsize=(8, 8))
    tools.display_vector_field(
        "test.txt",
        ax=ax,
        scaling_factor=scaling_factor,
        scale=5,
        width=0.0035,
        on_img=True,
        image_name="frame_a.png",
    )

    plt.figure()
    plt.imshow(np.c_[frame_a, frame_b], cmap="gray")
    plt.show()
