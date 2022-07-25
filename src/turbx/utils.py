import argparse
import sys
from pathlib import Path
from typing import Generic, Tuple, TypeVar, NewType

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq, fftn, fftshift

from turbx import log

Shape = TypeVar("Shape")
DType = TypeVar("DType")


class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)


def standard_parser():
    parser = DefaultHelpParser(description="Input path of video to filter")
    parser.add_argument(
        "--data_dir",
        nargs="?",
        required=True,
        type=str,
        help="Data directory that can be used to reference files without supplying a full path.",
    )
    # name = "2010-09-08_081500_HF_S021"
    # name = "2010-09-09_020001_HF_S013"
    parser.add_argument(
        "--file_name",
        nargs="?",
        default="2010-09-08_074500_HF_S002_S001",
        type=str,
        help="Name of video file on which to run experiments.",
    )
    args = parser.parse_args()

    assert args.data_dir is not None
    assert args.file_name is not None

    return args


def generate_sinusoid(freq: int, fps: int, shape: Tuple, length=float):
    """
    Create a waveform
    """

    u, v = shape
    t = np.linspace(0, length, int(fps * length))
    # 8-bit-valued waveform
    waveform = (np.sin(freq * (2 * np.pi) * t) / 2.0) + 0.5

    log.debug(f"\nSinusoid shape: {waveform.shape}")

    waveform_video = np.tile(waveform, (u, v, 1))

    log.debug(f"\nSinusoid video shape: {waveform_video.shape}")

    assert waveform_video.shape == (u, v, len(t)), "Waveform video incorrectly shaped"

    return waveform_video


def generate_sinusoid_tile(freqs, element_shape, n_frames):
    """
    Returns numpy array representing a video of
    multi-frequency image tiles
    """
    # hold fps constant
    fps = 10
    length = n_frames / fps

    waveforms = []
    for freq in freqs:
        waveforms.append(
            generate_sinusoid(freq=freq, fps=fps, shape=element_shape, length=length)
        )

    output = np.concatenate(waveforms, axis=1)
    return output, fps, length


def crop_polygon(img, pts):
    """
    crop polygon of image, and return with black background
    """
    # (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = img[y : y + h, x : x + w].copy()

    # (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    # (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    return dst
