"""
Implementation of the sliding discrete fourier transform. Used to filter out 
pixels oscillating within a range of frequencies on live video.
"""

from typing import Callable, Optional, Tuple

import numpy as np
import cv2
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
from river import misc
from turbx import log


def sdft_filter(
    video: np.ndarray,
    fps: int,
    windowSize: int = 20,
    freq_range: Optional[Tuple] = (1.5, 3.0),
) -> np.ndarray:
    """
    Uses the sliding dft in order to generate filtered video,
    at the moment just takes in the whole video and a frame size and
    just writes filtered video

    In the future it should take in a mask, N frames, and a window size of N
    and return the new mask
    """

    log.debug(f"input video shape:{video.shape}")

    sdft = misc.SDFT(windowSize)

    for i, frame in enumerate(video):
        sdft = sdft.update(frame)

    sdft.coefficients = np.fft.fftshift(np.abs(sdft.coefficients), axes=0)  # 0 freq at center
    # For each component get the fr1equency center that it represents
    freq = fftfreq(windowSize, d=1 / fps)
    freq = fftshift(freq)
    log.debug(f"fft_video {sdft.coefficients.shape}")
    # only operate on positive frequencies (greater than 0 plus fudge)
    freq_thresh = 0.0001
    pos_range = np.argwhere(freq > (0 + freq_thresh)).squeeze()
    sdft.coefficients = sdft.coefficients[pos_range, ...]
    log.debug(f"pos_range video: {sdft.coefficients.shape}")
    freq = freq[pos_range, ...]

    # get magnitude and phase of each frequency component
    # magnitude of that frequency component
    mag = np.absolute(sdft.coefficients)
    # phase between sine (im) and cosine (re) of that freq. component
    # phase = np.angle(fft_video)
    log.debug(f"magnitude array shape: {mag.shape}")

    # mask = average_threshold(fft_video, freq, mag, freq_range, factor=2)
    mask = max_threshold(sdft.coefficients, freq, mag, freq_range)
    log.debug(f"per pixel mask: {mask.shape}")

    mask = np.abs(mask - 1.0)
    mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
    #mask = cv2.medianBlur(mask.astype(np.uint8), 15)

    print(mask.shape)
    mask = np.squeeze(mask)
    cv2.imwrite('mask_blurred.png', mask*255)
    out = video * mask
    out = out.astype(np.uint8)

    return out


def get_in_range(fft_video: np.array, freq: np.array, freq_range: Tuple):
    # get components within the desired frequency range
    in_freq_range = np.argwhere(
        (freq > freq_range[0]) * (freq < freq_range[1])
    ).squeeze()
    fft_video = fft_video[in_freq_range, ...]
    freq = freq[in_freq_range, ...]
    log.debug(f"in_freq_range video: {fft_video.shape}")

    return in_freq_range, fft_video, freq


def mean_threshold(
    fft_video: np.array,
    freq: np.array,
    fft_mag: np.array,
    freq_range: Tuple,
    factor: Optional[float] = 1.05,
):
    """
    Threshold pixels if within the frequency range they exhibit a
    magnitude greater than a factor times the mean of magnitudes
    across all frequencies.

    Args:
        fft_video: fft of the video [N, H, W, C]
        freq: frequency components of the fft video [N,]
        fft_mag: magnitude of frequency components [N, H, W, C]
        factor: factor above the average magnitude at which to filter [float]
    """

    avg_mag = np.mean(fft_mag, axis=0)
    in_freq_range, fft_video_in_range, freq_in_range = get_in_range(
        fft_video, freq, freq_range
    )
    mask = np.any(fft_video_in_range > factor * avg_mag, axis=0)

    return mask


def max_threshold(
    fft_video: np.array, freq: np.array, fft_mag: np.array, freq_range: Tuple
):
    """
    Threshold pixels if within the frequency range lies the maximum
    magnitude across all frequencies.

    Args:
        fft_video: fft of the video [N, H, W, C]
        freq: frequency components of the fft video [N,]
        fft_mag: magnitude of frequency components [N, H, W, C]
    """

    max_freqs = np.argmax(fft_mag, axis=0)
    log.debug(max_freqs.shape)
    in_freq_range, fft_video_in_range, freq_in_range = get_in_range(
        fft_video, freq, freq_range
    )
    mask = np.isin(max_freqs, in_freq_range)

    return mask
