"""
Implementation of the sliding discrete fourier transform. Used to filter out 
pixels oscillating within a range of frequencies on live video.
"""

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
from river import utils
from fish import logger
from fish.utils import Array


def sdft_filter(video: Array["N,H,W,C", np.uint8],
                fps: int,
                windowSize: int = 40, 
                freq_range: Optional[Tuple] = (1.5, 3.0),
                thresh_func: Optional[Callable[[Tuple],
                                               Array["N,H,W,C",
                                                     np.float32]]] = None
                ) -> Array["H,W,C", np.uint8]:
    """
    Uses the sliding dft in order to generate filtered video,
    at the moment just takes in the whole video and a frame size and
    just writes filtered video

    In the future it should take in a mask, N frames, and a window size of N
    and return the new mask
    """

    if thresh_func is None:
        thresh_func = max_threshold

    logger.debug(f"input video shape:{video.shape}")
    fft_video = utils.SDFT(windowSize)
    # take the initial fft of the video

    for i in range(40):
        fft_video = fft_video.update(video[i])   

    ''' 
    for n in range(windowSize, video.shape[0]):
        delta = np.subtract(video[n, :, :, :], video[n-windowSize, :, :, :])
        for i in range(windowSize):
            fft_video[i] = (fft_video[i] + delta)*(np.exp(2j*np.pi*i/windowSize))
    '''
    fft_video = fftshift(fft_video, axes=0)  # 0 freq at center
    # For each component get the fr1equency center that it represents
    freq = fftfreq(windowSize, d=1/fps)
    freq = fftshift(freq)
    logger.debug(f"fft_video {fft_video.shape}")

    # only operate on positive frequencies (greater than 0 plus fudge)
    freq_thresh = 1e-4
    pos_range = np.argwhere(freq > (0+freq_thresh)).squeeze()
    fft_video = fft_video[pos_range, ...]
    logger.debug(f"pos_range video: {fft_video.shape}")
    freq = freq[pos_range, ...]

    # get magnitude and phase of each frequency component
    # magnitude of that frequency component
    mag = np.absolute(fft_video)
    # phase between sine (im) and cosine (re) of that freq. component
    # phase = np.angle(fft_video)
    logger.debug(f"magnitude array shape: {mag.shape}")

    # mask = average_threshold(fft_video, freq, mag, freq_range, factor=2)
    mask = thresh_func(fft_video, freq, mag, freq_range)
    logger.debug(f"per pixel mask: {mask.shape}")

    plt.imsave("mask.png", mask[:,:,0], cmap='Greys')

    return mask


def get_in_range(fft_video: np.array, freq: np.array, freq_range: Tuple):
    # get components within the desired frequency range
    in_freq_range = np.argwhere(
        (freq > freq_range[0]) * (freq < freq_range[1])).squeeze()
    fft_video = fft_video[in_freq_range, ...]
    freq = freq[in_freq_range, ...]
    logger.debug(f"in_freq_range video: {fft_video.shape}")

    return in_freq_range, fft_video, freq


def mean_threshold(fft_video: np.array,
                   freq: np.array,
                   fft_mag: np.array,
                   freq_range: Tuple,
                   factor: Optional[float] = 1.05):
    '''
        Threshold pixels if within the frequency range they exhibit a
        magnitude greater than a factor times the mean of magnitudes
        across all frequencies.

        Args:
            fft_video: fft of the video [N, H, W, C]
            freq: frequency components of the fft video [N,]
            fft_mag: magnitude of frequency components [N, H, W, C]
            factor: factor above the average magnitude at which to filter [float]
    '''

    avg_mag = np.mean(fft_mag, axis=0)
    in_freq_range, fft_video_in_range, freq_in_range = get_in_range(
        fft_video, freq, freq_range)
    mask = np.any(fft_video_in_range > factor*avg_mag, axis=0)

    return mask


def max_threshold(fft_video: np.array,
                  freq: np.array,
                  fft_mag: np.array,
                  freq_range: Tuple):
    '''
        Threshold pixels if within the frequency range lies the maximum
        magnitude across all frequencies.

        Args:
            fft_video: fft of the video [N, H, W, C]
            freq: frequency components of the fft video [N,]
            fft_mag: magnitude of frequency components [N, H, W, C]
    '''

    max_freqs = np.argmax(fft_mag, axis=0)
    logger.debug(max_freqs.shape)
    in_freq_range, fft_video_in_range, freq_in_range = get_in_range(
        fft_video, freq, freq_range)
    mask = np.isin(max_freqs, in_freq_range)

    return mask
