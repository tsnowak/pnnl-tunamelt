"""
Implementation of DFT used to mask pixels exhibiting certain frequencies
"""
from typing import Optional, Tuple

import numpy as np
from scipy.fft import fft, fftfreq, fftshift

from fish.filter.base import OfflineFilter
from fish import logger
from fish.utils import Array


class DFTFilter(OfflineFilter):
    """Generates a binary mask of pixels which change at a certain
    periodicity within the video

    Args:
        video (np.array): [N, H, W, C] Video to process
        fps (int): Frames per seconds of the video
        freq_range (Optional[Tuple], optional): Range of frequencies to detect. Defaults to (1.5, 3.0).
        thresh_func (Callable, optional): Function used to determine whether frequency component is sufficiently high. Defaults to None.

    Returns:
        np.array: [H, W, C] Binary mask with pixels exhibiting frequencies in freq_range = 1
    """

    def __init__(self,
                 video: Array["N,H,W,C", np.uint8],
                 fps: int,
                 freq_range:  Optional[Tuple] = (1.5, 3.0),
                 thresh_func: Optional[str] = "max"):

        super().__init__(video, fps)

        # range of frequences to filter
        self.freq_range = freq_range

        # method used to determine when magnitude is above threshold
        if thresh_func == "max":
            thresh_func = self._max_threshold
        elif thresh_func == "mean":
            thresh_func = self._mean_threshold
        else:
            thresh_func = self._max_threshold

        self.thresh_func = thresh_func
        logger.debug(f"Using {thresh_func.__name__} thresholding function.")

        logger.info(f"Initialized {self.__class__} filter.")

    def filter(self,) -> Array["H,W,1", np.bool_]:

        # take the fft of the video
        fft_video = fft(self.video, axis=0, workers=-1)
        fft_video = fftshift(fft_video, axes=0)  # 0 freq at center
        # For each component get the frequency center that it represents
        freq = fftfreq(self.video.shape[0], d=1/self.fps)
        freq = fftshift(freq)
        logger.debug(f"FFT'd Video shape {fft_video.shape}")

        # only operate on positive frequencies (greater than 0 plus fudge)
        freq_thresh = 1e-4
        pos_range = np.argwhere(freq > (0+freq_thresh)).squeeze()
        fft_video = fft_video[pos_range, ...]
        logger.debug(f"Positive frequency video shape: {fft_video.shape}")
        freq = freq[pos_range, ...]

        # get magnitude and phase of each frequency component
        # magnitude of that frequency component
        mag = np.absolute(fft_video)
        # phase between sine (im) and cosine (re) of that freq. component
        # phase = np.angle(fft_video)
        logger.debug(f"FFT magnitude shape: {mag.shape}")

        # mask = average_threshold(fft_video, freq, mag, freq_range, factor=2)
        mask = self.thresh_func(fft_video, freq, mag, self.freq_range)

        assert mask.shape[:2] == self.video.shape[1:3], \
            f"Mask shape differs from video shape \
                \nMask: {mask.shape[:2]} \
                \nVideo: {self.video.shape[1:2]}"

        assert mask.dtype == np.bool_, \
            f"Mask dtype is not bool\
                \n{mask.dtype}"

        return mask

    def _mean_threshold(self, fft_video: np.array,
                        freq: np.array,
                        fft_mag: np.array,
                        freq_range: Tuple,
                        factor: Optional[float] = 1.25):
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
        in_freq_range, fft_video_in_range, freq_in_range = self._get_in_range(
            fft_video, freq, freq_range)
        mask = np.any(fft_video_in_range > factor*avg_mag, axis=0)
        logger.debug(f"Mean thresholded mask shape: {mask.shape}")

        return mask

    def _max_threshold(self, fft_video: np.array,
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
        in_freq_range, fft_video_in_range, freq_in_range = self._get_in_range(
            fft_video, freq, freq_range)
        mask = np.isin(max_freqs, in_freq_range)
        logger.debug(f"Max thresholded mask shape: {mask.shape}")

        return mask

    def _get_in_range(self, fft_video: np.array, freq: np.array, freq_range: Tuple):
        """Returns frequencies from FFT that are within the given frequency range

        Args:
            fft_video (np.array): fft of the original video
            freq (np.array): frequency bins of the fft
            freq_range (Tuple): range of frequencies to keep

        Returns:
            in_freq_range: [description]
            fft_video: [description]
            freq: [description]
        """
        in_freq_range = np.argwhere(
            (freq >= freq_range[0]) * (freq <= freq_range[1])).squeeze()
        fft_video = fft_video[in_freq_range, ...]
        freq = freq[in_freq_range, ...]
        logger.debug(
            f"Shape of mask of pixels within the frequency range: {fft_video.shape}")

        return in_freq_range, fft_video, freq
