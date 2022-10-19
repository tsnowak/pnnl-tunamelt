"""
Implementation of DFT used to mask pixels exhibiting certain frequencies
"""
from typing import Optional, Tuple
import cv2
import numpy as np
from turbx import log
from turbx.filter.base import OfflineFilter
from scipy.fft import fft, fftfreq, fftshift


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

    def __init__(
        self,
        video: Optional[np.ndarray] = None,
        fps: Optional[int] = None,
        freq_range: Optional[Tuple] = (1.5, 3.0),
        thresh_func: Optional[str] = "max",
        mask_smoothing: Optional[int] = 9,
    ):

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
        log.debug(f"Using {thresh_func.__name__} thresholding function.")

        super().__init__(video, fps)
        if self.mask is None:
            log.debug(f"Did not generate {self.__class__} filter.")
        else:
            log.debug(f"Generated {self.__class__} filter mask.")

        self.fps = fps
        self.out_format = "GRAY"
        self.mask_smoothing = mask_smoothing

    def filter(
        self,
        video: np.ndarray,
        fps: Optional[int] = None,
    ):
        if fps is None:
            if self.fps is None:
                raise ValueError("fps must be given if filter.fps is not set.")
            fps = self.fps

        self.calculate(video, fps)
        # video = video.astype(np.float32)
        cv2.imwrite("dft_mask_original.png", self.mask * 255)
        # added smoothing to turbine mask to prevent noisy cancellation
        self.mask = cv2.medianBlur(
            np.expand_dims(self.mask, axis=-1).astype("uint8"), self.mask_smoothing
        )
        inv_mask = np.abs(self.mask - 1.0)

        assert (
            video.shape[1:3] == self.mask.shape[:2]
        ), f"Incompatible video shape for generated filter.\nVideo shape: {video.shape}\nFilter shape:{self.mask.shape}"

        cv2.imwrite("dft_mask_smooth.png", self.mask * 255)
        out = video * inv_mask
        out = out.astype(np.uint8)
        log.debug(f"Returning filtered video of shape {out.shape}")
        log.debug(f"Video dtype {out.dtype}")

        return out

    def calculate(
        self,
        video: np.ndarray,
        fps: int,
    ) -> np.ndarray:

        video = video.astype(np.float32)
        # take the fft of the video
        video_fft = fft(video, axis=0, workers=-1)
        video_fft = fftshift(video_fft, axes=0)  # 0 freq at center
        # For each component get the frequency center that it represents
        freq_bins = fftfreq(video.shape[0], d=(1.0 / fps))
        freq_bins = fftshift(freq_bins)
        log.debug(f"FFT'd Video shape {video_fft.shape}")

        # only operate on positive frequencies (greater than 0 plus fudge)
        pos_video_fft, pos_freq_bins = self._extract_positive_bins(video_fft, freq_bins)

        # get magnitude and phase of each frequency component
        # magnitude of that frequency component
        mag = np.absolute(pos_video_fft)
        # phase between sine (im) and cosine (re) of that freq. component
        # phase = np.angle(fft_video)
        log.debug(f"FFT magnitude shape: {mag.shape}")

        # remove pixels based on some thresholding strategy
        # mask = average_threshold(fft_video, freq, mag, freq_range, factor=2)
        mask = self.thresh_func(pos_video_fft, pos_freq_bins, mag, self.freq_range)

        assert (
            mask.shape[:2] == video.shape[1:3]
        ), f"Mask shape differs from video shape \
                \nMask: {mask.shape[:2]} \
                \nVideo: {video.shape[1:2]}"

        assert (
            mask.dtype == np.bool_
        ), f"Mask dtype is not bool\
                \n{mask.dtype}"

        self.mask = mask

        return mask

    def _mean_threshold(
        self,
        fft_video: np.array,
        freq: np.array,
        fft_mag: np.array,
        freq_range: Tuple,
        factor: Optional[float] = 1.25,
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
        in_freq_range, fft_video_in_range, freq_in_range = self._get_in_range(
            fft_video, freq, freq_range
        )
        mask = np.any(fft_video_in_range > factor * avg_mag, axis=0)
        log.debug(f"Mean thresholded mask shape: {mask.shape}")

        return mask

    def _max_threshold(
        self, fft_video: np.array, freq: np.array, fft_mag: np.array, freq_range: Tuple
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
        in_freq_range, fft_video_in_range, freq_in_range = self._get_in_range(
            fft_video, freq, freq_range
        )
        mask = np.isin(max_freqs, in_freq_range)
        log.debug(f"Max thresholded mask shape: {mask.shape}")

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
            (freq >= freq_range[0]) * (freq <= freq_range[1])
        ).squeeze()
        fft_video = fft_video[in_freq_range, ...]
        freq = freq[in_freq_range, ...]
        log.debug(
            f"Shape of mask of pixels within the frequency range: {fft_video.shape}"
        )

        return in_freq_range, fft_video, freq

    def _extract_positive_bins(self, video_fft, freq_bins, FREQ_THRESH=1e-4):
        pos_range = np.argwhere(freq_bins > (0 + FREQ_THRESH)).squeeze()
        video_fft = video_fft[pos_range, ...]
        log.debug(f"Positive frequency video shape: {video_fft.shape}")
        freq_bins = freq_bins[pos_range, ...]
        return video_fft, freq_bins
