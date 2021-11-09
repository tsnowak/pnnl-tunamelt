
from typing import Optional, List, Tuple
import numpy as np
import cv2

from skimage.restoration import denoise_wavelet
from sklearn.preprocessing import normalize
from scipy.fft import fft, fftn, fftfreq, fftshift

from fish import logger


def wavelet_denoising(x, method='BayesShrink'):
    '''
        ineffective on low-res videos
    '''

    x = denoise_wavelet(x, multichannel=True, convert2ycbcr=False,
                        method=method, mode='soft', rescale_sigma=True)

    return x


def crop_polygon(img, pts):
    '''
        crop polygon of image, and return with black background
    '''
    # (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = img[y:y+h, x:x+w].copy()

    # (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    # (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    return dst


def mean_filter(video, thresh):
    mean = np.mean(video, axis=0)
    abs_diff = np.abs(
        video - np.repeat(mean[np.newaxis, :, :], video.shape[0], axis=0))
    out = (abs_diff > thresh) * video
    return out


def fourier_filter(video, fps, freq_range: Optional[Tuple] = (1.5, 3.0), f_thresh=None):
    '''
    Generates a binary mask of pixels which change at a certain
    periodicity within the video

    Args:
        video: np.array (N, H, W, C) of video frames to process
        thresh: threshold in hertz of values to pass through the filter
    '''

    if f_thresh is None:
        f_thresh = max_threshold

    logger.debug(f"input video shape:{video.shape}")
    # take the fft of the video
    fft_video = fft(video, axis=0, workers=-1)
    fft_video = fftshift(fft_video, axes=0)  # 0 freq at center
    # For each component get the frequency center that it represents
    freq = fftfreq(video.shape[0], d=1/fps)
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
    #phase = np.angle(fft_video)
    logger.debug(f"magnitude array shape: {mag.shape}")

    #mask = average_threshold(fft_video, freq, mag, freq_range, factor=2)
    mask = f_thresh(fft_video, freq, mag, freq_range)
    logger.debug(f"per pixel mask: {mask.shape}")

    return mask


def get_in_range(fft_video, freq, freq_range):
    # get components within the desired frequency range
    in_freq_range = np.argwhere(
        (freq > freq_range[0]) * (freq < freq_range[1])).squeeze()
    fft_video = fft_video[in_freq_range, ...]
    freq = freq[in_freq_range, ...]
    logger.debug(f"in_freq_range video: {fft_video.shape}")

    return in_freq_range, fft_video, freq


def mean_threshold(fft_video, freq, fft_mag, freq_range, factor=1.25):
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


def max_threshold(fft_video, freq, fft_mag, freq_range):
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
