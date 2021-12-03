
from typing import Optional, List, Tuple
import numpy as np
import cv2
import imageio
from scipy.ndimage.filters import gaussian_filter


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


def mean_filter(video: np.array) -> Tuple[np.array, np.array]:
    video = video.astype(np.float32)

    # calculate background
    mean = np.mean(video, axis=0)
    avg_value = np.mean(mean)

    # remove background
    diff = np.subtract(video, mean)
    out = np.multiply(video, diff > avg_value)

    out = out.astype(np.uint8)
    return out, mean


def intensity_filter(video: np.array) -> np.array:
    video = video.astype(np.float32)

    # calculate the std and max of each pixel over time
    # per pixel
    std = np.max(np.std(video, axis=0))
    max = np.max(video, axis=0)

    print(np.max(max))
    # take average of n-largest pixel maxima
    n = 500
    max = np.mean(np.sort(max, axis=None)[-n:])
    print(std, max)

    # over entire image
    # only one pixel passes this
    #std = np.std(video)
    #max = np.max(video)
    #print(std, max)

    # keep pixels in the video that are within std of their max
    out = np.multiply(video, video > (max - std))

    out = out.astype(np.uint8)
    return out


def volume_filter(video: np.array):
    # can calculate probable volume range of clusters in the video
    # given their range, pixel area, and AC intrinsics
    pass
