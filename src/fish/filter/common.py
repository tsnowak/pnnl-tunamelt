
from typing import Optional, List, Tuple
import numpy as np
import cv2
import imageio
from scipy.ndimage.filters import gaussian_filter


from skimage.restoration import denoise_wavelet
from sklearn.preprocessing import normalize
from scipy.fft import fft, fftn, fftfreq, fftshift

from fish import logger
from fish.filter.base import OfflineFilter
from fish.utils import Array


class MeanFilter(OfflineFilter):

    def __init__(self, video: Array["N,H,W,C", np.uint8], fps):
        super().__init__(video, fps)
        self.video = video.astype(np.float32)
        logger.debug(f"Initialized {self.__class__} filter.")

    def apply(self, video=None):
        if video is None:
            video = self.video
        mask = self.generate()
        out = np.multiply(video, mask)
        out = out.astype(np.uint8)
        logger.debug(f"Mean background filtered video of shape: {out.shape}")
        return out

    def generate(self,):
        # calculate background
        mean = np.mean(self.video, axis=0)
        avg_value = np.mean(mean)

        # remove background
        diff = np.subtract(self.video, mean)
        mask = diff > avg_value
        return mask


class IntensityFilter(OfflineFilter):

    def __init__(self, video: Array["N,H,W,C", np.uint8], fps, n=500):
        super().__init__(video, fps)
        self.video = video.astype(np.float32)
        self.n = n

        logger.debug(f"Initialized {self.__class__} filter.")

    def apply(self, video=None):

        if video is None:
            video = self.video
        mask = self.generate()
        out = np.multiply(video, mask)
        out = out.astype(np.uint8)
        logger.debug(f"Intensity filtered video of shape: {out.shape}")
        return out

    def generate(self,):
        std_val = np.max(np.std(self.video, axis=0))
        max_val = np.max(self.video, axis=0)

        logger.debug(f"Max value: {np.max(max_val)}")
        # take average of n-largest pixel maxima
        max_val = np.mean(np.sort(max_val, axis=None)[-self.n:])
        logger.debug(f"std and max: {std_val, max_val}")

        # over entire image
        # only one pixel passes this
        #std = np.std(video)
        #max = np.max(video)
        #print(std, max_val)

        # only keep pixels in the video that are within std of their max
        # apply mask in apply
        mask = self.video > (max_val - std_val)

        logger.debug(f"Generated intensity filter mask of shape: {mask.shape}")

        return mask
