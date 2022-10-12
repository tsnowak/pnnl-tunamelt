from typing import Optional, List, Tuple
import numpy as np
import cv2
import pywt
from turbx import log
from turbx.filter.base import OfflineFilter


class MeanFilter(OfflineFilter):
    def __init__(
        self,
        video: Optional[np.ndarray] = None,
        fps: Optional[int] = None,
    ):
        """
        Removes static background by zeroing pixels in frames which .
        Args:
            - video: video to filter - Optional(np.array [N, H, W, C] or [N, H, W])
            - fps: fps of video - Optional(int)

        Memory Optimized
        """
        super().__init__(video, fps)
        if self.mask is None:
            log.debug(f"Did not generate {self.__class__} filter.")
        else:
            log.debug(f"Generated {self.__class__} filter mask.")
        self.fps = fps
        self.out_format = "GRAY"

    def filter(
        self,
        video: np.ndarray,
        fps: Optional[int] = None,
    ):
        """
        Applies the mask to filter the video
        Args:
            - video: video to filter - np.array [N, H, W, C] or [N, H, W]
            - fps: fps of video - int
        """
        if fps is None:
            if self.fps is None:
                raise ValueError("fps not given.")
            fps = self.fps


        if len(video.shape) == 4:
            self.calculate(video[..., 2], fps)
            filtered_video = np.multiply(video, np.stack([self.mask] * 3, axis=3))
        elif len(video.shape) == 3:
            self.calculate(video, fps)
            filtered_video = np.multiply(video, self.mask)
        else:
            raise ValueError(
                "Input video is neither NxWxHxC nor NxWxH. Verify its structure."
            )

        filtered_video = filtered_video.astype(np.uint8)
        log.debug(f"Mean background filtered video of shape: {filtered_video.shape}")
        return filtered_video

    def calculate(
        self,
        value_channel: np.ndarray,
        fps: int,
    ):
        """
        Calculates the filter mask
        Args:
            - value_channel: value_channel to filter - np.array [N, H, W, C]
            - fps: fps of value_channel - int
        """
        self.fps = fps
<<<<<<< HEAD
        # calculate background
        mean = np.mean(video, axis=0, dtype=np.float16)
        avg_value = np.mean(mean)
=======
        value_channel = value_channel.astype(np.float16)
        # calculate background
        # mean := frame of average values from throughout the value_channel
        mean = np.mean(value_channel, axis=0)
        var = np.var(value_channel.astype(np.float64), axis=0)
        # avg_value := single average pixel value for the value_channel
        # avg_mean = np.mean(mean)
        avg_var = np.mean(var)
>>>>>>> master

        # remove background
        # only compare h,s,V - Value values (N, W, H, 1)
        value_channel = np.subtract(value_channel, mean)
        # np.boolean mask, N, W, H, 1
        # difference values > per-pixel standard dev
        self.mask = value_channel > 2 * np.sqrt(var)

        return self.mask


class DeNoiseFilter(OfflineFilter):
    def __init__(
        self,
        video: Optional[np.ndarray] = None,
        fps: Optional[int] = None,
    ):
        """
        Removes noise from the video.
        Args:
            - video: video to filter - Optional(np.array [N, H, W, C] or [N, H, W])
            - fps: fps of video - Optional(int)

        Memory Optimized
        """
        super().__init__(video, fps)
        if self.mask is None:
            log.debug(f"Did not generate {self.__class__} filter.")
        else:
            log.debug(f"Generated {self.__class__} filter mask.")
        self.fps = fps
        self.out_format = "GRAY"

    def filter(
        self,
        video: np.ndarray,
        fps: Optional[int] = None,
    ):
        """
        Applies the mask to filter the video
        Args:
            - video: video to filter - np.array [N, H, W, C] or [N, H, W]
            - fps: fps of video - int
        """
        if fps is None:
            if self.fps is None:
                raise ValueError("fps not given.")
            fps = self.fps

        video = video.copy()
        if len(video.shape) == 4:
            filtered_video = self.calculate(video[..., 2], fps)
        elif len(video.shape) == 3:
            filtered_video = self.calculate(video, fps)
        else:
            raise ValueError(
                "Input video is neither NxWxHxC nor NxWxH. Verify its structure."
            )
        log.debug(f"Noise filtered video of shape: {filtered_video.shape}")
        return filtered_video

    def calculate(
        self,
        value_channel: np.ndarray,
        fps: int,
    ):
        """
        Calculates the filter mask
        Args:
            - value_channel: value_channel to filter - np.array [N, H, W, C]
            - fps: fps of value_channel - int
        """
        self.fps = fps
        ## per frame NlMeansDenoising -> VERY SLOW
        for i, frame in enumerate(value_channel):
            frame = cv2.fastNlMeansDenoising(frame, None, 20, 21, 41)
            value_channel[i, ...] = frame
        ## time-windowed NlMeansDenoising -> VERY SLOW
        # batch_size = 5
        # for i in range(len(value_channel)):
        #    start = i
        #    end = i + 5
        #    diff = (len(value_channel) - 1) - start
        #    avg_frame = 2
        #    # adjust for end of video
        #    if diff < 5:
        #        start = i - (5 - diff)  # set start to 5 from end
        #        end = len(value_channel) - 1  # end of video
        #        avg_frame = i  # set to current frame
        #    batch = value_channel[start:end, ...]
        #    print(len(batch))
        #    frame = cv2.fastNlMeansDenoisingMulti(
        #        batch, avg_frame, batch_size, None, 4, 7, 35
        #    )
        #    value_channel[i, ...] = frame

        ## Wavelet denoising

        print(value_channel.shape)

        return value_channel


class IntensityFilter(OfflineFilter):
    def __init__(
        self,
        video: Optional[np.ndarray] = None,
        fps: Optional[int] = None,
        n: Optional[int] = 500,
    ):
        """
        Removes pixels are below
        """
        self.n = n
        super().__init__(video, fps)
        if self.mask is None:
            log.debug(f"Did not generate {self.__class__} filter.")
        else:
            log.debug(f"Generated {self.__class__} filter mask.")
        self.fps = fps
        self.out_format = "GRAY"

    def filter(
        self,
        video: np.ndarray,
        fps: Optional[int] = None,
    ):
        if fps is None:
            if self.fps is None:
                raise ValueError("fps not given.")
            fps = self.fps
        self.mask = self.calculate(video, fps)

        out = np.multiply(video, self.mask, dtype=np.uint8)
        out = out.astype(np.uint8)
        log.debug(f"Intensity filtered video of shape: {out.shape}")
        return out

    def calculate(
        self,
        video: np.ndarray,
        fps: int,
    ):
        self.fps = fps
        # video = video.astype(np.float32)
        std_val = np.max(np.std(video, axis=0, dtype=np.float16))
        max_val = np.max(video, axis=0)

        log.debug(f"Max value: {np.max(max_val)}")
        # take average of n-largest pixel maxima
        max_val = np.mean(np.sort(max_val, axis=None)[-self.n :])
        log.debug(f"std and max: {std_val, max_val}")

        # over entire image
        # only one pixel passes this
        # std = np.std(video)
        # max = np.max(video)
        # print(std, max_val)

        # only keep pixels in the video that are within std of their max
        # apply mask in apply
        self.mask = video > (max_val - std_val)

        log.debug(f"Generated intensity filter mask of shape: {self.mask.shape}")

        return self.mask


class SpeckleFilter(OfflineFilter):
    def __init__(
        self,
        video: Optional[np.ndarray] = None,
        fps: Optional[int] = None,
        kernel_shape: Optional[Tuple] = (5, 5),
    ):
        """
        Try to remove speckles by building up then eroding
        """
        super().__init__(video, fps)
        if self.mask is None:
            log.debug(f"Did not generate {self.__class__} filter.")
        else:
            log.debug(f"Generated {self.__class__} filter mask.")
        self.fps = fps
        self.out_format = "GRAY"
        self.kernel_shape = kernel_shape

    def filter(
        self,
        video: np.ndarray,
        fps: Optional[int] = None,
    ):
        if fps is None:
            if self.fps is None:
                raise ValueError("fps not given.")
            fps = self.fps
        out = self.calculate(video, fps)
        log.debug(f"BuildUpErode filtered video of shape: {out.shape}")
        return out

    def calculate(
        self,
        video: np.ndarray,
        fps: int,
    ):
        video = video.copy()
        self.fps = fps
        kernel = np.ones(self.kernel_shape, np.uint8)
        # dilation_kernel = np.ones((self.dilation, self.dilation), np.uint8)
        # erosion_kernel = np.ones((self.erosion, self.erosion), np.uint8)
        # diff_kernel = np.ones(
        #    (self.erosion - self.dilation, self.erosion - self.dilation), np.uint8
        # )
        for idx, frame in enumerate(video):
            opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
            # frame = cv2.erode(frame, erosion_kernel, iterations=1)
            # frame = cv2.dilate(frame, dilation_kernel, iterations=1)
            # frame = cv2.dilate(frame, diff_kernel, iterations=1)
            video[idx, ...] = opening

        return video


class ContourFilter:
    def __init__(
        self,
        video: Optional[np.ndarray] = None,
<<<<<<< HEAD
        min_area: int = 100,
        max_area: int = 1000,
=======
        min_area: int = 150,
        max_area: int = 1200,
>>>>>>> master
    ):
        """
        Detect contours of a certain size
        """
        self.min_area = min_area
        self.max_area = max_area

    def filter(
        self,
        video: np.ndarray,
    ):
        return self.calculate(video)

    def calculate(
        self,
        video: np.ndarray,
    ):
        boxes_per_frame = []
        for frame in range(len(video)):
            boxes = []
            # find contours
            if len(video.shape) == 4:
                thresh = cv2.cvtColor(video[frame, ...], cv2.COLOR_HSV2BGR)
                thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
            elif len(video.shape) == 3:
                thresh = video[frame, ...]
            contours, heirachy = cv2.findContours(
                image=thresh,
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )

            for cont in contours:
                rect = cv2.boundingRect(cont)
                area = rect[2] * rect[3]  # h*w
                if area >= self.min_area and area <= self.max_area:
                    boxes.append(rect)

            boxes_per_frame.append(boxes)

        return boxes_per_frame
