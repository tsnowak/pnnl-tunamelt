from typing import Optional, List, Tuple
import numpy as np
import cv2
import imageio
from scipy.ndimage.filters import gaussian_filter


from skimage.restoration import denoise_wavelet
from sklearn.preprocessing import normalize
from scipy.fft import fft, fftn, fftfreq, fftshift

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
            - video: video to filter - Optional(np.array [N, H, W, C])
            - fps: fps of video - Optional(int)
        """
        super().__init__(video, fps)
        if self.mask is None:
            log.debug(f"Did not generate {self.__class__} filter.")
        else:
            log.debug(f"Generated {self.__class__} filter mask.")
        self.fps = fps

    def filter(
        self,
        video: np.ndarray,
        fps: Optional[int] = None,
    ):
        """
        Applies the mask to filter the video
        Args:
            - video: video to filter - np.array [N, H, W, C]
            - fps: fps of video - int
        """
        if fps is None:
            if self.fps is None:
                raise ValueError("fps not given.")
            fps = self.fps

        self.calculate(video, fps)
        video = video.astype(np.float32)
        filtered_video = np.multiply(video, self.mask)
        filtered_video = filtered_video.astype(np.uint8)
        log.debug(f"Mean background filtered video of shape: {filtered_video.shape}")
        return filtered_video

    def calculate(
        self,
        video: np.ndarray,
        fps: int,
    ):
        """
        Calculates the filter mask
        Args:
            - video: video to filter - np.array [N, H, W, C]
            - fps: fps of video - int
        """
        self.fps = fps
        video = video.astype(np.float32)
        # calculate background
        mean = np.mean(video, axis=0)
        avg_value = np.mean(mean)

        # remove background
        diff = np.subtract(video, mean)  # TODO: still 4 dim?
        mask = diff > avg_value  # TODO: base it off unsigned magnitude?
        self.mask = mask

        return mask


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

        video = video.astype(np.float32)
        out = np.multiply(video, self.mask)
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
        video = video.astype(np.float16)
        std_val = np.max(np.std(video, axis=0))
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
        mask = video > (max_val - std_val)
        self.mask = mask

        log.debug(f"Generated intensity filter mask of shape: {mask.shape}")

        return mask


class ContourFilter:
    def __init__(
        self,
        video: Optional[np.ndarray] = None,
        min_area: int = 150,
        max_area: int = 1000,
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
            thresh = cv2.cvtColor(video[frame, ...], cv2.COLOR_HSV2BGR)
            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
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


## NOTE: Not implemented ##
class SimpleObjectTracking(OfflineFilter):
    """
    Detect contours, track, and keep those which have somewhat
    constant/reasonable velocity and permanance

    - contour detect > size
    - kalman filter, predict, update
    - if error too large, it's not an object
    """

    def __init__(
        self,
        video: Optional[np.ndarray] = None,
        fps: Optional[int] = None,
    ):

        # Meaning of the state vector
        # state is the centroid location, size, and speed + noise
        # state := [x, y, v_x, v_y, w, h]
        state_size = 6

        # Meaning of the measurement vector
        # observation is the centroid location and size + noise
        # observation = z := [x, y, w, h]
        measurement_size = 4

        # No controls; no control vector
        controls_size = 0

        kalman = cv2.KalmanFilter(
            dynamParams=state_size,
            measureParams=measurement_size,
            controlParams=controls_size,
        )
        measurement = np.array((measurement_size, 1), np.float32)
        prediction = np.zeros((measurement_size, 1), np.float32)

        state_transition_matrix = np.identity(n=state_size, dtype=np.float32)
        kalman.transitionMatrix = state_transition_matrix

        # state_size by measurement_size
        observation_matrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ],
            dtype=np.float32,
        )  # E201, E202
        kalman.measurementMatrix = observation_matrix

        # state/process noise
        # covariance of uncertainty in the state being what it is
        kalman.processNoiseCov = np.identity(n=state_size, dtype=np.float32) * 0.05

        # measurement/observation noise
        # covariance of uncertainty in the observation being what is measured
        kalman.measurementNoiseCov = np.identity(n=state_size, dtype=np.float32) * 0.03

    def object_check(
        self,
    ):
        """
        if not seen in CNTR, is not object
        """
        pass

    def apply(
        self,
        video: np.ndarray,
        fps: int,
    ):

        video = video.astype(np.float32)

        initialize_kf()

    def detect_contours(self, frame):
        pass

    def initialize_kf(
        self,
    ):
        pass

    def update_kf(self, kf, detections):
        pass
