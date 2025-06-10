from typing import Optional

import numpy as np

from tunamelt import log


class OfflineFilter:
    """
    Split filter genreration into two stages:
        Generate: calculates the filter mask from the given video/fps
        Apply: applies the generated mask to the given video/fps
    """

    def __init__(
        self,
        video: Optional[np.ndarray] = None,
        fps: Optional[int] = None,
    ):
        # control video type
        if video is not None:
            assert isinstance(
                video, np.ndarray
            ), f"Video must be np.ndarray {video.dtype}"

        # TODO: ensure fps is not confused w/ period
        if fps is not None:
            assert fps != 0, f"fps can't be zero {fps}"

        self.mask = None
        if video is not None and fps is not None:
            self.mask = self.calculate(video, fps)
            log.debug(
                f"Mask generated {self.mask.shape} from video of shape {video.shape} @ {fps}FPS"
            )

    def calculate(
        self,
        video: np.ndarray,
        fps: int,
    ):
        """Calculate the mask/filter to apply"""
        raise NotImplementedError()

    def filter(self, video: np.ndarray, fps: int):
        """Calculates (if not already done so) and applies the mask/filter to the given video/frame"""
        raise NotImplementedError()


class OnlineFilter:
    def __init__(
        self,
    ):
        """
        Filter has no apriori knowledge of video - is running live and stores current state
        """

        self.state = None

    def filter(self, frame: np.ndarray, fps: Optional[int]):
        """
        Filters a single frame using the current state and updates the
        current state of the filter
        """
        raise NotImplementedError()

    def reset_filter(
        self,
    ):
        """
        Resets the state of the filter
        """
        raise NotImplementedError()
