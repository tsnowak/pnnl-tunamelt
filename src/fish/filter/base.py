
from typing import Callable, Optional, Tuple
import numpy as np

from fish import logger
from fish.utils import Array


class OfflineFilter():

    def __init__(self,
                 video: Array["N,H,W,C", np.uint8],
                 fps: int):

        # control video type
        assert isinstance(video, np.ndarray), \
            f"Video is not np.ndarray {video.dtype}"
        self.video = video

        # ensure fps is not confused w/ period
        assert fps != 0, \
            f"fps is zero {fps}"
        self.fps = fps

        logger.info(f"Input video of shape {video.shape} @ {fps}FPS")

    def generate(self,):
        """ Calculate the mask/filter to apply
        """
        raise NotImplementedError()

    def apply(self, video: Optional[Array["N,H,W,C", np.uint8]] = None):
        """ Applies the mask/filter to the original video or given video/frame
        """
        raise NotImplementedError()


'''
class OnlineFilter():

    def __init__(self,
                 video_ptr: cv2.VideoReader,
                 fps: int):

        self.video_pointer = video_ptr
        self.fps = fps

    def __next__():
        pass

    def filter(self,):
        pass

    def load_video(self,):
        pass
'''
