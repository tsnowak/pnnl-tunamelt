
import numpy as np
import cv2

from fish import logger
from fish.filter.sdft import sdft_filter
from fish.utils import generate_sinusoid_tile


def test_filter_frequency():
    '''
        Verify that specified frequencies get removed from video
        using a video of tiled flashing black and white squares
        which oscillate at differing frequencies
    '''
    pass
