import numpy as np
import cv2

from turbx import log
from turbx.filter.sdft import sdft_filter
from turbx.utils import generate_sinusoid_tile


def test_filter_frequency():
    """
    Verify that specified frequencies get removed from video
    using a video of tiled flashing black and white squares
    which oscillate at differing frequencies
    """
    pass
