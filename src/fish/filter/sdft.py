"""
Implementation of the sliding discrete fourier transform. Used to filter out 
pixels oscillating within a range of frequencies on live video.
"""

import numpy as np

from fish import logger
from fish.utils import Array


def sdft_filter(video: Array["N,H,W,C", np.uint8],
                fps: int,
                freq_range: Optional[Tuple] = (1.5, 3.0),
                thresh_func: Optional[Callable[[Tuple],
                                               Array["N,H,W,C",
                                                     np.float32]]] = None
                ) -> Array["H,W,C", np.uint8]:
    pass
