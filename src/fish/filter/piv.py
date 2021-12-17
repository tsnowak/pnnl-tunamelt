
"""
Implements a particle image velocimetry (PIV) filter which is used to
detect circular motion in the video, detecting the turbine motion, and 
remove it from the video
"""

from fish import logger
from fish.utils import Array


def piv_filter(video: Array["N,H,W,C", np.uint8],
               fps: int,
               freq_range: Optional[Tuple] = (1.5, 3.0),
               thresh_func: Optional[Callable[[Tuple],
                                              Array["N,H,W,C",
                                                    np.float32]]] = None
               ) -> Array["H,W,C", np.uint8]:
    pass
