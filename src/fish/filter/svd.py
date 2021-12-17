
"""
Implements a singular value decomposition (SVD) filter which is used to
remove the high/med. frequency components induced by the turbine's motion
in the given video block
"""

from fish import logger
from fish.utils import Array


def svd_filter(video: Array["N,H,W,C", np.uint8],
               fps: int,
               freq_range: Optional[Tuple] = (1.5, 3.0),
               thresh_func: Optional[Callable[[Tuple],
                                              Array["N,H,W,C",
                                                    np.float32]]] = None
               ) -> Array["H,W,C", np.uint8]:
    pass
