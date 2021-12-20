
"""
Implements a singular value decomposition (SVD) filter which is used to
remove the high/med. frequency components induced by the turbine's motion
in the given video block
"""

import numpy as np
import imageio as iio
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt

import cv2

from fish import logger
from fish.utils import Array


def svd_filter(video: Array["N,H,W,C", np.uint8],
               fps: int,
               freq_range: Optional[Tuple] = (1.5, 3.0),
               thresh_func: Optional[Callable[[Tuple],
                                              Array["N,H,W,C",
                                                    np.float32]]] = None
               ) -> Array["H,W,C", np.uint8]:
   
    logger.debug(f"input video shape:{video.shape}");
    
    frame = video[0,:,:,0]
    frame_line = frame.ravel()
    M = frame_line

    # create big M matrix for SVD capped at 100 because of ram limits
    for i in range(1, 100):
        frame = video[i,:,:,0];
        frame_line = frame.ravel()
        M = np.vstack([M, frame_line])
    
    # SVD vals
    u, s, v = np.linalg.svd(M.T, full_matrices=False)
   
    # fram lines from the big M matrix
    low_rank = np.expand_dims(u[:,0],1)*s[0]*np.expand_dims(v[0,:],0)
    high_rank = np.expand_dims(u[:,99],1)*s[99]*np.expand_dims(v[99,:],0)


    # trying to take off vectores from either side 
    for i in range(1,47):
        low_rank += np.expand_dims(u[:,i],1)*s[i]*np.expand_dims(v[i,:],0)
    
    for i in range(98, 50, -1):
        high_rank += np.expand_dims(u[:,i],1)*s[i]*np.expand_dims(v[i,:],0)

    logger.debug(f"input video shape:{M.T.shape}");
    
    background = np.reshape(low_rank[:,0], (video.shape[1],video.shape[2]))
    foreground = np.reshape(high_rank[:,0], (video.shape[1],video.shape[2]))

    plt.imsave("background.png", background)
    plt.imsave("foreground.png", foreground)
    
    filter_writer = iio.get_writer('demo_svd.gif', mode = 'I', fps=fps)
    raw_writer = iio.get_writer('raw_video.gif', mode = 'I', fps=fps)

    for i in range(video.shape[0] - 60):
        frame_buffer = video[i,:,:,0] - background - foreground
        filter_writer.append_data(frame_buffer.astype(np.uint8))
        raw_writer.append_data(video[i,:,:,0])

    filter_writer.close()

    pass
