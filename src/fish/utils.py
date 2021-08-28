
from typing import Tuple
import numpy as np

from fish import logger

def generate_sinusoid(freq: int, sampling_rate: int, shape: Tuple, length=float):
    '''
        Create a waveform
    '''

    u, v = shape
    t = np.linspace(0, length, int(sampling_rate*length))
    # 8-bit-valued waveform
    waveform = (np.sin( freq * (2*np.pi) * t )/2.) + .5

    logger.debug(f"\nSinusoid shape: {waveform.shape}")

    waveform_video = np.tile(waveform, (u, v, 1))

    logger.debug(f"\nSinusoid video shape: {waveform_video.shape}")

    assert (waveform_video.shape == (u, v, len(t))), \
        "Waveform video incorrectly shaped"

    return waveform_video

def generate_sinusoid_tile(freqs, element_shape, n_frames):
    '''
        Returns numpy array representing a video of 
        multi-frequency image tiles
    '''

    length = max([1/f for f in freqs])  # video length in seconds
    sampling_rate = n_frames / length   # samples per second
    
    waveforms = []
    for freq in freqs:
        waveforms.append(generate_sinusoid( freq=freq,
                                            sampling_rate=sampling_rate,
                                            shape=element_shape,
                                            length=length))
    
    output = np.concatenate(waveforms, axis=1)
    return output, sampling_rate, length