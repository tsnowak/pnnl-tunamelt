
from typing import Tuple, TypeVar, Generic, TypeVar
import numpy as np

from scipy.fft import fft, fftn, fftfreq, fftshift
from matplotlib import pyplot as plt

from fish import logger

Shape = TypeVar("Shape")
DType = TypeVar("DType")


class Array(np.ndarray, Generic[Shape, DType]):
    """
        Numpy Array docstring use
        Ex:
        image: Array['H,W,3', np.uint8]
    """
    pass


def generate_sinusoid(freq: int, fps: int, shape: Tuple, length=float):
    '''
        Create a waveform
    '''

    u, v = shape
    t = np.linspace(0, length, int(fps*length))
    # 8-bit-valued waveform
    waveform = (np.sin(freq * (2*np.pi) * t)/2.) + .5

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
    fps = n_frames / length   # samples per second

    waveforms = []
    for freq in freqs:
        waveforms.append(generate_sinusoid(freq=freq,
                                           fps=fps,
                                           shape=element_shape,
                                           length=length))

    output = np.concatenate(waveforms, axis=1)
    return output, fps, length
