
from typing import Tuple
import numpy as np

from scipy.fft import fft, fftn, fftfreq, fftshift
from matplotlib import pyplot as plt

from fish import logger


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


def plot_time_domain_waveform(video, fps, pixel):
    '''
        Args:
            video: [N, W, H, C]
            fps: frames per second
            pixel: Tuple pixel location to plot
    '''

    assert(len(video.shape) == 4), "Video should be in format [N, W, H, C]"

    # get pixel-wise intensity over time
    time_domain = video[:, pixel[0], pixel[1], 0]
    N = len(time_domain)
    t = np.linspace(0, N*fps, N)

    # get fft of pixel over time
    fft_pixel = fft(time_domain, axis=0, workers=-1)
    fft_pixel = fftshift(fft_pixel, axes=0)  # 0 freq at center
    # For each component get the frequency center that it represents
    freq = fftfreq(N, d=1/fps)
    freq = fftshift(freq)
    logger.debug(f"fft_video {fft_pixel.shape}")

    # only operate on positive frequencies (greater than 0 plus fudge)
    freq_thresh = 0
    pos_range = np.argwhere(freq > freq_thresh).squeeze()
    fft_pixel = fft_pixel[pos_range, ...]
    logger.debug(f"pos_range video: {fft_pixel.shape}")
    freq = freq[pos_range, ...]

    # get magnitude and phase of each frequency component
    # magnitude of that frequency component
    mag = np.absolute(fft_pixel)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title(f"Pixel {pixel} intensity over time")
    plt.plot(t, time_domain)

    plt.subplot(1, 2, 2)
    plt.title(f"DFT of pixel {pixel}")
    plt.plot(freq, mag)
    plt.plot(freq, np.ones(mag.shape)*np.mean(mag))

    plt.show()

    return None
