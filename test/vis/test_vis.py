
import numpy as np
from fish.utils import generate_sinusoid_tile
from fish.vis import plot_time_domain_waveform


def test_plot_time_domain_waveform():

    freqs = [9, 3, 1, 1/3, 1/9]
    element_shape = (10, 15)
    n_frames = 1000

    waveform, fps, length = generate_sinusoid_tile(freqs=freqs,
                                                   element_shape=element_shape,
                                                   n_frames=n_frames
                                                   )

    waveform = np.expand_dims(np.transpose(waveform, (2, 0, 1)), axis=-1)
    plot_time_domain_waveform(waveform, fps, (0, 0))

    return None
