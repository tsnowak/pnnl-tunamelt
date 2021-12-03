
"""
    Generates two videos which illustrate the efficacy of our method. The first
    video is a video of blocks of pixels oscillating at fixed frequencies
    defined by the variable freqs. The second video is this original
    video with the filtered version of it below. The filtered
    version is generated using our method
    which identifies pixels in a video that oscillate within a given range
    of frequencies.

    Author: Theodore Nowak
"""

from pathlib import Path
import numpy as np
import imageio as iio
from skimage import img_as_ubyte

from fish import REPO_PATH
from fish.utils import generate_sinusoid_tile
from fish.filter.dft import dft_filter


def main():

    # define waveform properties
    freqs = [9, 3, 1, 1/3, 1/9]
    element_shape = (10, 15)
    n_frames = 1000
    filter_freq_range = (.5, 8.5)

    # define place to save outputs
    image_path = Path(REPO_PATH + '/experiments/dft/outputs')
    Path(image_path).mkdir(exist_ok=True)

    # Before filtering

    # Generate waveform video
    waveform, fps, length = generate_sinusoid_tile(freqs=freqs,
                                                   element_shape=element_shape,
                                                   n_frames=n_frames
                                                   )
    _, _, n = waveform.shape

    # write waveform video to gif
    writer = iio.get_writer(str(image_path) + '/demo_sine.gif',
                            mode='I', fps=int(1000/fps))
    for i in range(n):
        writer.append_data(img_as_ubyte(waveform[:, :, i]))
    writer.close()

    # convert waveform video to format of filter
    video = np.transpose(waveform, (2, 0, 1))
    video = np.expand_dims(video, axis=-1)

    # apply filter: mask - 1 in range, 0 out of range; inv_mask - 0 in range, 1 out of range
    mask = dft_filter(video=video, fps=fps, freq_range=filter_freq_range)
    inv_mask = np.abs(mask - 1.)

    # write filtered waveform video to file
    writer = iio.get_writer(str(image_path) + '/demo_filtered_sine.gif',
                            mode='I', fps=int(1000/fps))
    for i in range(n):
        combined_frame = np.concatenate(
            [waveform[:, :, i], waveform[:, :, i]*inv_mask.squeeze()], axis=0)
        writer.append_data(img_as_ubyte(combined_frame))
    writer.close()


if __name__ == "__main__":
    main()
