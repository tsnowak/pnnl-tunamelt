
import numpy as np
import cv2

from fish import logger
from fish.filter.dft import DFTFilter
from fish.utils import generate_sinusoid_tile


def test_filter_frequency():
    '''
        Verify that specified frequencies get removed from video
        using a video of tiled flashing black and white squares
        which oscillate at differing frequencies
    '''

    freqs = np.array([9, 3, 1, 1/3, 1/9], dtype=np.float32)
    filter_freq_range = (.5, 9)
    element_shape = (10, 15)
    n_frames = 1000

    waveform, fps, length = generate_sinusoid_tile(freqs=freqs,
                                                   element_shape=element_shape,
                                                   n_frames=n_frames)

    # convert to format of filter
    video = np.transpose(waveform, (2, 0, 1))
    video = np.expand_dims(video, axis=-1)

    logger.debug(f"\nInput video shape: {video.shape}")
    dft = DFTFilter(video, fps, freq_range=filter_freq_range)
    mask = dft.filter()

    # verify mask shape
    assert mask.shape[:2] == video.shape[1:3], \
        f"Mask and original video differ in H,W \
            \nMask: {mask.shape[:2]} \
            \nVideo: {video.shape[1:3]}"

    # verify mask results
    # fetch frequencies that should be masked
    freqs_in_range = (
        freqs <= filter_freq_range[1])*(freqs >= filter_freq_range[0])
    # Get the indices in the waveform that contain these frequencies
    indices = np.squeeze(np.argwhere(freqs_in_range == True))
    # step through waveform bins and verify pixels are correctly masked
    for i in range(len(freqs)):
        bin_is_one = np.all(
            mask[:, element_shape[1]*i:element_shape[1]*(i+1), :] == 1)
        logger.debug(
            f"Mask value for waveform bin {i} equals 1? {bin_is_one}")
        assert (bin_is_one) == (i in indices), \
            f"Mask did not remove all frequencies in range.\
                \nVideo frequencies: {freqs}\
                \nIn range: {freqs_in_range}.\
                \nFailure: Pixels {element_shape[1]*i} through {element_shape[1]*(i+1)} are not == 1"

    logger.info("DFT filter is working properly.")
