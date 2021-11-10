
import numpy as np
import cv2

from fish import logger
from fish.dft_filter import fourier_filter, mean_threshold, max_threshold
from fish.utils import generate_sinusoid_tile


def test_filter_frequency():
    '''
        Verify that specified frequencies get removed from video
        using a video of tiled flashing black and white squares
        which oscillate at differing frequencies
    '''

    freqs = [9, 3, 1, 1/3, 1/9]
    element_shape = (10, 15)
    n_frames = 1000

    waveform, fps, length = generate_sinusoid_tile(freqs=freqs,
                                                   element_shape=element_shape,
                                                   n_frames=n_frames)

    # convert to format of filter
    video = np.transpose(waveform, (2, 0, 1))
    video = np.expand_dims(video, axis=-1)

    logger.debug(f"\nInput video shape: {video.shape}")
    mask = fourier_filter(video=video, fps=fps, freq_range=(.5, 9))

    logger.debug(
        f"\nVideo Shape: {waveform.shape}\nMask Shape: {mask.shape}")

    cv2.namedWindow("Sine Video", cv2.WINDOW_AUTOSIZE)

    # show the video until escape is pressed
    n_frames = waveform.shape[2]
    cntr = 0
    while True:
        combined_frame = np.concatenate(
            [waveform[:, :, cntr], mask.squeeze()], axis=0)
        cv2.imshow("Sine Video", combined_frame)

        cntr += 1
        if cntr == n_frames:
            cntr = 0

        k = cv2.waitKey(int(1000*(1/fps)))
        if k == 27:
            cv2.destroyAllWindows()
            break

    return None
