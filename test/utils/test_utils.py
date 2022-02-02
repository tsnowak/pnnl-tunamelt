
import cv2
import numpy as np
from fish.utils import generate_sinusoid, generate_sinusoid_tile


def test_generate_sinusoid():

    video_length = 1000

    freq_1 = 1/10  # Hz
    freq_2 = 1  # Hz

    length = max([1/freq_1, 1/freq_2])  # second video
    fps = video_length/length  # samples per second; > 2*freq
    shape = (10, 15)  # video pixel resolution
    waveform_1 = generate_sinusoid(freq=freq_1,
                                   fps=fps,
                                   shape=shape,
                                   length=length
                                   )
    waveform_2 = generate_sinusoid(freq=freq_2,
                                   fps=fps,
                                   shape=shape,
                                   length=length
                                   )

    waveform = np.concatenate([waveform_1, waveform_2], axis=1)

    cv2.namedWindow("Sine Video", cv2.WINDOW_AUTOSIZE)

    # show the video until escape is pressed
    n_frames = waveform.shape[2]
    cntr = 0
    while True:
        cv2.imshow("Sine Video", waveform[:, :, cntr])

        cntr += 1
        if cntr == n_frames:
            cntr = 0

        k = cv2.waitKey(int(1000*(1/fps)))
        if k == 27:
            cv2.destroyAllWindows()
            break

    return None


def test_generate_sinusoid_tile():

    freqs = [9, 3, 1, 1/3, 1/9]
    element_shape = (10, 15)
    n_frames = 1000

    waveform, fps, length = generate_sinusoid_tile(freqs=freqs,
                                                   element_shape=element_shape,
                                                   n_frames=n_frames
                                                   )

    cv2.namedWindow("Sine Video", cv2.WINDOW_AUTOSIZE)

    # show the video until escape is pressed
    n_frames = waveform.shape[2]
    cntr = 0
    while True:
        cv2.imshow("Sine Video", waveform[:, :, cntr])

        cntr += 1
        if cntr == n_frames:
            cntr = 0

        k = cv2.waitKey(int(1000*(1/fps)))
        if k == 27:
            cv2.destroyAllWindows()
            break

    return None
