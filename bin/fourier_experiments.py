
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.fft import fftshift

from fish import logger
from fish.data import get_file_path, cap_to_nparray
from fish.filter import mean_filter, fourier_filter, crop_polygon
from fish.filter import mean_threshold, max_threshold
from fish.utils import plot_time_domain_waveform

if __name__ == "__main__":

    # load file
    data_paths = [
        '/Users/nowa201/Data/fish_detector',
        '/data/nowa201/Projects/fish_detection/mp4'
    ]

    #name = "2010-09-08_081500_HF_S021"
    name = "2010-09-08_074500_HF_S002_S001"
    #name = "school_avoiding_upstream"
    vid_path = get_file_path(f"{name}.mp4", data_paths, absolute=True)

    fps = 10
    freq_range = (1.25, 2.75)
    # define polygon region
    # bl, tl, tr, br
    polygon = np.array(
        [(502, 1753), (69, 21), (967, 21), (541, 1753)], np.int32)

    # create cv video
    print("Opening video...")
    cap = cv2.VideoCapture(str(vid_path))
    #o_video = cap_to_nparray(cap, format="BGR")
    video = cap_to_nparray(cap, format="HSV")
    s_channel = video[..., 2].squeeze()
    logger.debug(s_channel.shape)

    s_channel = np.expand_dims(s_channel, axis=-1)

    point_of_interest = polygon[1] + (300, 300)
    plot_time_domain_waveform(
        s_channel, fps, point_of_interest, freq_range=freq_range)

    # TODO: why does normalizing include static pieces of the video
    # get boolean mask of pixels with magnitude above mag_thresh in frequency range f_range
    print("Generating fourier mask...")
    fourier_pos = fourier_filter(s_channel, fps, freq_range=freq_range)
    fourier_zero = np.abs(fourier_pos - 1.)
    # plt.figure(0)
    #plt.plot(s_freq, s_video[:, 690, 375])
    # plt.show()
    #cv2.namedWindow("boolean", cv2.WINDOW_NORMAL)
    #cv2.imshow("boolean", bool_video)
    # cv2.waitKey(0)

    #pixels = [()]
    #view_frequency_swaths(s_video, pixels)

    print("Generating mean filter...")
    mean_video = mean_filter(s_channel, thresh=40)
    #cv2.namedWindow("mean", cv2.WINDOW_NORMAL)
    #cv2.imshow("mean", mean_video[39])
    # cv2.waitKey(0)

    print("Done!")
    cv2.namedWindow("frequency mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("final", cv2.WINDOW_NORMAL)
    i = 0
    while True:
        if i < mean_video.shape[0]:

            o_frame = cv2.cvtColor(video[i], cv2.COLOR_HSV2BGR)
            frame = mean_video[i]*fourier_zero

            # logger.debug(f"{frame.shape}")
            #frame = cv2.medianBlur(frame, 15)
            #_, frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)
            # contours, hierarchy = cv2.findContours(
            #    image=frame, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

            fourier_zero_rgb = cv2.cvtColor(
                fourier_zero.astype("float32"), cv2.COLOR_GRAY2BGR)
            frame = cv2.cvtColor(frame.astype("float32"), cv2.COLOR_GRAY2BGR)

            #big_contours = []
            # for cnt in contours:
            #    area = cv2.contourArea(cnt)
            #    if area > 70:
            #        big_contours.append(cnt)

            # cv2.drawContours(image=frame, contours=big_contours, contourIdx=-1,
            #                 color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

            point_of_interest = tuple(point_of_interest)
            cv2.circle(o_frame, point_of_interest, color=(
                255, 0, 0), radius=10, thickness=10)
            cv2.circle(frame, point_of_interest, color=(
                255, 0, 0), radius=10, thickness=10)
            cv2.circle(fourier_zero_rgb, point_of_interest,
                       color=(255, 0, 0), radius=10, thickness=10)

            cv2.putText(o_frame, f"Frame: {i}",
                        org=(100, 100),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Frame: {i}",
                        org=(100, 100),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA)
            cv2.imshow("frequency mask", fourier_zero_rgb)
            cv2.imshow("original", o_frame)
            cv2.imshow("final", frame)
            k = cv2.waitKey(int(1000/fps))
            if k == 27:
                break
            i += 1
        else:
            i = 0

    '''
    def post_mouse(event, x, y, flags, params):
        global mousex, mousey
        if event == cv2.EVENT_LBUTTONDOWN:
            mousex = x
            mousey = y

    cv2.namedWindow("Win", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Win', post_mouse)
    while(1):
        cv2.imshow("Win", f_video[9].real)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            print(mousex, mousey)
    '''
