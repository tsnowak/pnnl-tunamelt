
import numpy as np
import cv2

from fish.data import get_file_path
from fish.filter import crop_polygon



if __name__ == "__main__":

    # load file
    data_paths = [
        '/Users/nowa201/Data/fish_detector',
        '/data/nowa201/Projects/fish_detection/mp4'
    ]
    name = "2010-09-08_081500_HF_S021"
    vid_path = get_file_path(f"{name}.mp4", data_paths, absolute=True)

    # create cv video
    cap = cv2.VideoCapture(str(vid_path))
    cv2.namedWindow("Raw", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Cropped", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Filtered", cv2.WINDOW_NORMAL)
    cv2.namedWindow("H", cv2.WINDOW_NORMAL)
    cv2.namedWindow("S", cv2.WINDOW_NORMAL)
    cv2.namedWindow("V", cv2.WINDOW_NORMAL)
    fps = 10

    # define polygon region
    # bl, tl, tr, br
    polygon = np.array([ (502, 1753), (69, 21), (967, 21), (541, 1753) ], np.int32)
    v_thresh = 150
    backSub = cv2.createBackgroundSubtractorMOG2()

    # catch error
    if (cap.isOpened() == False):
        print ("Error opening video")

    # get frame by frame
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # process frame

            # crop actual image polygon
            cropped = crop_polygon(frame, polygon) 

            # maybe convert to other space
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            hsv = np.where(np.repeat((hsv[:, :, 2] > v_thresh)[:, :, np.newaxis], 3, axis=2),
                           hsv, 0)

            filtered = backSub.apply(cropped)
            #filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('Raw', frame)
            cv2.imshow('Cropped', cropped)
            cv2.imshow('Filtered', filtered)
            cv2.imshow('H', hsv[..., 0])
            cv2.imshow('S', hsv[..., 1])
            cv2.imshow('V', hsv[..., 2])
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()
    cv2.destroyAllWindows()

            
            
    print("done.")