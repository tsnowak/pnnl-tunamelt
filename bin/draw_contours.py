
import cv2

from fish.data import get_file_path


if __name__ == "__main__":

    data_paths = [
        './media',
        '/Users/nowa201/Data/fish_detector'
    ]

    vid_path = get_file_path("school_avoiding_upstream.mp4", data_paths, absolute=True)

    # assume frame 1 contains object
    cap = cv2.VideoCapture(str(vid_path))
    _, frame = cap.read()
    if frame is not None:
        print("agh")
            
    print("done.")