
import sys
import numpy as np
import cv2
from pathlib import Path

from fish import REPO_PATH, logger


def prep_exp_data(data_dir, file_name, rel_out_dir):

    if Path(data_dir + "/" + file_name).is_file():
        file_name = file_name
        data_dir = data_dir
    elif Path(data_dir + "/" + file_name + ".mp4").is_file():
        file_name = file_name + ".mp4"
        data_dir = data_dir
    elif Path(file_name).is_file():
        data_dir = str(Path(file_name).parent)
        file_name = str(Path(file_name).name)
    else:
        raise ValueError(
            f"Invalid data_dir or file_name provided.\ndata_dir: {data_dir}\nfile_name: {file_name}")

    logger.info(f"Found file at: {data_dir}/{file_name}")
    vid_path = get_file_path(file_name, [data_dir], absolute=True)

    # define place to save outputs
    image_path = Path(
        REPO_PATH + rel_out_dir)
    Path(image_path).mkdir(exist_ok=True)

    return vid_path, image_path


def get_file_path(f: str,
                  data_paths: list,
                  return_first: bool = True,
                  absolute: bool = False) -> str:

    def d_func(x):
        if absolute:
            return Path(x).absolute()
        else:
            return Path(x)

    data_paths = [d_func(d) if isinstance(d, str) else exit(1)
                  for d in data_paths]
    f_paths = []
    for d in data_paths:
        f_paths += d.glob('**/*')

    out = []
    for f_path in f_paths:
        if f_path.name == f:
            if return_first:
                return f_path
            else:
                out.append(f_path)

    return out


def cap_to_nparray(cap, format='BGR'):
    # catch error
    if (cap.isOpened() == False):
        print("Error opening video")
        sys.exit(1)

    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True
    while (fc < frameCount and ret):
        ret, frame = cap.read()
        if (format == "BGR"):
            buf[fc] = frame
        else:
            try:
                color_cmd = eval(f"cv2.COLOR_BGR2{format}")
            except Exception as e:
                print("Color format does not exist")
                raise e
            buf[fc] = cv2.cvtColor(frame, color_cmd)
        fc += 1
    cap.release()
    return buf
