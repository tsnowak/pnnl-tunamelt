from doctest import UnexpectedException
import sys
from pathlib import Path
from typing import Union, List, Dict, Tuple
import xmltodict
from pprint import pprint

import cv2
import numpy as np

from turbx import REPO_PATH, log

VALID_LABEL_TYPES = ["cvat_video-1.1"]


class Dataset:
    """
    The DataLoader abstracts away video input loading and preprocessing
    - lazy: directory with files to glob
    - exact file

    It does the same for video labels.
    Labels for our data were annotated using a local CVAT server instance
    and labels were exported using the cvat-video-1.1 label format
    The label code translate this format into a usable list/dict format

    DataLoaderIterator to get next video from list of videos

    Videos are [N, H, W, C] uint8 np.ndarrays
    """

    def __init__(
        self,
        videos: Union[str, Path, list],
        labels: Union[str, Path, list, None] = None,
        video_format: str = "HSV",
    ):

        log.info("Initializing Dataset...")
        # Find and load video data
        # ensure path is a list of pathlib.Paths
        self.videos = find_files(videos)
        log.info(f"{len(self.videos)} video files found")
        # parse given labels files
        # is None if labels is None
        self.labels = parse_labels(labels)

        # create a data structure which matches
        # videos and labels
        self.aligned_data = self._align_videos_labels()

        # video format
        self.video_format = video_format

    def _align_videos_labels(self):
        """
        Returns:
            {<split>: [(<video path>, <video info and tracks>)]}
        """

        if self.labels is None:
            # create default split name of "None"
            return {"None": list(zip(self.videos, [None] * len(self.videos)))}
        dataset = {}
        # TODO: inefficiently implemented...
        for split, data in self.labels.items():
            for label in data:
                # find label file path in labeled videos
                video = None
                for v in self.videos:
                    # labeler didn't always use .mp4 in label["filename"]
                    if label["filename"] in v.name:
                        video = v
                        break
                if video is not None:
                    try:
                        dataset[split].append((video, label))
                    except KeyError:
                        dataset[split] = [(video, label)]
        return dataset


class DataLoader:
    """
    Fetch and return (video, label) from the dataset
    """

    def __init__(self, dataset, split: str = "train"):
        """Create loader from data and only return split if given"""
        self.dataset = dataset
        # with no labels is no split info
        if self.dataset.labels is None:
            self.split = "None"
        else:
            self.split = split
        self.data_split = self.dataset.aligned_data[self.split]
        self._idx = 0

    def __iter__(self):
        return self

    def __getitem__(self, idx: int):
        v_path, label = self.data_split[idx]
        video = to_numpy(v_path)
        return video, label

    def __setitem__(self):
        raise NotImplementedError

    def __delitem__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data_split)

    def __next__(self):
        if self._idx < len(self):
            video, label = self.__getitem__(self._idx)
            self._idx += 1
            return video, label
        else:
            raise StopIteration

    def reset(self):
        self._idx = 0

    def get_vid_id(self, vid_id: int) -> Tuple[np.ndarray, Dict]:
        for v_path, label in self.data_split:
            if label["video_id"] == vid_id:
                video = to_numpy(v_path)
                return video, label
        raise ValueError(f"{vid_id} does not exist.")


def find_files(path, file_type=".mp4"):
    """
    Accepts lazy file path inputs and discovers all relevant files

    Returns:
        [<file paths>]
    """
    o_path = []

    if not isinstance(path, list):
        path = [path]

    for p in path:
        if isinstance(p, str):
            p = Path(p)
        if isinstance(p, Path):
            if p.is_dir():
                p = list(p.glob(f"**/*{file_type}"))
                o_path = o_path + p
            elif (p.is_file()) and (p.suffix == file_type):
                o_path.append(p)
        else:
            raise TypeError(f"{p} is path contains neither path nor string.")

    return o_path


def parse_labels(path, label_type: str = "cvat_video-1.1", file_type: str = ".xml"):
    """
    Accepts a path to a label file or directory of label files,
    parses each file into dicts/lists, and combines separate label,
    files, into a single large label dataset

    Returns:
        {<splits>: [<video info and tracks]}
    """
    assert (
        label_type in VALID_LABEL_TYPES
    ), f"Given label type is not one of the following: {VALID_LABEL_TYPES}"

    def parse_labels_file(f, label_type):
        # open the label file and parse according to label type function
        with open(f, "rb") as s:
            # replace - and . with function-worthy alternatives: see parse_cvat_video_11
            cmd = f"parse_{label_type}".replace("-", "_").replace(".", "")
            out = eval(f"{cmd}")(s)
        return out

    # return None if no path given
    if path is None:
        return None

    path_list = find_files(path, file_type=file_type)

    # Structure is roughly Files[Splits{Videos[Video{Tracks[]}]}]
    # See parse_cvat_video_11 for specifics
    tmp_labels = []
    for l in path_list:
        tmp_labels.append(parse_labels_file(l, label_type="cvat_video-1.1"))

    # merge labels across files into single split-separated dictionary
    labels = {}
    for l in tmp_labels:
        for k, v in l.items():
            try:
                labels[k] = labels[k] + v
            except KeyError:
                labels[k] = v

    return labels


def parse_cvat_video_11(xml_f):
    """
    Returns {<split>: [<video info and tracks>]}
    """
    xml_dict = xmltodict.parse(xml_f, encoding="utf-8", xml_attribs=True)
    # dict of dicts containing video info
    try:
        video = xml_dict["annotations"]["meta"]["task"]
    except KeyError:
        videos = xml_dict["annotations"]["meta"]["project"]["tasks"]["task"]
        videos = {v["id"]: v for v in videos}
        raise TypeError(
            "Project-wide annotations.xml file is being used. Label frames are incorrect in this file. Download annotations per task!"
        )

    # dict of dicts containing each label/annotation info
    try:
        labels = xml_dict["annotations"]["track"]
        # handle single track videos
        labels = [labels] if not isinstance(labels, list) else labels
    except KeyError:
        # no tracks case
        labels = []

    video_id = int(video["id"])
    # create data splits and store associated video ids
    split = video["subset"].lower()

    output = {
        "video_id": video_id,
        "filename": video["name"],
        "video_length": int(video["size"]),
        "video_shape": {
            "height": int(video["original_size"]["height"]),
            "width": int(video["original_size"]["width"]),
        },
    }

    # loop through labels and videos once each
    # and compose dictionary datastructures on the fly
    tracks = []
    for l in labels:
        # clean and reorder individual frame labels
        frames = []
        for frame in l["box"]:
            try:
                d = {
                    "frame": int(frame["@frame"]),
                    "box": (
                        (int(float(frame["@xtl"])), int(float(frame["@ytl"]))),
                        (int(float(frame["@xbr"])), int(float(frame["@ybr"]))),
                    ),
                    "occluded": int(frame["@occluded"]),
                    "outside": int(frame["@outside"]),
                    "keyframe": int(frame["@keyframe"]),
                }
                frames.append(d)
            except TypeError:
                # Not sure why a type error is being thrown on frame: int(frame['@frame']) on the last iter
                pass
        # create per object tracks
        track_id = int(l["@id"])
        track = {"track_id": track_id, "label": l["@label"], "frames": frames}
        tracks.append(track)

    output.update({"tracks": tracks})
    dataset = {split: [output]}

    return dataset


def numpy_to_cv2(videos: Union[List, np.ndarray], input_format, output_format):

    # support multi-video input - cast singletons to list
    if not isinstance(videos, list):
        videos = [videos]

    # loop through videos
    for idx, video in enumerate(videos):
        if len(video.shape) == 4:
            out_video = np.ndarray(video.shape, dtype=video.dtype)
        elif len(video.shape) == 3:
            out_video = np.ndarray(video.shape + (3,), dtype=video.dtype)
        else:
            raise ValueError("Video is neither of len(shape) 4 or 3.")
        # loop through frames
        for i in range(video.shape[0]):
            frame = video[i, ...]
            color_cmd = eval(f"cv2.COLOR_{input_format}2{output_format}")
            out_video[i, ...] = cv2.cvtColor(frame, color_cmd)
        videos[idx] = out_video

    if len(videos) == 1:
        return videos[0]
    else:
        return videos


def to_numpy(path, video_format="HSV"):
    """Loads single video and returns as a numpy array"""
    cap = cv2.VideoCapture(str(path))
    # NOTE: should we auto convert to HSV?
    video = cap_to_nparray(cap, video_format=video_format)
    return video


def cap_to_nparray(cap, video_format="BGR"):
    # catch error
    if cap.isOpened() is False:
        print("Error opening video")
        sys.exit(1)

    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype("uint8"))

    fc = 0
    ret = True
    buf = []
    while fc < frameCount and ret:
        ret, frame = cap.read()
        if video_format == "BGR":
            buf.append(frame)
        else:
            try:
                color_cmd = eval(f"cv2.COLOR_BGR2{video_format}")
            except Exception as e:
                print("Color format does not exist")
                raise e
            buf.append(cv2.cvtColor(frame, color_cmd))
        fc += 1
    cap.release()
    return np.asarray(buf)


## UNTESTED ##


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
            f"Invalid data_dir or file_name provided.\ndata_dir: {data_dir}\nfile_name: {file_name}"
        )

    log.info(f"Found file at: {data_dir}/{file_name}")
    vid_path = get_file_path(file_name, [data_dir], absolute=True)

    # define place to save outputs
    image_path = Path(REPO_PATH + rel_out_dir)
    Path(image_path).mkdir(parents=True, exist_ok=True)

    return vid_path, image_path


def get_file_path(
    f: str, data_paths: list, return_first: bool = True, absolute: bool = False
) -> str:
    def d_func(x):
        if absolute:
            return Path(x).absolute()
        else:
            return Path(x)

    data_paths = [d_func(d) if isinstance(d, str) else exit(1) for d in data_paths]
    f_paths = []
    for d in data_paths:
        f_paths += d.glob("**/*")

    out = []
    for f_path in f_paths:
        if f_path.name == f:
            if return_first:
                return f_path
            else:
                out.append(f_path)

    return out
