import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import xmltodict

from tunamelt import REPO_PATH, log

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
                p = list(p.glob(f"**/[!.]*{file_type}"))
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

    def parse_labels_file(file_name, label_type):
        # open the label file and parse according to label type function
        with open(file_name, "rb") as _file_name:
            out = parse_cvat_video_11(_file_name)
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
        "start_frame": int(video["start_frame"]),
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
            try:
                color_cmd = getattr(cv2, f"COLOR_{input_format}2{output_format}")
                out_video[i, ...] = cv2.cvtColor(frame, color_cmd)
            except Exception as e:
                log.error(
                    e,
                    f"COLOR_{input_format}2{output_format} is not a valid conversion.",
                )
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
        log.info("Error opening video")
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
            if video_format == "RGB":
                color_cmd = cv2.COLOR_BGR2RGB
            elif video_format == "HSV":
                color_cmd = cv2.COLOR_BGR2HSV
            elif video_format == "GRAY":
                color_cmd = cv2.COLOR_BGR2GRAY
            else:
                raise TypeError("Color format does not exist")
            buf.append(cv2.cvtColor(frame, color_cmd))
        fc += 1
    cap.release()
    return np.asarray(buf)


# - frame - start_frame not in bounds for last frame
def label_to_per_frame_list(label: Dict, key="tracks"):
    """
    Returns a list of bounding boxes per frame
    """
    # support split videos while back support full videos
    try:
        start_frame = label["start_frame"]
    except KeyError:
        start_frame = 0

    # boxes = e.g. [0, 300], e.g. [300, 600] -> boxes[600] out of range
    boxes = [[] for _ in range(start_frame, start_frame + label["video_length"])]
    for track in label[key]:
        for frame in track["frames"]:
            boxes[frame["frame"] - start_frame].append(frame["box"])

    return boxes


def label_to_per_frame_targets(label: Dict) -> List:
    """
    Returns a list of target_ids per frame
    """
    # support split videos while back support full videos
    try:
        start_frame = label["start_frame"]
    except KeyError:
        start_frame = 0

    targets = [[] for _ in range(start_frame, start_frame + label["video_length"])]
    for track in label["tracks"]:
        for frame in track["frames"]:
            targets[frame["frame"] - start_frame].append(track["track_id"])

    return targets


def xywh_to_xyxy(box):
    return ((box[0], box[1]), (box[0] + box[2], box[1] + box[3]))


def xyxy_to_xywh(box):
    return (box[0][0], box[0][1], box[0][0] - box[1][0], box[0][1] - box[1][1])


def load_video(filename, data_path=f"{REPO_PATH}/data/PNNL-TUNAMELT/mp4"):
    video_path = list(Path(data_path).glob(f"**/{filename}"))[0]
    video = to_numpy(video_path, video_format="BGR")
    return video
