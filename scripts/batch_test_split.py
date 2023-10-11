import xmltodict
from collections import OrderedDict
import imageio.v3 as iio
from pathlib import Path
from copy import deepcopy

from afdme import REPO_PATH, log
from afdme.data import find_files

# max number of frames per video
MAX_LEN = 300
FPS = 10.0

# create paths
label_path = f"{REPO_PATH}/data/AFD-ME/labels/cvat-video-1.1/test"
video_path = f"{REPO_PATH}/data/AFD-ME/mp4/test"
label_out_path = Path(f"{REPO_PATH}/data/AFD-ME/labels/cvat-video-1.1/batched_test")
video_out_path = Path(f"{REPO_PATH}/data/AFD-ME/mp4/batched_test")
label_out_path.mkdir(exist_ok=True)
video_out_path.mkdir(exist_ok=True)

# collect list of all files in paths
video_files = find_files(video_path)
label_files = find_files(label_path, file_type=".xml")

log.debug(f"First 3 video files: {video_files[:3]}")
log.debug(f"First 3 label files: {label_files[:3]}")


# loop through label files, match with video, batch into subfiles
def split_tracks(track_list, start_frame, end_frame):
    if not isinstance(track_list, list):
        track_list = [track_list]

    otrack_list = []
    # otrack_dict = {'@id': None, '@label': None, '@source': None, 'box': []}
    for track in track_list:
        otrack_dict = OrderedDict(
            {
                "@id": track["@id"],
                "@label": track["@label"],
                "@source": track["@source"],
                "box": [],
            }
        )
        for box in track["box"]:
            box_frame_idx = int(box["@frame"])
            if start_frame <= box_frame_idx <= end_frame:
                otrack_dict["box"].append(box)
        if len(otrack_dict["box"]) > 0:
            otrack_list.append(otrack_dict.copy())

    return otrack_list


# loop through every label file
for label_file in label_files:
    # vars
    video_file = Path()
    sub_video_counter = 0

    with open(str(label_file), "rb") as f:
        label_xml_dict = xmltodict.parse(f, encoding="utf-8", xml_attribs=True)
    video_info_dict = label_xml_dict["annotations"]["meta"]["task"]
    video_size = int(video_info_dict["size"])
    log.info(f"Splitting video: {video_info_dict['id']}")

    # verify video with name exists
    for v in video_files:
        if video_info_dict["name"] in v.name:
            video_file = v
            break
    # go to next iteration if video didn't exist
    if not video_file.exists():
        continue

    # loop through greater video splitting into sub_videos
    video = iio.imread(str(video_file))
    while (sub_video_counter * MAX_LEN) < video_size:
        log.info(f"Creating sub-video: {video_info_dict['id']}/{sub_video_counter:04}")

        # NOTE: USE INCLUSIVE INDEXING OR ADD 1/USE SIZE
        start_frame = sub_video_counter * MAX_LEN  # 0
        end_frame = ((sub_video_counter + 1) * MAX_LEN) - 1  # 299
        end_frame = end_frame if end_frame < video_size else (video_size - 1)
        size = (end_frame - start_frame) + 1
        sub_video_label = deepcopy(label_xml_dict)

        # E.g: name-0001.mp4
        sub_video_file = Path(
            f"{video_out_path}/{video_file.stem}-{sub_video_counter:04}{video_file.suffix}"
        )
        # write start_frame to end from video_file
        sub_video = video[start_frame : (start_frame + size), ...]
        assert len(sub_video) == size, "Sub video does not match label 'video_size'."
        writer = iio.imwrite(
            sub_video_file, sub_video, fps=FPS
        )  # so much faster than writing each frame

        # modify label file for sub-video
        # split tracks based on those in sub-video (split track if necessary - can keep track id?)
        id = f"{video_info_dict['id']}{sub_video_counter:04}"
        sub_video_label["annotations"]["meta"]["task"]["id"] = id
        sub_video_label["annotations"]["meta"]["task"][
            "name"
        ] = f"{sub_video_file.name}"
        sub_video_label["annotations"]["meta"]["task"]["size"] = size
        sub_video_label["annotations"]["meta"]["task"]["start_frame"] = start_frame
        sub_video_label["annotations"]["meta"]["task"]["stop_frame"] = end_frame
        sub_video_label["annotations"]["meta"]["task"]["segments"] = OrderedDict(
            {
                "segment": OrderedDict(
                    {
                        "id": video_info_dict["segments"]["segment"]["id"],
                        "start": start_frame,
                        "stop": end_frame,
                        "url": video_info_dict["segments"]["segment"]["url"],
                    }
                )
            }
        )
        try:
            track_list = split_tracks(
                deepcopy(label_xml_dict["annotations"]["track"]), start_frame, end_frame
            )
            if len(track_list) > 0:
                sub_video_label["annotations"]["track"] = track_list
            else:
                del sub_video_label["annotations"]["track"]
        except KeyError:
            pass

        with open(f"{str(label_out_path)}/{id}.xml", "w") as f:
            xmltodict.unparse(sub_video_label, f)

        # increment for next iter
        sub_video_counter += 1
