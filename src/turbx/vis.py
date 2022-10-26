import logging
import json
from pathlib import Path
from typing import List, Tuple, Union, Dict, Optional
import sys
from multiprocessing import Process, Pool
import psutil
import base64
import asyncio
import cv2
import imageio as iio
import numpy as np
from IPython.core.display import HTML
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq, fftn, fftshift

from turbx import log
from turbx.metrics import (
    calc_box_area,
    boxes_to_binary,
    target_detection_rate,
    tfpnr,
)

log.setLevel(logging.INFO)


# TODO: in progress
def viz_metrics_results(
    label: Union[Dict, None],
    pred: Union[List, None],
    params: Dict = {},
    show: bool = True,
    save: bool = True,
    out_path: Union[Path, None] = None,
):
    """
    Results display. Create and save plots of results
    """
    # plot binary label and predictions
    if (label is None) or (pred is None):
        log.info("Skipping metrics because either label or predictions is not given")
        return None

    # calculates per frame average target size, video min, max, avg
    per_frame_avg_size, min_size, max_size, avg_size = calc_size(label)
    # calculate # frames removed using predictions
    _, n_pos_dets, n_neg_dets = calc_frames_removed(pred)
    # calculates and show/saves true/false positive/negative rates
    binary_label, binary_pred, tfpnr_dict = calc_tfpnr(label, pred)
    # calculates # unique targets, # detections of targets, and per video target detection rate
    unique_targs, det_targs, tdr = calc_tdr(label, pred)
    perc_frames_removed = n_neg_dets / len(binary_pred)

    # TODO: refactor saving of results
    # save results to json
    results = {
        "per_frame_results": tfpnr_dict,
        "per_target_results": (unique_targs, det_targs, tdr),
    }
    with open(f"{str(out_path)}/{label['filename']}.results.json", "w") as f:
        json_content = json.dumps(
            {
                "label": label,
                "prediction": pred,
                "parameters": params,
                "results": results,
            },
            indent=4,
        )
        outputs.update({"results": results})
        f.write(json_content)


def write_results(
    params: Dict, label: Dict, pred: List, out_path: Optional[Union[str, Path]] = ""
):
    with open(f"{str(out_path)}/{label['filename']}.results.json", "w") as f:
        json_content = json.dumps(
            {
                "label": label,
                "prediction": pred,
                "parameters": params,
            },
            indent=4,
        )
        f.write(json_content)


def viz_video_results(
    videos: Dict,
    label: Union[Dict, None],
    pred: Union[List, None],
    fps: int,
    params: Dict = {},
    loop: bool = True,
    show: bool = True,
    save: bool = True,
    out_path: Union[Path, None] = None,
    video_type: str = ".mp4",
):
    """
    Video results display. Create pane per filtered video with labels
    """

    # bounds length variable
    assert len(videos) != 0, "No videos given. Exiting."

    # process labels into list of bboxes (but preservice dictionary)
    if label is not None:
        label = label_to_per_frame_list(label)

    # launch separate processes to save videos
    if save:
        pool = save_videos(
            videos, label, pred, fps, out_path=out_path, video_type=video_type
        )

    # display videos
    if show:
        display_videos(videos, label, pred, fps, loop=loop)

    # cleanly exit after videos are saved
    if save:
        log.info("Waiting for videos to write...")
        pool.join()
        log.info("Done writing videos")


def calc_size(label: Dict):
    """
    Calculate per frame average target size, video min size, video max size, and video average size
    """
    min_size, max_size, avg_size = None, None, None
    per_target_size = []
    per_frame_avg_size = [[] for _ in range(label["video_length"])]
    for track in label["tracks"]:
        for frame in track["frames"]:
            box_area = calc_box_area(frame["box"])
            per_frame_avg_size[frame["frame"]].append(box_area)
            per_target_size.append(box_area)
    per_frame_avg_size = [sum(x) / len(x) for x in per_frame_avg_size]
    min_size = min(per_target_size)
    max_size = max(per_target_size)
    avg_size = sum(per_target_size) / len(per_target_size)
    return per_frame_avg_size, min_size, max_size, avg_size


def calc_frames_removed(pred: List):
    binary_preds = boxes_to_binary(pred)
    pos_dets = [x for x in binary_preds if x == 1]
    neg_dets = [x for x in binary_preds if x == 0]
    return binary_preds, len(pos_dets), len(neg_dets)


def save_videos(videos, label, pred, fps, out_path=Path(), video_type=".mp4"):
    procs = []
    procs = len(videos) if len(videos) < psutil.cpu_count() else psutil.cpu_count()
    args = [
        (v, name, fps, label, pred, len(v), out_path, video_type)
        for name, v in videos.items()
    ]
    # single process
    # [write_video(*arg) for arg in args]
    # multi-process
    pool = Pool(processes=procs)
    pool.starmap_async(func=write_video, iterable=iter(args))
    pool.close()
    return pool


def write_video(
    video: np.ndarray,
    name: str,
    fps: int,
    label: Union[List, None] = None,
    pred: Union[List, None] = None,
    video_length: Union[int, None] = None,
    out_path: Union[Path, None] = None,
    video_type: str = ".mp4",
):
    """
    video: numpy array - [N, H, W, C]
    name: string - name of video
    fps: int - frames per second of video
    video_length: int, None - number of frames in the video
    out_path: Path, None - path at which to write the video
    """

    if video_length is None:
        video_length = video.shape[0]
    if out_path is None:
        out_path = Path().absolute()

    log.info(f"Started writing {name}{video_type}")
    writer = iio.get_writer(
        str(out_path) + "/" + f"{name}{video_type}", mode="I", fps=fps
    )

    for i in range(video_length):
        image = video[i, ...]
        if label is not None:
            image = draw_label(image, label[i])
        if pred is not None:
            image = draw_pred(image, pred[i])
        writer.append_data(image[:, :, ::-1].astype(np.uint8))

    writer.close()
    log.info(f"Finished writing {name}{video_type}")


def display_videos(videos, label, pred, fps, loop=True):
    """
    display videos and label/preds in either a loop or once
    """
    interval = int(1000 / fps)
    frame = 0
    loc = (0, 0)
    win_size = (900, 400)

    # create opencv windows
    for name, v in videos.items():
        length = v.shape[0]
        cv2.namedWindow(f"{name}")
        cv2.moveWindow(f"{name}", loc[1], loc[0])
        loc = (loc[0], loc[1] + win_size[1])

    # update windows per frame
    while True:
        # update frame per pane
        for name, v in videos.items():
            image = v[frame]
            if label is not None:
                image = draw_label(image, label[frame])
            if pred is not None:
                image = draw_pred(image, pred[frame])
            cv2.imshow(f"{name}", image)

        # exit on key press
        if cv2.waitKey(interval) & 0xFF == ord("q"):
            break

        # increment frame and loop
        frame += 1
        if loop and (frame == length):
            frame = 0

    # cleanly destroy windows
    plt.close("all")
    cv2.destroyAllWindows()


def calc_tfpnr(label: Dict, pred: List, show=False, save=False, out_path=Path()):

    # only need list of bbox as labels
    label = label_to_per_frame_list(label)

    # convert to per frame binary target presence labels
    binary_label = boxes_to_binary(label)
    binary_pred = boxes_to_binary(pred)
    # calculare TPR and FPR metrics
    tfpnr_dict = tfpnr(binary_label, binary_pred)

    # plot binary per frame results
    plt.figure("per_frame")
    plt.plot(binary_label)
    plt.plot(binary_pred)

    plt.figure("metrics")
    keys = ["tpr", "tnr", "fpr", "fnr"]
    data = [tfpnr_dict[k] for k in keys]
    plt.bar(keys, data)
    plt.ylim(bottom=0.0, top=1.0)

    if show:
        plt.show(block=False)
    if save:
        plt.savefig(out_path)

    return binary_label, binary_pred, tfpnr_dict


def calc_tdr(label: Dict, pred: List):
    # convert to per frame binary target presence labels
    targets = label_to_per_frame_targets(label)
    binary_pred = boxes_to_binary(pred)
    # calculare TPR and FPR metrics
    return target_detection_rate(targets, binary_pred)


def label_to_per_frame_list(label: Dict):
    """
    Returns a list of bounding boxes per frame
    """
    boxes = [[] for _ in range(label["video_length"])]
    for track in label["tracks"]:
        for frame in track["frames"]:
            boxes[frame["frame"]].append(frame["box"])

    return boxes


def label_to_per_frame_targets(label: Dict) -> List:
    """
    Returns a list of target_ids per frame
    """
    targets = [[] for _ in range(label["video_length"])]
    for track in label["tracks"]:
        for frame in track["frames"]:
            targets[frame["frame"]].append(track["track_id"])

    return targets


def xywh_to_xyxy(box):
    return ((box[0], box[1]), (box[0] + box[2], box[1] + box[3]))


def xyxy_to_xywh(box):
    return (box[0][0], box[0][1], box[0][0] - box[1][0], box[0][1] - box[1][1])


def draw_pred(image, frame_pred, color=(0, 255, 0)):
    """
    Draws prediction bounding box on the images
    """
    # grayscale - box should be white
    if len(image.shape) != 3:
        color = (255, 255, 255)
    for box in frame_pred:
        box = xywh_to_xyxy(box)
        image = cv2.rectangle(image, box[0], box[1], color, 4)
    return image


def draw_label(image, frame_label, color=(0, 0, 255)):
    """
    Draws label bounding box on the images
    """
    # grayscale - box should be white
    if len(image.shape) != 3:
        color = (255, 255, 255)
    for box in frame_label:
        image = cv2.rectangle(image, box[0], box[1], color, 4)
    return image


def plot_time_domain_waveform(video, fps, pixel, freq_range=None):
    """
    Args:
        video: [N, W, H, C]
        fps: frames per second
        pixel: Tuple pixel location to plot
    """

    assert len(video.shape) == 4, "Video should be in format [N, W, H, C]"

    # get pixel-wise intensity over time
    time_domain = video[:, pixel[0], pixel[1], 0]
    N = len(time_domain)
    t = np.linspace(0, N * fps, N)

    # get fft of pixel over time
    fft_pixel = fft(time_domain, axis=0, workers=-1)
    fft_pixel = fftshift(fft_pixel, axes=0)  # 0 freq at center
    # For each component get the frequency center that it represents
    freq = fftfreq(N, d=1 / fps)
    freq = fftshift(freq)
    log.debug(f"fft_video {fft_pixel.shape}")

    # only operate on positive frequencies (greater than 0 plus fudge)
    freq_thresh = 0
    pos_range = np.argwhere(freq > freq_thresh).squeeze()
    fft_pixel = fft_pixel[pos_range, ...]
    log.debug(f"pos_range video: {fft_pixel.shape}")
    freq = freq[pos_range, ...]

    # get magnitude and phase of each frequency component
    # magnitude of that frequency component
    mag = np.absolute(fft_pixel)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title(f"Pixel {pixel} intensity over time")
    plt.plot(t, time_domain)

    plt.subplot(1, 2, 2)
    plt.title(f"DFT of pixel {pixel}")
    plt.plot(freq, mag)
    plt.plot(freq, np.ones(mag.shape) * np.mean(mag))

    if freq_range is not None:
        plt.vlines(freq_range, -100, np.max(mag), colors="red", linestyles="dashed")

    plt.show()

    return None


# DEPRICATED
class VideoStream(object):
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        self.max_idx = self.video.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        self.idx = self.video.get(cv2.CAP_PROP_POS_FRAMES)

    def __del__(self):
        self.video.release()

    def get_frame(self, labels=None):

        # loop video
        self.idx = self.video.get(cv2.CAP_PROP_POS_FRAMES)
        if self.idx == self.max_idx:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # return byte-encoded image
        success, image = self.video.read()
        if success:
            cv2.putText(
                img=image,
                text=f"{int(self.idx)}/{int(self.max_idx)}",
                org=(10, 50),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=1,
            )
            _, jpeg = cv2.imencode(".jpg", image)
            return jpeg.tobytes()
        else:
            raise IOError("Failed to retrieve video frame.")


# DEPRICATED
def show_gif(f_path, img_width=100):
    return HTML(
        f'<img src="{f_path}" alt="Acoustic Camera GIF" style="width:{img_width}%"/>'
    )
