import logging
import json
from pathlib import Path
from typing import List, Tuple, Union, Dict
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
from turbx.metrics import boxes_to_binary, tfpnr

log.setLevel(logging.INFO)


def plot_label_and_pred(label: List, pred: List):
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

    plt.show(block=False)
    # TODO: save to json results file
    return binary_label, binary_pred, tfpnr_dict


def label_to_per_frame_list(label: Dict):
    """
    Returns a list of bounding boxes per frame
    """
    boxes = [[] for _ in range(label["video_length"])]
    for track in label["tracks"]:
        for frame in track["frames"]:
            boxes[frame["frame"]].append(frame["box"])

    return boxes


def xywh_to_xyxy(box):
    return ((box[0], box[1]), (box[0] + box[2], box[1] + box[3]))


def draw_pred(image, frame_pred, color=(0, 255, 0)):
    """
    Draws prediction bounding box on the images
    """
    for box in frame_pred:
        box = xywh_to_xyxy(box)
        image = cv2.rectangle(image, box[0], box[1], color, 4)
    return image


def draw_label(image, frame_label, color=(0, 0, 255)):
    """
    Draws label bounding box on the images
    """
    for box in frame_label:
        image = cv2.rectangle(image, box[0], box[1], color, 4)
    return image


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


def view(
    videos: Dict,
    label: Union[Dict, None],
    pred: Union[List, None],
    fps: int,
    loop: bool = True,
    save: bool = True,
    out_path: Union[Path, None] = None,
    video_type: str = ".mp4",
):
    """
    Results display. Create pane per filtered video and for plot of results
    """

    # bounds length variable
    assert len(videos) != 0, "No videos given. Exiting."

    label_dict = label
    if label is not None:
        label = label_to_per_frame_list(label)

    # launch separate processes to save videos
    # TODO: add annotations to saved videos
    if save:
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

    # create opencv windows
    for name, v in videos.items():
        length = v.shape[0]
        cv2.namedWindow(f"{name}")

    # plot binary label and predictions
    if (label is not None) and (pred is not None):
        binary_label, binary_pred, tfpnr_dict = plot_label_and_pred(label, pred)
        # save results to json
        with open(f"{str(out_path)}/{label_dict['filename']}.results.json", "w") as f:
            json_content = json.dumps(
                {
                    "label": label_dict,
                    "prediction": pred,
                    "parameters": {"TODO": None},
                    "results": tfpnr_dict,
                },
                indent=4,
            )
            f.write(json_content)

    interval = int(1000 / fps)
    frame = 0
    # loop over videos
    while True:

        # update frame per pane
        for name, v in videos.items():
            # TODO: draw per frame labels and predictions
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

    # cleanly exit after videos are saved
    if save:
        log.info("Waiting for videos to write...")
        pool.join()
        log.info("Done writing videos")


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


def show_gif(f_path, img_width=100):
    return HTML(
        f'<img src="{f_path}" alt="Acoustic Camera GIF" style="width:{img_width}%"/>'
    )


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
