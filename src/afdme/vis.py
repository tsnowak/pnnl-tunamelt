import json
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import imageio as iio
import numpy as np
import psutil
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

from afdme import log
from afdme.data import label_to_per_frame_list, xywh_to_xyxy

log.setLevel(logging.INFO)


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
        log.info("Press q to close video windows.")
        display_videos(videos, label, pred, fps, loop=loop)

    # cleanly exit after videos are saved
    if save:
        log.info("Waiting for videos to write...")
        pool.join()
        log.info("Done writing videos")


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


def draw_pred(image, frame_pred, color=(0, 255, 0)):
    """
    Draws prediction bounding box on the images
    """
    # grayscale - box should be white
    if len(image.shape) != 3:
        color = (255, 255, 255)
    for box in frame_pred:
        box = xywh_to_xyxy(box)
        image = cv2.rectangle(image, box[0], box[1], color, thickness=2)
    return image


def draw_label(image, frame_label, color=(0, 0, 255)):
    """
    Draws label bounding box on the images
    """
    # grayscale - box should be white
    if len(image.shape) != 3:
        color = (255, 255, 255)
    for box in frame_label:
        image = cv2.rectangle(image, box[0], box[1], color, thickness=2)
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
