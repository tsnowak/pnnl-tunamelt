from collections import OrderedDict
from typing import Dict, List, Optional

import cv2
import numpy as np

from tunamelt import log
from tunamelt.filter.base import OfflineFilter
from tunamelt.metrics import safe_division


class MeanFilter(OfflineFilter):
    def __init__(
        self,
        video: Optional[np.ndarray] = None,
        fps: Optional[int] = None,
        params: Optional[Dict] = {"std_devs": 2.5},
    ):
        """
        Removes static background by zeroing pixels in frames which .
        Args:
            - video: video to filter - Optional(np.array [N, H, W, C] or [N, H, W])
            - fps: fps of video - Optional(int)

        Memory Optimized
        """
        super().__init__(video, fps)
        if self.mask is None:
            log.debug(f"Did not generate {self.__class__} filter.")
        else:
            log.debug(f"Generated {self.__class__} filter mask.")
        self.fps = fps
        self.out_format = "GRAY"
        self.std_devs = params["std_devs"]

    def filter(
        self,
        video: np.ndarray,
        fps: Optional[int] = None,
    ):
        """
        Applies the mask to filter the video
        Args:
            - video: video to filter - np.array [N, H, W, C] or [N, H, W]
            - fps: fps of video - int
        """
        if fps is None:
            if self.fps is None:
                raise ValueError("fps not given.")
            fps = self.fps

        if len(video.shape) == 4:
            self.calculate(video[..., 2], fps)
            filtered_video = np.multiply(video, np.stack([self.mask] * 3, axis=3))
        elif len(video.shape) == 3:
            self.calculate(video, fps)
            filtered_video = np.multiply(video, self.mask)
        else:
            raise ValueError(
                "Input video is neither NxWxHxC nor NxWxH. Verify its structure."
            )

        filtered_video = filtered_video.astype(np.uint8)
        log.debug(f"Mean background filtered video of shape: {filtered_video.shape}")
        return filtered_video

    def calculate(
        self,
        value_channel: np.ndarray,
        fps: int,
    ):
        """
        Calculates the filter mask
        Args:
            - value_channel: value_channel to filter - np.array [N, H, W, C]
            - fps: fps of value_channel - int
        """
        self.fps = fps
        value_channel = value_channel.astype(np.float16)
        # calculate background
        # mean := frame of average values from throughout the value_channel
        mean = np.mean(value_channel, axis=0)
        var = np.var(value_channel.astype(np.float64), axis=0)
        # avg_value := single average pixel value for the value_channel
        # avg_mean = np.mean(mean)
        # avg_var = np.mean(var)

        # remove background
        # only compare h,s,V - Value values (N, W, H, 1)
        value_channel = np.subtract(value_channel, mean)
        # np.boolean mask, N, W, H, 1
        # difference values > per-pixel standard dev
        self.mask = value_channel > self.std_devs * np.sqrt(var)

        return self.mask


class GaussianBlurDenoiseFilter(OfflineFilter):
    def __init__(
        self,
        video: Optional[np.ndarray] = None,
        fps: Optional[int] = None,
        params: Optional[Dict] = {"blur_size": 11},
    ):
        """
        Removes noise from the video.
        Args:
            - video: video to filter - Optional(np.array [N, H, W, C] or [N, H, W])
            - fps: fps of video - Optional(int)
        Memory Optimized
        """
        super().__init__(video, fps)
        if self.mask is None:
            log.debug(f"Did not generate {self.__class__} filter.")
        else:
            log.debug(f"Generated {self.__class__} filter mask.")
        self.fps = fps
        self.out_format = "GRAY"
        self.blur_size = params["blur_size"]

    def filter(
        self,
        video: np.ndarray,
        fps: Optional[int] = None,
    ):
        """
        Applies the mask to filter the video
        Args:
            - video: video to filter - np.array [N, H, W, C] or [N, H, W]
            - fps: fps of video - int
        """
        if fps is None:
            if self.fps is None:
                raise ValueError("fps not given.")
            fps = self.fps

        video = video.copy()
        if len(video.shape) == 4:
            filtered_video = self.calculate(video[..., 2], fps)
        elif len(video.shape) == 3:
            filtered_video = self.calculate(video, fps)
        else:
            raise ValueError(
                "Input video is neither NxWxHxC nor NxWxH. Verify its structure."
            )
        log.debug(f"Noise filtered video of shape: {filtered_video.shape}")
        return filtered_video

    def calculate(
        self,
        value_channel: np.ndarray,
        fps: int,
    ):
        """
        Calculates the filter mask
        Args:
            - value_channel: value_channel to filter - np.array [N, H, W, C]
            - fps: fps of value_channel - int
        """
        self.fps = fps

        for i, frame in enumerate(value_channel):
            frame = cv2.medianBlur(frame, self.blur_size)
            value_channel[i, ...] = frame
        return value_channel


class IntensityFilter(OfflineFilter):
    def __init__(
        self,
        video: Optional[np.ndarray] = None,
        fps: Optional[int] = None,
        # n: Optional[int] = 500,
        params: Optional[Dict] = {"thresh": 100},
    ):
        """
        Removes pixels are below
        """
        # self.n = n
        super().__init__(video, fps)
        if self.mask is None:
            log.debug(f"Did not generate {self.__class__} filter.")
        else:
            log.debug(f"Generated {self.__class__} filter mask.")
        self.fps = fps
        self.out_format = "GRAY"
        self.thresh = params["thresh"]

    def filter(
        self,
        video: np.ndarray,
        fps: Optional[int] = None,
    ):
        if fps is None:
            if self.fps is None:
                raise ValueError("fps not given.")
            fps = self.fps
        self.mask = self.calculate(video, fps)
        # out = np.multiply(video, self.mask)

        out = np.multiply(video, self.mask, dtype=np.uint8)
        out = out.astype(np.uint8)
        log.debug(f"Intensity filtered video of shape: {out.shape}")
        return out

    def calculate(
        self,
        video: np.ndarray,
        fps: int,
    ):
        self.fps = fps

        # only keep pixels in the video that are within std of their max
        # apply mask in apply
        mask = video > (self.thresh)

        log.debug(f"Generated intensity filter mask of shape: {mask.shape}")

        return mask


class ContourFilter:
    def __init__(
        self,
        video: Optional[np.ndarray] = None,
        params: Optional[Dict] = {"min_area": 36, "max_area": 14355},
    ):
        """
        Detect contours of a certain size
        """
        self.min_area = params["min_area"]
        self.max_area = params["max_area"]

    def filter(
        self,
        video: np.ndarray,
    ):
        return self.calculate(video)

    def calculate(
        self,
        video: np.ndarray,
    ):
        boxes_per_frame = []
        for frame in range(len(video)):
            boxes = []
            # find contours
            if len(video.shape) == 4:
                thresh = cv2.cvtColor(video[frame, ...], cv2.COLOR_HSV2BGR)
                thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
            elif len(video.shape) == 3:
                thresh = video[frame, ...]
            contours, heirachy = cv2.findContours(
                image=thresh,
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )

            for cont in contours:
                rect = cv2.boundingRect(cont)
                area = rect[2] * rect[3]  # h*w
                if area >= self.min_area and area <= self.max_area:
                    boxes.append(rect)

            boxes_per_frame.append(boxes)

        return boxes_per_frame


class TrackletAssociation:
    def __init__(
        self,
        preds: Optional[List[List]] = None,
        params: Optional[Dict] = {"window": 4, "thresh": 300.0},
    ):
        """
        preds: List[List] - per-frame bounding boxes
        #http://vision.cse.psu.edu/courses/Tracking/vlpr12/lzhang_cvpr08global.pdf

        No need for something so powerful - simply want to discard spurious bounding boxes
        Maintaining offline assumption, could perform forward/backward or iterative refinement of boxes
        """
        self.preds = preds
        self.window = params["window"]
        self.thresh = params["thresh"]

    def filter(self, preds: List[List], window_length: int = 4):
        self.preds = preds
        self.window_length = window_length
        return self.calculate(preds)

    def _xyxy_centroid(self, box: List[List]):
        x_avg = int((box[0] + box[2]) / 2.0)
        y_avg = int((box[1] + box[3]) / 2.0)
        return np.array([x_avg, y_avg], dtype=np.uint16)

    def _distance_cost(self, box1, box2):
        box1_cent = self._xyxy_centroid(box1).astype(np.float32)
        box2_cent = self._xyxy_centroid(box2).astype(np.float32)
        # log.info(f"centroid: {box1_cent}, {box2_cent}")
        # https://reference.wolfram.com/language/ref/NormalizedSquaredEuclideanDistance.html
        # scaled between 0 and 1
        num = np.var(box1_cent - box2_cent)
        den = 2 * (np.var(box1_cent) + np.var(box2_cent))
        out = safe_division(num, den)
        # out = np.divide(num, den)
        return out

    def _scale_cost(self, box1, box2):
        box1_scale = np.array(box1, np.float32)[2:]
        box2_scale = np.array(box2, np.float32)[2:]
        # log.info(f"scale: {box1_scale}, {box2_scale}")
        # https://reference.wolfram.com/language/ref/NormalizedSquaredEuclideanDistance.html
        # scaled between 0 and 1
        num = np.var(box1_scale - box2_scale)
        den = 2 * (np.var(box1_scale) + np.var(box2_scale))
        out = safe_division(num, den)
        # out = np.divide(num, den)
        return out

    def _min_cost_neighbor(self, box: List[List], other_boxes: List[List[List]]):
        """
        Return the minimum scoring bbox relative to box
        """
        box_scores = []
        for obox in other_boxes:
            d_cost = self._distance_cost(box, obox)
            s_cost = self._scale_cost(box, obox)
            box_scores.append(0.5 * (d_cost + s_cost))
        if len(box_scores) > 0:
            min_idx = np.argmin(box_scores)
            min_cost = box_scores[min_idx]
            min_box = other_boxes[min_idx]
            # log.info(f"Min Cost: {min_cost}")
            return min_cost, min_box
        else:
            return 1.0, None

    def _cost_over_window(
        self,
        pred: List[List[List[List]]],
        idx: int,
        box: List[List],
        window: int,
        cost_f: str = "minimum",
    ):
        """
        cost of bbox compared to best neighbors in prior window number of frames
        """

        min_neighbors = []
        min_boxes = OrderedDict()
        for w in range(1, window + 1):
            frame_idx = idx - w
            min_neighbor, min_box = self._min_cost_neighbor(box, pred[frame_idx])
            min_neighbors.append(min_neighbor)
            if min_box is not None:
                min_boxes[frame_idx] = min_box

        if cost_f == "average":
            cost = sum(min_neighbors) / window
        elif cost_f == "minimum":
            cost = min(min_neighbors)
        else:
            raise ValueError(
                f'cost_f must be either "average" or "minimum". Not {cost_f}.'
            )

        return cost, min_boxes

    def calculate(self, preds):
        # initialize empty list of verified tracks (according to association alg.)
        # use sets to avoid redundant bboxes
        valid_tracks = [set() for _ in range(len(preds))]

        # define window of frames to consider
        start_idx = (
            self.window_length - 1
        )  # window_length = 4 -> start at index 3 (0 indexed)
        assert (
            start_idx >= 0
        ), "window_length must be greater than or equal to 1"  # make sure starting index is valid
        frame_idxs = range(
            start_idx, len(preds)
        )  # frames to loop over - b/c _cost_over_window is going to look at prior frames

        # TODO: multi-proc this (slow with large numbers of boxes)
        # TODO: just cast for i in frame_idxs to MP pool? - maybe not worth time to dev?
        # TODO: would need to perform cost_over_window, return (cost, i, min_boxes) in queue, then unify into set?
        # iterate over all frames in video
        for i in frame_idxs:
            for box in preds[i]:
                cost, min_boxes = self._cost_over_window(
                    preds, i, box, self.window_length
                )
                if cost <= self.thresh:
                    valid_tracks[i].add(box)
                    # convert to dict(frame_idx: box)
                    for frame_idx, value in min_boxes.items():
                        valid_tracks[frame_idx].add(value)

        # convert sets to list
        valid_tracks = [list(tracklet) for tracklet in valid_tracks]
        return valid_tracks
