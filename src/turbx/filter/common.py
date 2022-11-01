from typing import Optional, List, Tuple, Dict
import numpy as np
import cv2
from findpeaks import findpeaks
from turbx import log
from turbx.filter.base import OfflineFilter
from turbx.vis import xyxy_to_xywh


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


class NlMeansDenoiseFilter(OfflineFilter):
    def __init__(
        self,
        video: Optional[np.ndarray] = None,
        fps: Optional[int] = None,
        params: Optional[Dict] = {
            "filter_strength": 20,
            "template_size": 31,
            "search_size": 61,
        },
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
        self.filter_strength = params["filter_strength"]
        self.template_size = params["template_size"]
        self.window_size = params["window_size"]

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

        ## per frame NlMeansDenoising -> VERY SLOW
        for i, frame in enumerate(value_channel):
            frame = cv2.fastNlMeansDenoising(
                frame, None, self.filter_strength, self.template_size, self.window_size
            )
            value_channel[i, ...] = frame

        ## time-windowed NlMeansDenoising -> VERY SLOW
        # batch_size = 5
        # for i in range(len(value_channel)):
        #    start = i
        #    end = i + 5
        #    diff = (len(value_channel) - 1) - start
        #    avg_frame = 2
        #    # adjust for end of video
        #    if diff < 5:
        #        start = i - (5 - diff)  # set start to 5 from end
        #        end = len(value_channel) - 1  # end of video
        #        avg_frame = i  # set to current frame
        #    batch = value_channel[start:end, ...]
        #    print(len(batch))
        #    frame = cv2.fastNlMeansDenoisingMulti(
        #        batch, avg_frame, batch_size, None, 4, 7, 35
        #    )
        #    value_channel[i, ...] = frame

        return value_channel


class FindPeaksFilter(OfflineFilter):
    def __init__(
        self,
        video: Optional[np.ndarray] = None,
        fps: Optional[int] = None,
        method: Optional[str] = "lee",
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
        self.method = findpeaks(
            method="topology",
            scale=False,
            denoise=method,
            togray=True,
            imsize=False,
            window=15,
        )

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
            - value_channel: value_channel to filter - np.array [N, H, W]
            - fps: fps of value_channel - int
        """
        for idx, frame in enumerate(value_channel):
            out = self.method.fit(frame)
            value_channel[idx, ...] = out["Xproc"]

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
        out = np.multiply(video, self.mask)
        out = out.astype(np.uint8)
        log.debug(f"Intensity filtered video of shape: {out.shape}")
        return out

    def calculate(
        self,
        video: np.ndarray,
        fps: int,
    ):
        self.fps = fps
        # video = video.astype(np.float32)
        # video = video.astype(np.float16)
        # std_val = np.max(np.std(video, axis=0))
        # max_val = np.max(video, axis=0)

        # log.debug(f"Max value: {np.max(max_val)}")
        ## take average of n-largest pixel maxima
        # max_val = np.mean(np.sort(max_val, axis=None)[-self.n :])
        # log.debug(f"std and max: {std_val, max_val}")

        # over entire image
        # only one pixel passes this
        # std = np.std(video)
        # max = np.max(video)
        # print(std, max_val)

        # only keep pixels in the video that are within std of their max
        # apply mask in apply
        mask = video > (self.thresh)
        self.mask = mask

        log.debug(f"Generated intensity filter mask of shape: {mask.shape}")

        return mask


class DilateErodeFilter(OfflineFilter):
    def __init__(
        self,
        video: Optional[np.ndarray] = None,
        fps: Optional[int] = None,
        kernel_shape: Optional[Tuple] = (5, 5),
    ):
        """
        Try to remove speckles by building up then eroding
        """
        super().__init__(video, fps)
        if self.mask is None:
            log.debug(f"Did not generate {self.__class__} filter.")
        else:
            log.debug(f"Generated {self.__class__} filter mask.")
        self.fps = fps
        self.out_format = "GRAY"
        self.kernel_shape = kernel_shape

    def filter(
        self,
        video: np.ndarray,
        fps: Optional[int] = None,
    ):
        if fps is None:
            if self.fps is None:
                raise ValueError("fps not given.")
            fps = self.fps
        out = self.calculate(video, fps)
        log.debug(f"BuildUpErode filtered video of shape: {out.shape}")
        return out

    def calculate(
        self,
        video: np.ndarray,
        fps: int,
    ):
        video = video.copy()
        self.fps = fps
        kernel = np.ones(self.kernel_shape, np.uint8)
        # dilation_kernel = np.ones((self.dilation, self.dilation), np.uint8)
        # erosion_kernel = np.ones((self.erosion, self.erosion), np.uint8)
        # diff_kernel = np.ones(
        #    (self.erosion - self.dilation, self.erosion - self.dilation), np.uint8
        # )
        for idx, frame in enumerate(video):
            opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
            # frame = cv2.erode(frame, erosion_kernel, iterations=1)
            # frame = cv2.dilate(frame, dilation_kernel, iterations=1)
            # frame = cv2.dilate(frame, diff_kernel, iterations=1)
            video[idx, ...] = opening

        return video


class ContourFilter:
    def __init__(
        self,
        video: Optional[np.ndarray] = None,
        params: Optional[Dict] = {"min_area": 200, "max_area": 6000},
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
        # print(f"centroid: {box1_cent}, {box2_cent}")
        # https://reference.wolfram.com/language/ref/NormalizedSquaredEuclideanDistance.html
        # scaled between 0 and 1
        return np.var(box1_cent - box2_cent) / (
            2 * (np.var(box1_cent) + np.var(box2_cent))
        )

    def _scale_cost(self, box1, box2):
        box1_scale = np.array(box1, np.float32)[2:]
        box2_scale = np.array(box2, np.float32)[2:]
        # print(f"scale: {box1_scale}, {box2_scale}")
        # https://reference.wolfram.com/language/ref/NormalizedSquaredEuclideanDistance.html
        # scaled between 0 and 1
        return np.var(box1_scale - box2_scale) / (
            2 * (np.var(box1_scale) + np.var(box2_scale))
        )

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
            # print(f"Min Cost: {min_cost}")
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
        min_boxes = []
        for w in range(1, window + 1):
            min_neighbor, min_box = self._min_cost_neighbor(box, pred[idx - w])
            min_neighbors.append(min_neighbor)
            min_boxes.append(min_box)

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
        window = self.window_length - 1
        assert window >= 0, "window_length must be greater than or equal to 1"
        frame_idxs = range(window, len(preds))

        # TODO: implement forward and backward verificaton
        # iterate over all frames in video
        for i in frame_idxs:
            for box in preds[i]:
                cost, min_boxes = self._cost_over_window(
                    preds, i, box, self.window_length
                )
                if cost <= self.thresh:
                    valid_tracks[i].add(box)
                    for j in range(1, self.window_length + 1):
                        if min_boxes[j - 1] is not None:
                            valid_tracks[i - j].add(min_boxes[j - 1])
                            # TODO: box interpolation
        # convert sets to list
        valid_tracks = [list(tracklet) for tracklet in valid_tracks]
        return valid_tracks
