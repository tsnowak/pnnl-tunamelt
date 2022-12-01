from typing import List, Dict, Union, Optional, Callable
from copy import deepcopy
import numpy as np
from itertools import product
from scipy.optimize import linear_sum_assignment
from filterpy import kalman
from filterpy.common import Q_discrete_white_noise


class Tracker:
    """
    Goal: Generic object tracking class that is can be as simple as
        inputting subsequent detections, or as complex as custom similarity metrics,
        and custom Kalman formulations.

    MVP:
        [] - Simple step function to process new detections
        [] - Variable dt
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        score_fn: Callable,
        score_thresh: float,
        window: int = 1,
        use_kalman: bool = True,
        maximize: bool = False,
    ):
        """
        Args:
            x: np.ndarray - defines the shape of the initial state used in the Kalman filter
        """

        if use_kalman:
            self.base_kalman: kalman.KalmanFilter = kalman.KalmanFilter(
                dim_x=dim_x, dim_z=dim_z
            )

        self.score_fn = score_fn
        self.score_thresh = score_thresh
        self.tracks = set()
        self.hidden = set()
        self.step_idx = 0
        self.window = window
        self.maximize = maximize

    def step(self, detections: List):
        detections = np.asarray(detections, dtype=np.float32)

        # remove detections that never got redetected
        for x in self.hidden:
            if x.frame < self.step_idx - self.window:
                self.hidden.remove(x)
        # combine tracks and possible tracks
        tracks = self.tracks.union(self.hidden)

        # higher is better?
        similarity_matrix = self._compose_similarity_matrix(detections, tracks)
        row_ind, col_ind = linear_sum_assignment(similarity_matrix, self.maximize)
        best_pairs = zip(row_ind, col_ind)
        # TODO: keep pairs that are better than score_thresh

    def _compose_similarity_matrix(self, detections: np.ndarray, tracks: np.ndarray):
        # verify nonzero tracks and detections
        assert (
            len(detections) > 0 and len(tracks) > 0
        ), "Either detections or tracks is empty."

        # verify tracks and detections all have same shape
        datum_shape = detections[0].shape
        assert all(
            [datum_shape == a.shape for a in detections]
        ), "Detections must contain similarly sized data"
        assert all(
            [datum_shape == a.shape for a in tracks]
        ), "Detections and tracks must contain similarly sized data"

        try:
            _ = self.score_fn(detections[0], tracks[0])
        except Exception as e:
            raise ValueError(f"Error testing score function: {e}")

        # TODO: parallelize/optimize
        # TODO: make dimensions equal?
        matrix_shape = (len(detections), len(tracks))
        similarity_matrix = np.array(matrix_shape, dtype=np.float32)
        detection_indices = range(0, matrix_shape[0])
        track_indices = range(0, matrix_shape[1])
        indices = product(detection_indices, track_indices)
        for x, y in indices:
            similarity_matrix[x, y] = self.score_fn(detections[x], tracks[y])

        return similarity_matrix
