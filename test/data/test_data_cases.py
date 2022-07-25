from pathlib import Path
import numpy as np

from turbx import REPO_PATH, log
from turbx.data import Dataset, DataLoader, to_numpy, parse_labels
from turbx.filter import dft

# VERIFIED

# create test output directory
file_path = Path(__file__).absolute()
out_path = Path(f"{file_path.parent}/outputs/{str(file_path.name).split('.')[0]}")
out_path.mkdir(parents=True, exist_ok=True)

# create filter class used in each test
fps = 10
filter = dft.DFTFilter(fps=fps)


def test_single_video():
    """Case when a single video with no labels has filters run on it"""

    file_path = f"{REPO_PATH}/data/mp4/2010-09-08_074500_HF_S001.mp4"
    labels = None

    video = to_numpy(file_path)
    filter.calculate(video, fps)
    out = filter.filter(video)

    assert isinstance(out, np.ndarray), "filter output is not numpy array"


def test_multiple_video():
    """Case when multiple videos with no labels have filters run on them"""

    file_path = f"{REPO_PATH}/data/mp4"
    labels = None

    dataloader = DataLoader(Dataset(videos=file_path, labels=labels))
    cntr = 0
    while cntr < 2:
        try:
            video, label = next(dataloader)
            filter.calculate(video, fps)
            out = filter.filter(video)
            assert isinstance(out, np.ndarray), "filter output is not numpy array"
            cntr += 1
        except StopIteration:
            return None


def test_single_video_with_labels():
    """Case when a single video with labels has filters run on it"""

    file_path = f"{REPO_PATH}/data/mp4/2010-09-08_074500_HF_S001.mp4"
    labels = f"{REPO_PATH}/data/annotations"

    video = to_numpy(file_path)
    labels = parse_labels(labels)
    filter.calculate(video, fps)
    out = filter.filter(video)
    assert isinstance(out, np.ndarray), "filter output is not numpy array"
    # score = metric(out, labels)


def test_multiple_video_with_labels():
    """Case when multiple videos with labels have filters run on them"""

    file_path = f"{REPO_PATH}/data/mp4"
    labels = f"{REPO_PATH}/data/labels"

    dataloader = DataLoader(Dataset(videos=file_path, labels=labels))
    cntr = 0
    while cntr < 2:
        try:
            video, label = next(dataloader)
            filter.calculate(video, fps)
            out = filter.filter(video)
            assert isinstance(out, np.ndarray), "filter output is not numpy array"
            # score = metric(out, label)
            cntr += 1
        except StopIteration:
            return None
