from pathlib import Path
import numpy as np

from turbx import REPO_PATH, log
from turbx.data import Dataset, DataLoader

# VERIFIED

# create test output directory
file_path = Path(__file__).absolute()
out_path = Path(f"{file_path.parent}/outputs/{str(file_path.name).split('.')[0]}")
out_path.mkdir(parents=True, exist_ok=True)


def test_string():
    """Single video given as a string"""
    dl = DataLoader(
        Dataset(
            videos=f"{REPO_PATH}/data/mp4/2010-09-08_074500_HF_S001.mp4", labels=None,
        )
    )
    vid, _ = next(dl)
    assert isinstance(vid, np.ndarray), f"Returned video is not numpy array: {vid}"
    log.debug(f"Output video shape: {vid.shape}")


def test_path():
    """Single video given as a pathlib.Path"""
    dl = DataLoader(
        Dataset(
            videos=Path(f"{REPO_PATH}/data/mp4/2010-09-08_074500_HF_S001.mp4"),
            labels=None,
        )
    )
    _, _ = next(dl)


# test dataloading list of strings
def test_list_string():
    """List of videos given as strings"""
    dl = DataLoader(
        Dataset(
            videos=[
                f"{REPO_PATH}/data/mp4/2010-09-08_074500_HF_S001.mp4",
                f"{REPO_PATH}/data/mp4/2010-09-08_074500_HF_S002_S001.mp4",
            ],
            labels=None,
        )
    )
    _, _ = next(dl)
    _, _ = next(dl)


# test dataloading list of paths
def test_list_path():
    """List of videos given as pathlib.Paths"""
    dl = DataLoader(
        Dataset(
            videos=[
                Path(f"{REPO_PATH}/data/mp4/2010-09-08_074500_HF_S001.mp4"),
                Path(f"{REPO_PATH}/data/mp4/2010-09-08_074500_HF_S002_S001.mp4"),
            ],
            labels=None,
        )
    )
    _, _ = next(dl)
    _, _ = next(dl)


# test dataloading all in directory
def test_glob_string():
    """Directory of videos given as a string"""
    dl = DataLoader(Dataset(videos=f"{REPO_PATH}/data/mp4", labels=None,))
    log.debug(f"Number of files in the dataloader {len(dl)}")
    _, _ = next(dl)
    _, _ = next(dl)


# test dataloading all in directory
def test_glob_path():
    """Directory of videos given as a path"""
    dl = DataLoader(Dataset(videos=Path(f"{REPO_PATH}/data/mp4"), labels=None,))
    log.debug(f"Number of files in the dataloader {len(dl)}")
    _, _ = next(dl)
    _, _ = next(dl)


# test dataloading all in list of directories
def test_list_glob():
    """List of directories of videos given as strings"""
    dl = DataLoader(
        Dataset(videos=[f"{REPO_PATH}/data/mp4", f"{REPO_PATH}/data/mp4"], labels=None)
    )
    log.debug(f"Number of files in the dataloader {len(dl)}")
    _, _ = next(dl)
    _, _ = next(dl)


# test label loading
def test_label_file():
    """List of directories of videos and a label file"""
    dl = DataLoader(
        Dataset(
            videos=[f"{REPO_PATH}/data/mp4"],
            labels=f"{REPO_PATH}/data/labels/cvat-video-1.1/default/12.xml",
        ),
        split="default",
    )
    log.debug(f"Aligned data length: {len(dl.dataset.aligned_data['default'])}")
    video, label = next(dl)


def test_label_dir():
    """List of directories of videos and a label directory"""
    dl = DataLoader(
        Dataset(videos=[f"{REPO_PATH}/data/mp4"], labels=f"{REPO_PATH}/data/labels",),
        split="default",
    )
    log.debug(f"Aligned data length: {len(dl.dataset.aligned_data['default'])}")
    video, label = next(dl)


def test_label_dir_and_video_dir():
    """Videoo directory path and label directory path"""
    dl = DataLoader(
        Dataset(videos=f"{REPO_PATH}/data/mp4", labels=f"{REPO_PATH}/data/labels")
    )
    log.debug(f"Aligned data length: {len(dl.dataset.aligned_data['train'])}")
    video, label = next(dl)
