import imageio as iio
from turbx import REPO_PATH, log
from turbx.data import DataLoader, Dataset

# args = standard_parser()

if __name__ == "__main__":

    file_path = f"{REPO_PATH}/data/mp4"
    labels = f"{REPO_PATH}/data/labels"

    dataloader = DataLoader(Dataset(videos=file_path, labels=labels))

    # TODO can I get this from video file?
    fps = 10
    frame_delay = 1.0 / fps

    # get video, label
    video, label = dataloader[3]
    frame = 0
    log.info(f"Saving HSV of frame {frame}...")
    h = video[frame, ..., 0]
    s = video[frame, ..., 1]
    v = video[frame, ..., 2]
    iio.imwrite("hue.png", h)
    iio.imwrite("saturation.png", s)
    iio.imwrite("value.png", v)
