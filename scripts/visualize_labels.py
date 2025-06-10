import argparse
from pathlib import Path
from typing import OrderedDict
from tunamelt.data import DataLoader, Dataset, numpy_to_cv2, label_to_per_frame_list
from tunamelt.run import create_dataloader, create_paths
from tunamelt import REPO_PATH, log
from tunamelt.vis import write_video

## Saves the videos from --data_path/--dataset with overlaid label bounding boxes to --out_path


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", help="dataset to use: [train, test]", default="train"
    )
    parser.add_argument(
        "--data_path",
        help="path to the root of the data directory",
        default=f"{REPO_PATH}/data/PNNL-TUNAMELT",
    )
    parser.add_argument(
        "--out_path",
        help="path at which to save the labeled videos",
        default=f"{REPO_PATH}/data/labeled_mp4",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = vars(parser.parse_args())
    full_out_path = Path(args["out_path"] + "/" + args["dataset"])
    log.info(f"Generating and saving videos with overlaid labels at {full_out_path}...")
    Path(full_out_path).mkdir(parents=True, exist_ok=True)

    fps = 10
    frame_delay = 1.0 / fps

    # compose the dataloader
    dataloader = create_dataloader(args)
    for video, label in dataloader:
        write_video(
            numpy_to_cv2(video, "HSV", "BGR"),
            str(Path(label["filename"]).with_suffix("")),
            fps,
            label_to_per_frame_list(label),
            None,
            video_length=None,
            out_path=full_out_path,
            video_type=".mp4",
        )
