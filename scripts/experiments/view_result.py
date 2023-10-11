# used to view the video and bounding box predictions from
# a specific experiment run

from afdme import REPO_PATH
from afdme.vis import viz_video_results
from afdme.utils import load_results_json
from afdme.data import to_numpy
from pathlib import Path
import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", help="full path to the results folder to visualize")
    return parser


def load_video(filename, data_path=f"{REPO_PATH}/data/AFD-ME/mp4"):
    video_name = result["label"]["filename"]
    video_path = list(Path(data_path).glob(f"**/{video_name}"))[0]
    video = to_numpy(video_path, video_format="BGR")
    return video


if __name__ == "__main__":
    # parse args
    parser = create_parser()
    args = vars(parser.parse_args())

    # load results files
    full_exp_path = Path(str(args["result"])).absolute()
    results, results_files = load_results_json(full_exp_path)

    for result in results:
        video = {result["label"]["filename"]: load_video(result["label"]["filename"])}
        label = result["label"]
        preds = result["prediction"]

        viz_video_results(
            video,
            label,
            preds,
            10,
            params=result["parameters"],
            show=True,
            save=True,
            out_path=".",
            video_type=".mp4",
        )
