# used to view the video and bounding box predictions from
# a specific experiment run

import argparse
from pathlib import Path

from tunamelt.data import load_video
from tunamelt.utils import load_results_json
from tunamelt.vis import viz_video_results


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", help="full path to the results folder to visualize")
    return parser


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
            save=False,
            out_path=None,
            video_type=".mp4",
        )
