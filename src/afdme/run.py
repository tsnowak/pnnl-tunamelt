import os
from pathlib import Path
from typing import Dict, OrderedDict, Union

from afdme import REPO_PATH, log
from afdme.data import DataLoader, numpy_to_cv2
from afdme.vis import viz_video_results, write_results


def run(
    itr: int,
    filters: OrderedDict,
    params: Dict,
    dataloader: DataLoader,
    run_path: Path,
    max_vid_itrs: Union[int, None] = None,
):
    """
    run the filters on every video in the dataset;
    or on N=<max_vid_itrs> videos (for testing)
    """

    # create experiment folder
    curr_run_path = f"{str(run_path)}/{itr}"
    os.makedirs(curr_run_path, exist_ok=True)

    # initialize vid_itr counter
    vid_itr = 0
    # loop through dataset
    for video, label in dataloader:
        # stop if hit max_vid_itrs
        if (max_vid_itrs) and (vid_itr >= max_vid_itrs):
            break

        # create per video iteration output path
        log.info(f"Using video {label['video_id']}...")
        vid_path = f"{curr_run_path}/{label['video_id']}"
        os.makedirs(vid_path, exist_ok=True)
        os.chdir(vid_path)

        # run data through the filters in order
        log.info("Calculating filters...")
        outputs = OrderedDict()
        for idx, (filter_name, filter) in enumerate(filters.items()):
            log.info(f"\tCalculating {filter_name}...")
            if filter_name == "original":
                outputs["original"] = video[..., 2]
            else:
                tmp = list(outputs.items())[-1]
                outputs[filter_name] = filter.filter(tmp[1])

        # get filter outputs in order
        display = OrderedDict()
        pred = None
        idx = 0
        filters_list = list(filters.items())
        for name, output in outputs.items():
            if name == "original":
                display[name] = numpy_to_cv2(video, "HSV", "BGR")
            elif filters[name].__class__.__name__ == "ContourFilter":
                pred = output
                # continue
            elif filters[name].__class__.__name__ == "TrackletAssociation":
                # pred = output
                continue
            else:
                display[name] = numpy_to_cv2(
                    output, filters_list[idx][1].out_format, "BGR"
                )
            idx += 1

        # view and save videos with results + save results
        log.info("Visualizing filter outputs...")
        viz_video_results(
            display,
            label,
            pred,
            filters_list[1][1].fps,  # hacky, just get fps from a filter
            params=params,
            show=False,
            save=False,
            out_path=Path(),
            video_type=".mp4",
        )

        write_results(params, label, pred, out_path=vid_path)

        # update counter
        vid_itr += 1
