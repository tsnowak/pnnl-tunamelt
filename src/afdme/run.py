import os
import json
from pathlib import Path
from typing import Dict, OrderedDict, Union

from afdme import REPO_PATH, log
from afdme.filter import common, dft
from afdme.data import DataLoader, Dataset, numpy_to_cv2
from afdme.vis import viz_video_results, write_results
from afdme.utils import create_exp_dirs, load_params, generate_param_batches


def create_paths(args):
    """
    Save meta data; prep for run batches
    """

    # parse params arg
    assert (args["params"] is not None) and (
        len(args["params"])
    ) > 0, "Path to one or more params json files must be passed to the params argument"

    # create output folder structure
    run_path = create_exp_dirs(args["results_path"])
    params_list = load_params(run_path, args["params"])

    # save flag arguments to run path
    args_json = json.dumps(args, indent=4)
    with open(f"{run_path}/args.json", "w") as outfile:
        outfile.write(args_json)

    # create run batches
    param_batches = []
    for params in params_list:
        param_batches += generate_param_batches(params)

    return run_path, param_batches


def create_dataloader(args):
    if args["dataset"] == "test":
        data_path = f"{args['data_path']}/mp4/batched_test"
        label_path = f"{args['data_path']}/labels/cvat-video-1.1/batched_test"
    else:
        data_path = f"{args['data_path']}/mp4/{args['dataset']}"
        label_path = f"{args['data_path']}/labels/cvat-video-1.1/{args['dataset']}"
    dataloader = DataLoader(
        Dataset(videos=data_path, labels=label_path), split=args["dataset"]
    )
    return dataloader


def create_pipeline(args, params, fps):
    # define filter order
    # reinitialize filters given params
    mean_filter = common.MeanFilter(fps=fps, params=params["mean_filter"])
    turbine_filter = dft.DFTFilter(fps=fps, params=params["turbine_filter"])
    denoise_filter = common.GaussianBlurDenoiseFilter(
        fps=fps, params=params["denoise_filter"]
    )
    intensity_filter = common.IntensityFilter(
        fps=fps, params=params["intensity_filter"]
    )
    contour_filter = common.ContourFilter(params=params["contour_filter"])
    tracklet_association = common.TrackletAssociation(
        params=params["tracklet_association"]
    )

    # automatically load filters at runtime and ensure required filters are present
    # ie {"original": None} and {"contour_filter": contour_filter}
    filters = OrderedDict()
    filters.update({"original": None})
    for filter in args["filters"]:
        try:
            filters.update({filter: eval(filter)})
        except SyntaxError:
            raise SyntaxError(
                "The only valid filters to pass to --filters are: "
                + "[mean_filter, turbine_filter, "
                + "denoise_filter, intensity_filter, "
                + "tracklet_association"
            )
    # always end with contour_filter
    filters.update({"contour_filter": contour_filter})
    # unless tracklet_association present
    if "tracklet_association" in filters.keys():
        filters.move_to_end("tracklet_association")

    return filters


def run_pipeline(filters, video):
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
    return outputs


def get_pipeline_outputs(filters, outputs, video):
    # get filter outputs in order
    display = OrderedDict()
    pred = None
    idx = 0
    filters_list = list(filters.items())
    print(list(outputs.keys()))
    for name, output in outputs.items():
        if name == "original":
            display[name] = numpy_to_cv2(video, "HSV", "BGR")
        elif filters[name].__class__.__name__ == "ContourFilter":
            pred = output
        elif filters[name].__class__.__name__ == "TrackletAssociation":
            pred = output
        else:
            display[name] = numpy_to_cv2(output, filters_list[idx][1].out_format, "BGR")
        idx += 1

    return display, pred, filters_list


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

        outputs = run_pipeline(filters, video)
        display, pred, filters_list = get_pipeline_outputs(filters, outputs, video)

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
