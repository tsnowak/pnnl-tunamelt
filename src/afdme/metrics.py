from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing.pool import Pool
from afdme.data import label_to_per_frame_list, label_to_per_frame_targets


def calc_best_of_metrics(ids: List, multirun_metrics: List) -> Dict:
    amax_ap = np.argmax([x["AP"] for x in multirun_metrics])
    amax_ar = np.argmax([x["AR"] for x in multirun_metrics])
    amax_f1 = np.argmax([x["F1"] for x in multirun_metrics])
    amax_frr = np.argmax([x["frame_removal_rate"] for x in multirun_metrics])
    amax_tnfrr = np.argmax([x["tn_frame_removal_rate"] for x in multirun_metrics])
    amax_tdr = np.argmax([x["tdr"] for x in multirun_metrics])
    amax_frtdr = np.argmax(
        [(x["tdr"] + x["frame_removal_rate"]) / 2.0 for x in multirun_metrics]
    )
    satisfies_criterion = [
        (x["tdr"] >= 0.9 and x["frame_removal_rate"] >= 0.2) for x in multirun_metrics
    ]

    best_of_dict = {
        "AP": {ids[amax_ap]: multirun_metrics[amax_ap]},
        "AR": {ids[amax_ar]: multirun_metrics[amax_ar]},
        "F1": {ids[amax_f1]: multirun_metrics[amax_f1]},
        "FRR": {ids[amax_frr]: multirun_metrics[amax_frr]},
        "TNFRR": {ids[amax_tnfrr]: multirun_metrics[amax_tnfrr]},
        "TDR": {ids[amax_tdr]: multirun_metrics[amax_tdr]},
        "FRTDR": {ids[amax_frtdr]: multirun_metrics[amax_frtdr]},
        "satisfies_criterion": {
            id: metrics
            for id, metrics, sc in zip(ids, multirun_metrics, satisfies_criterion)
            if sc
        },
        "all_runs": {id: metrics for id, metrics in zip(ids, multirun_metrics)},
    }

    return best_of_dict


def calc_multi_video_metrics(results, results_files):
    mp_pool = Pool()
    per_video_results = mp_pool.starmap(
        calc_per_video_metrics, zip(results, results_files)
    )

    all_video_results = aggregate_metrics(per_video_results)

    return per_video_results, all_video_results


def aggregate_metrics(results):
    outputs = {
        "all_neg_dets": 0,
        "all_neg_labels": 0,
        "all_frames": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "tdrsum": 0,
    }

    for r in results:
        outputs["all_neg_dets"] += r["n_neg_dets"]
        outputs["all_neg_labels"] += r["n_neg_labels"]
        outputs["all_frames"] += len(r["binary_preds"])
        outputs["tp"] += r["tfpnr_dict"]["tp"]
        outputs["fp"] += r["tfpnr_dict"]["fp"]
        outputs["fn"] += r["tfpnr_dict"]["fn"]
        outputs["tn"] += r["tfpnr_dict"]["tn"]
        outputs["tdrsum"] += r["target_detection_rate"]

    outputs["AP"] = safe_division(outputs["tp"], (outputs["tp"] + outputs["fp"]))
    outputs["AR"] = safe_division(outputs["tp"], (outputs["tp"] + outputs["fn"]))
    outputs["F1"] = 2 / ((1 / outputs["AR"]) + (1 / outputs["AP"]))
    outputs["tdr"] = outputs["tdrsum"] / len(results)
    outputs["frame_removal_rate"] = outputs["all_neg_dets"] / outputs["all_frames"]
    outputs["tn_frame_rate"] = outputs["all_neg_labels"] / outputs["all_frames"]
    outputs["tn_frame_removal_rate"] = outputs["tn"] / outputs["all_neg_labels"]

    return outputs


def calc_per_video_metrics(result, filename):
    metric_results = {
        "results_filename": filename,
        "video_filename": result["label"]["filename"],
        "video_id": result["label"]["video_id"],
        "parameters": result["parameters"],
    }

    (
        metric_results["binary_labels"],
        metric_results["binary_preds"],
        metric_results["tfpnr_dict"],
    ) = calc_tfpnr(result["label"], result["prediction"], show=False, save=False)
    # binary_label, binary_pred, tfpnr_dict = calc_tfpnr(
    #    result["label"], result["prediction"], show=False, save=False
    # )
    (
        metric_results["unique_targs"],
        metric_results["detected_targs"],
        metric_results["target_detection_rate"],
    ) = calc_tdr(result["label"], result["prediction"])
    # unique_targs, det_targs, tdr = calc_tdr(
    #    result["label"], result["prediction"])
    _, metric_results["n_pos_dets"], metric_results["n_neg_dets"] = calc_frames_removed(
        result["prediction"]
    )
    # _, n_pos_dets, n_neg_dets = calc_frames_removed(result["prediction"])
    (
        _,
        metric_results["n_pos_labels"],
        metric_results["n_neg_labels"],
    ) = calc_frames_removed(label_to_per_frame_list(result["label"]))
    # _, n_pos_lab, n_neg_lab = calc_frames_removed(
    #    label_to_per_frame_list(result["label"])
    # )
    metric_results["perc_frames_removed"] = 100 * (
        metric_results["n_neg_dets"] / len(metric_results["binary_preds"])
    )
    metric_results["perc_neg_frame_removed"] = 100 * (
        metric_results["n_neg_dets"] / metric_results["n_neg_labels"]
    )

    return metric_results


def safe_division(n, d, value=0.0):
    return n / d if d else value


def boxes_to_binary(pred: List):
    return [1 if len(x) > 0 else 0 for x in pred]


def calc_box_area(box: List[List]):
    """
    Calculate the area of a [[xy][xy]] bbox
    """
    return (box[1][0] - box[0][0]) * (box[1][1] - box[0][1])


def avg_box_area(box_list: List[List]):
    """
    Calculate the area of a list of [xyxy] bboxes
    """
    box_areas = []
    for box in box_list:
        box_area = calc_box_area(box)
        box_areas.append(box_area)

    if len(box_areas) == 0:
        return np.nan
    else:
        return np.mean(box_areas)


def roc(label: List, pred: List):
    """
    Plot of the True Positive Rate (TPR) versus the
    False Positive Rate (FPR) across a changing sensitivity parameter
    """
    raise NotImplementedError


def target_detection_rate(targets: List, preds: List):
    detected = set()
    unique = set()
    # for each frame
    for i, frame in enumerate(targets):
        # if there is target
        if len(frame) > 0:
            # iterate through targets
            for target in frame:
                # add unique target
                unique.add(target)
                if preds[i] != 0:
                    # add if detection in frame
                    detected.add(target)

    tdr = safe_division(len(detected), len(unique), value=1.0)
    return len(unique), len(detected), tdr


def tfpnr(label: List, pred: List) -> Dict:
    """
    True Positive Rate (TPR)
    TP / All Positives :=
    TP / (#TP + #FN)
    ---
    True Negative Rate (TNR)
    TN / All Negatives :=
    TN / (#FP + #TN)
    ---
    False Positive Rate (FPR)
    FP / All Negatives :=
    FP / (#FP + #TN)
    ---
    False Negative Rate (FNR)
    FN / All Positives :=
    FN / (#TP + #FN)
    """
    assert len(label) == len(
        pred
    ), "Number of frames in labels and predictions should be the same."
    p = sum(label)
    n = len(label) - p
    tp, tn, fp, fn = 0, 0, 0, 0
    for l in range(len(label)):
        if (label[l] == 1) and (pred[l] == 1):
            tp += 1
        elif (label[l] == 0) and (pred[l] == 0):
            tn += 1
        elif (label[l] == 0) and (pred[l] == 1):
            fp += 1
        elif (label[l] == 1) and (pred[l] == 0):
            fn += 1

    return {
        "p": p,
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tpr": safe_division(float(tp), p),
        "tnr": safe_division(float(tn), n),
        "fpr": safe_division(float(fp), n),
        "fnr": safe_division(float(fn), p),
    }


def calc_size(label: Dict):
    """
    Calculate per frame average target size, video min size, video max size, and video average size
    """
    min_size, max_size, avg_size = None, None, None
    per_target_size = []
    per_frame_avg_size = [[] for _ in range(label["video_length"])]
    for track in label["tracks"]:
        for frame in track["frames"]:
            box_area = calc_box_area(frame["box"])
            per_frame_avg_size[frame["frame"]].append(box_area)
            per_target_size.append(box_area)
    per_frame_avg_size = [sum(x) / len(x) for x in per_frame_avg_size]
    min_size = min(per_target_size)
    max_size = max(per_target_size)
    avg_size = sum(per_target_size) / len(per_target_size)
    return per_frame_avg_size, min_size, max_size, avg_size


def calc_frames_removed(pred: List):
    binary_preds = boxes_to_binary(pred)
    pos_dets = [x for x in binary_preds if x == 1]
    neg_dets = [x for x in binary_preds if x == 0]
    return binary_preds, len(pos_dets), len(neg_dets)


def calc_tfpnr(label: Dict, pred: List, show=False, save=False, out_path=Path()):
    # only need list of bbox as labels
    label = label_to_per_frame_list(label)

    # convert to per frame binary target presence labels
    binary_label = boxes_to_binary(label)
    binary_pred = boxes_to_binary(pred)
    # calculare TPR and FPR metrics
    tfpnr_dict = tfpnr(binary_label, binary_pred)

    if show or save:
        # plot binary per frame results
        plt.figure("per_frame")
        plt.plot(binary_label)
        plt.plot(binary_pred)

        plt.figure("metrics")
        keys = ["tpr", "tnr", "fpr", "fnr"]
        data = [tfpnr_dict[k] for k in keys]
        plt.bar(keys, data)
        plt.ylim(bottom=0.0, top=1.0)

    if show:
        plt.show(block=False)
    if save:
        plt.savefig(out_path)

    return binary_label, binary_pred, tfpnr_dict


def calc_tdr(label: Dict, pred: List):
    # convert to per frame binary target presence labels
    targets = label_to_per_frame_targets(label)
    binary_pred = boxes_to_binary(pred)
    # calculare TPR and FPR metrics
    return target_detection_rate(targets, binary_pred)
