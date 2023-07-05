from typing import List, Dict
import numpy as np


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
