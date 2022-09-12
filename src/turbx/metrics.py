from typing import List, Dict


def boxes_to_binary(pred: List):
    return [1 if len(x) > 0 else 0 for x in pred]


def roc(label: List, pred: List):
    """
    Plot of the True Positive Rate (TPR) versus the
    False Positive Rate (FPR) across a changing sensitivity parameter
    """
    raise NotImplementedError


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
        "tpr": float(tp) / p,
        "tnr": float(tn) / n,
        "fpr": float(fp) / n,
        "fnr": float(fn) / p,
    }


def iou(label: List, pred: List):
    """
    Intersection Over Union (IoU)
    For a given bounding box
    Prediction := P; Ground Truth := T
    Intersection (latex) := \cap; Union (latex) := \cup
    (P \cap T) / (P \cup T)
    """
    raise NotImplementedError
