"""
fairness_metrics.py — the measurements this whole project is judging
itself by. Pure numpy (no torch), so it's usable from train.py (during
training, every epoch), runs.py (reading a finished run, no torch
dependency), and app.py (the dashboard) alike — the same no-torch-in-the-
reading-path discipline ml/runs.py uses.

The headline number is `tpr_gap`: the difference in recall (true positive
rate — of people who actually earn >$50K, what fraction the model
correctly flags) between men and women. This is the "equal opportunity"
fairness criterion (Hardt, Price & Srebro, 2016): it asks whether
qualified people are caught at the same rate regardless of group,
independent of how many people in each group happen to qualify.

A large positive tpr_gap means the model is missing qualified women at a
higher rate than qualified men — the bias this project's mitigations
target. tpr_gap near 0 means roughly equal treatment of qualified people
across groups.
"""
from __future__ import annotations

import numpy as np

GROUP_NAMES = {0: "male", 1: "female"}


def _safe_div(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def _group_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    n = tp + fp + fn + tn
    return {
        "n": int(n),
        "base_rate": _safe_div(tp + fn, n),  # how often this group actually earns >50K
        "selection_rate": _safe_div(tp + fp, n),  # how often the model predicts >50K
        "accuracy": _safe_div(tp + tn, n),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),  # a.k.a. true positive rate (TPR)
        "false_positive_rate": _safe_div(fp, fp + tn),
    }


def compute_fairness_report(y_true: np.ndarray, y_pred: np.ndarray, sex: np.ndarray) -> dict:
    """y_true/y_pred: 0/1 arrays. sex: 0=male, 1=female (see census_data.py).

    Returns {"overall": {...}, "male": {...}, "female": {...}, "tpr_gap": float|None,
    "selection_rate_gap": float|None} where the two gaps are male - female
    (positive means the model favors men on that metric).
    """
    report = {"overall": _group_metrics(y_true, y_pred)}
    for code, name in GROUP_NAMES.items():
        mask = sex == code
        report[name] = _group_metrics(y_true[mask], y_pred[mask])

    male_recall = report["male"]["recall"]
    female_recall = report["female"]["recall"]
    report["tpr_gap"] = (
        male_recall - female_recall if male_recall is not None and female_recall is not None else None
    )

    male_sel = report["male"]["selection_rate"]
    female_sel = report["female"]["selection_rate"]
    report["selection_rate_gap"] = (
        male_sel - female_sel if male_sel is not None and female_sel is not None else None
    )
    return report
