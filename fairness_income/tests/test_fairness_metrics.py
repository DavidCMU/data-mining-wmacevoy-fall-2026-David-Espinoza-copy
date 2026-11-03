"""
Tests for fairness_metrics.py — the equal-opportunity (TPR gap) report.
"""
from __future__ import annotations

import numpy as np
import pytest

from fairness_metrics import compute_fairness_report


def test_group_metrics_sum_to_overall():
    rng = np.random.default_rng(0)
    n = 1000
    y_true = rng.choice([0, 1], size=n)
    y_pred = rng.choice([0, 1], size=n)
    sex = rng.choice([0, 1], size=n)

    report = compute_fairness_report(y_true, y_pred, sex)
    assert report["male"]["n"] + report["female"]["n"] == report["overall"]["n"]


def test_tpr_gap_is_zero_when_no_bias():
    rng = np.random.default_rng(0)
    n = 4000
    sex = rng.choice([0, 1], size=n)
    y_true = rng.choice([0, 1], size=n)
    # A "model" that's equally accurate for both groups: predicts the true
    # label with the same noise rate regardless of sex.
    noise = rng.random(n) < 0.1
    y_pred = np.where(noise, 1 - y_true, y_true)

    report = compute_fairness_report(y_true, y_pred, sex)
    assert abs(report["tpr_gap"]) < 0.05


def test_tpr_gap_detects_injected_bias():
    rng = np.random.default_rng(0)
    n = 4000
    sex = rng.choice([0, 1], size=n)  # 1 = female
    y_true = rng.choice([0, 1], size=n)

    # Model "misses" about half of qualified women (flips their prediction
    # to 0) on top of 5% uniform noise for everyone else.
    uniform_noise = rng.random(n) < 0.05
    y_pred = np.where(uniform_noise, 1 - y_true, y_true)
    female_qualified = (sex == 1) & (y_true == 1)
    extra_miss = rng.random(n) < 0.5
    y_pred = np.where(female_qualified & extra_miss, 0, y_pred)

    report = compute_fairness_report(y_true, y_pred, sex)
    assert report["tpr_gap"] > 0.2  # men's recall notably higher than women's
    assert report["male"]["recall"] > report["female"]["recall"]


def test_safe_div_returns_none_for_empty_group():
    # If a "group" has zero people (e.g. a tiny slice in a smoke test),
    # metrics for it should come back as None rather than raising.
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    sex = np.array([0, 0, 0, 0])  # no females at all in this batch

    report = compute_fairness_report(y_true, y_pred, sex)
    assert report["female"]["n"] == 0
    assert report["female"]["recall"] is None
    assert report["tpr_gap"] is None  # can't compute a gap without both sides
