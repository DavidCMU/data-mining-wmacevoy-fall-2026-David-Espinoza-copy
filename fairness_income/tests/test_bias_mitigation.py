"""
Tests for bias_mitigation.py: the reweighing math and the oversampling
target-share math, on synthetic data with a known, injected sex/label bias.
"""
from __future__ import annotations

import numpy as np
import pytest

import bias_mitigation


def _biased_labels(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    sex = rng.choice([0, 1], size=n, p=[0.6, 0.4])  # 1 = female
    # Women are much less likely to have label 1, by construction.
    prob_pos = np.where(sex == 1, 0.1, 0.5)
    y = (rng.random(n) < prob_pos).astype("float64")
    return y, sex, rng


def test_compute_sample_weights_achieves_statistical_independence():
    y, sex, _ = _biased_labels()
    weights = bias_mitigation.compute_sample_weights(y, sex)

    assert weights.shape == y.shape
    assert (weights > 0).all()

    # Under the weighted distribution, P(sex=1, label=1) should factorize
    # into P(sex=1) * P(label=1) (within floating-point tolerance).
    total_w = weights.sum()
    p_fem_w = weights[sex == 1].sum() / total_w
    p_pos_w = weights[y == 1].sum() / total_w
    p_joint_w = weights[(sex == 1) & (y == 1)].sum() / total_w

    assert abs(p_joint_w - p_fem_w * p_pos_w) < 1e-6


def test_compute_sample_weights_upweights_the_disadvantaged_group():
    y, sex, _ = _biased_labels()
    weights = bias_mitigation.compute_sample_weights(y, sex)
    # Female-positive rows are rare relative to what independence would
    # predict, so reweighing should scale them up.
    female_positive_weight = weights[(sex == 1) & (y == 1)].mean()
    male_positive_weight = weights[(sex == 0) & (y == 1)].mean()
    assert female_positive_weight > male_positive_weight


def test_oversample_indices_brings_female_share_of_positives_to_target():
    y, sex, rng = _biased_labels()
    female_target_share = (sex == 1).mean()

    idx = bias_mitigation.oversample_indices(y, sex, rng)
    resampled_y, resampled_sex = y[idx], sex[idx]

    female_share_after = (resampled_sex[resampled_y == 1] == 1).mean()
    assert abs(female_share_after - female_target_share) < 0.02


def test_oversample_indices_only_adds_rows_never_removes():
    y, sex, rng = _biased_labels()
    idx = bias_mitigation.oversample_indices(y, sex, rng)
    # Every original index must still appear at least once.
    assert set(np.arange(len(y))).issubset(set(idx.tolist()))
    # Negative-label rows are untouched (duplicates only come from the
    # female-positive group).
    neg_idx = np.arange(len(y))[y == 0]
    counts = {i: 0 for i in neg_idx}
    for i in idx:
        if i in counts:
            counts[i] += 1
    assert all(c == 1 for c in counts.values())


def test_oversample_is_a_noop_when_already_balanced():
    n = 2000
    rng = np.random.default_rng(1)
    sex = rng.choice([0, 1], size=n, p=[0.5, 0.5])
    y = rng.choice([0.0, 1.0], size=n)  # no sex/label relationship at all
    idx = bias_mitigation.oversample_indices(y, sex, rng)
    # Share should already be close to the target, so little/no padding
    # should be added.
    assert len(idx) - n < 0.05 * n


def test_describe_mitigation_reports_expected_keys():
    y, sex, _ = _biased_labels()

    none_info = bias_mitigation.describe_mitigation("none", y, sex)
    assert none_info["method"] == "none"
    assert "female_share_overall" in none_info
    assert "female_share_of_positives_before" in none_info

    over_info = bias_mitigation.describe_mitigation("oversample", y, sex)
    assert "female_share_of_positives_after" in over_info
    # Oversampling should move the female share of positives up, toward
    # (not necessarily all the way to, since this is a single-epoch draw)
    # the overall female share.
    assert over_info["female_share_of_positives_after"] > over_info["female_share_of_positives_before"]

    reweight_info = bias_mitigation.describe_mitigation("reweight", y, sex)
    assert "weight_range" in reweight_info
    assert reweight_info["weight_range"][0] <= reweight_info["weight_range"][1]
