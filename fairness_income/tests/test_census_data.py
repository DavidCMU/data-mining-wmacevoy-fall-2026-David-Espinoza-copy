"""
Tests for census_data.py's pure pandas/numpy logic — clean, encode, split,
standardize. No network, no OpenML fetch: each test builds a small
synthetic DataFrame with the same column names/dtypes the real Adult
dataset has, and feeds it straight to the internal (`_`-prefixed)
functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import census_data


def _fake_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    sex = rng.choice(["Male", "Female"], size=n, p=[0.6, 0.4])
    # Bake in a bias: females are much less likely to be labeled >50K even
    # at the same "merit" (hours-per-week), so tests on this fixture can
    # exercise the bias-detection/mitigation path meaningfully elsewhere.
    hours = rng.integers(20, 60, size=n)
    merit = (hours - hours.mean()) / hours.std()
    female_penalty = np.where(sex == "Female", -1.5, 0.0)
    prob_high = 1 / (1 + np.exp(-(merit + female_penalty)))
    label = np.where(rng.random(n) < prob_high, ">50K", "<=50K")

    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=n),
            "workclass": rng.choice(["Private", "Self-emp", "Government"], size=n),
            "fnlwgt": rng.integers(10_000, 500_000, size=n),
            "education": rng.choice(["Bachelors", "HS-grad", "Masters"], size=n),
            "education-num": rng.integers(5, 16, size=n),
            "marital-status": rng.choice(["Married", "Never-married"], size=n),
            "occupation": rng.choice(["Tech", "Sales", "Admin"], size=n),
            "relationship": rng.choice(["Husband", "Wife", "Not-in-family"], size=n),
            "race": rng.choice(["White", "Black", "Other"], size=n),
            "sex": sex,
            "capital-gain": rng.integers(0, 5000, size=n),
            "capital-loss": np.zeros(n),  # constant column -> exercises the zero-std guard
            "hours-per-week": hours,
            "native-country": rng.choice(["United-States", "Mexico"], size=n),
            "class": label,
        }
    )
    # Inject a few missing values the way the real dataset has them.
    frame.loc[frame.index[:5], "workclass"] = np.nan
    return frame


def test_clean_drops_rows_with_missing_values():
    frame = _fake_frame()
    assert frame.isna().any(axis=None)
    cleaned = census_data._clean(frame)
    assert not cleaned.isna().any(axis=None)
    # Exactly the 5 injected-NaN rows should be gone (none of the other
    # columns had NaNs injected).
    assert len(cleaned) == len(frame) - 5


def test_encode_shapes_and_binary_targets():
    frame = census_data._clean(_fake_frame())
    X, y, sex, feature_names = census_data._encode(frame)

    assert X.shape[0] == len(frame)
    assert X.shape[1] == len(feature_names)
    assert set(np.unique(y)).issubset({0, 1})
    assert set(np.unique(sex)).issubset({0, 1})
    # sex=1 means Female, by convention.
    assert sex[(frame["sex"] == "Female").to_numpy()].min() == 1
    assert sex[(frame["sex"] == "Male").to_numpy()].max() == 0
    # The protected attribute must not leak into the feature matrix.
    assert "sex" not in feature_names
    assert not any(name.startswith("sex_") for name in feature_names)


def test_split_indices_are_disjoint_and_cover_everything():
    n = 500
    train_idx, val_idx, test_idx = census_data._split_indices(n)
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    assert len(np.unique(all_idx)) == n  # disjoint and covers every row
    # Roughly 70/15/15.
    assert abs(len(train_idx) / n - 0.7) < 0.02
    assert abs(len(val_idx) / n - 0.15) < 0.02
    assert abs(len(test_idx) / n - 0.15) < 0.02


def test_split_indices_is_deterministic():
    a = census_data._split_indices(300)
    b = census_data._split_indices(300)
    for part_a, part_b in zip(a, b):
        assert np.array_equal(part_a, part_b)


def test_standardize_gives_zero_mean_unit_std_and_handles_constant_column():
    frame = census_data._clean(_fake_frame())
    X, y, sex, feature_names = census_data._encode(frame)
    train_idx, val_idx, test_idx = census_data._split_indices(len(y))
    n_numeric = len(census_data.NUMERIC_FEATURES)

    train_X, val_X, test_X = census_data._standardize(
        X[train_idx], X[val_idx], X[test_idx], n_numeric
    )

    # Training numeric columns should be ~standardized.
    numeric_train = train_X[:, :n_numeric]
    assert np.allclose(numeric_train.mean(axis=0), 0, atol=1e-6)
    stds = numeric_train.std(axis=0)
    # capital-loss was constant -> zero-std guard should avoid dividing by
    # zero (no NaN/inf produced), everything else should be ~1.
    assert np.isfinite(train_X).all()
    assert np.isfinite(val_X).all()
    assert np.isfinite(test_X).all()
    non_constant = [s for s in stds if s > 1e-9]
    assert all(abs(s - 1.0) < 1e-6 for s in non_constant)

    # Categorical (one-hot) columns are untouched by standardization.
    assert set(np.unique(train_X[:, n_numeric:])).issubset({0.0, 1.0})
