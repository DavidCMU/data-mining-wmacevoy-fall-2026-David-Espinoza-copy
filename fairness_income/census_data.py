"""
census_data.py — load, clean, and encode the UCI Adult (Census Income)
dataset, mirroring the role of ml/mnist_data.py in the template this
project is based on: download -> cache -> clean -> split -> standardize.

Dataset: "Adult" / "Census Income" (OpenML id 1590, version 2 — the
commonly-used cleaned version with missing values as NaN rather than "?").
~48,842 rows of 1994 US Census data; the task is predicting whether a
person's income is above $50K/year from demographic and employment
features.

Why this dataset for a bias/fairness project: it's one of the most studied
fairness benchmarks in ML. `sex` (Male/Female) is a protected attribute
present in the data, and a plain classifier trained on it reproduces a
real, measurable bias — see bias_mitigation.py and README.md for exactly
what that bias is and how this project addresses it.

Caching: the raw fetch (via sklearn's fetch_openml, which hits openml.org)
happens once; after that we cache a cleaned, encoded copy as
data/adult.npz, the same pattern mnist_data.py uses for MNIST.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_PATH = os.path.join(DATA_DIR, "adult.npz")

# Columns used as model features. `sex` is deliberately EXCLUDED from the
# feature matrix the model trains on (the model should predict income from
# qualifications/work, not from sex directly) — it's tracked separately as
# the protected attribute used only for measuring and mitigating bias, never
# as an input to the network. `fnlwgt` (a census sampling weight) is dropped
# because it isn't a property of the person. `education-num` is kept and
# `education` (its categorical, duplicate-information twin) is dropped.
NUMERIC_FEATURES = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL_FEATURES = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "native-country",
]
PROTECTED_ATTRIBUTE = "sex"
TARGET = "class"

RANDOM_SEED = 42


@dataclass
class CensusSplit:
    X: np.ndarray  # float32, standardized numeric + one-hot categorical features
    y: np.ndarray  # float32, 1.0 if income > $50K else 0.0
    sex: np.ndarray  # int64, 1 if Female else 0 (the protected attribute)
    feature_names: list[str]


@dataclass
class CensusData:
    train: CensusSplit
    val: CensusSplit
    test: CensusSplit
    feature_names: list[str]


def _fetch_raw() -> pd.DataFrame:
    """Download the Adult dataset from OpenML (id 1590, version 2)."""
    from sklearn.datasets import fetch_openml  # imported lazily; needs network once

    # data_id already pins an exact dataset version on OpenML; passing
    # `version=` as well is rejected by sklearn ("you can only specify a
    # numeric data_id or a version, not both").
    bunch = fetch_openml(data_id=1590, as_frame=True, parser="pandas")
    frame = bunch.frame.copy()
    # OpenML's `class` target sometimes arrives as a different column name
    # depending on sklearn version; normalize it.
    if TARGET not in frame.columns and "class" not in frame.columns:
        frame = frame.rename(columns={bunch.target_names[0]: TARGET})
    return frame


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any missing value in the columns we use.

    The standard approach for this dataset (used in the original UCI
    distribution and most published analyses) is to drop rows with missing
    workclass/occupation/native-country rather than impute them — about 7%
    of rows, concentrated in a few categorical columns.
    """
    keep_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [PROTECTED_ATTRIBUTE, TARGET]
    frame = frame[keep_cols].copy()
    frame = frame.dropna(axis=0, how="any")
    return frame.reset_index(drop=True)


def _encode(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Turn the cleaned frame into (X, y, sex, feature_names).

    Numeric columns pass through as float64 (standardized later, after the
    split, using train-set statistics only — see `_standardize`).
    Categorical columns are one-hot encoded with pandas.get_dummies.
    """
    numeric = frame[NUMERIC_FEATURES].astype("float64")
    categorical = pd.get_dummies(frame[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)
    X_frame = pd.concat([numeric, categorical], axis=1)
    feature_names = list(X_frame.columns)
    X = X_frame.to_numpy(dtype="float64")

    target_str = frame[TARGET].astype(str).str.strip()
    # OpenML's version 2 uses "<=50K" / ">50K" (no trailing period, unlike
    # the raw UCI .test file's "<=50K." / ">50K.").
    y = (target_str.str.startswith(">50K")).to_numpy(dtype="float64")

    sex_str = frame[PROTECTED_ATTRIBUTE].astype(str).str.strip()
    sex = (sex_str == "Female").to_numpy(dtype="int64")

    return X, y, sex, feature_names


def _split_indices(n: int, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70/15/15 train/val/test split, shuffled with a fixed seed.

    (The original UCI distribution ships a fixed train/test file split, but
    OpenML's single combined table doesn't preserve that split, so — like
    mnist_data.py's own from-scratch split of MNIST's 60k training images
    into train/val — we make our own seeded, reproducible one.)
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _standardize(
    train_X: np.ndarray, val_X: np.ndarray, test_X: np.ndarray, n_numeric: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize only the numeric columns (one-hot columns are already 0/1),
    using mean/std computed from the TRAINING set only, applied to all three
    splits — the same discipline mnist_data.py uses for pixel standardization,
    and the reason the held-out test score stays an honest estimate.
    """
    mean = train_X[:, :n_numeric].mean(axis=0)
    std = train_X[:, :n_numeric].std(axis=0)
    std[std == 0] = 1.0  # guard against a constant column

    def apply(X: np.ndarray) -> np.ndarray:
        out = X.copy()
        out[:, :n_numeric] = (out[:, :n_numeric] - mean) / std
        return out

    return apply(train_X), apply(val_X), apply(test_X)


def load_or_fetch() -> CensusData:
    """Return the cached, cleaned, split, standardized dataset, fetching
    and building the cache on first call."""
    if os.path.exists(CACHE_PATH):
        cached = np.load(CACHE_PATH, allow_pickle=True)
        feature_names = list(cached["feature_names"])
        return CensusData(
            train=CensusSplit(cached["train_X"], cached["train_y"], cached["train_sex"], feature_names),
            val=CensusSplit(cached["val_X"], cached["val_y"], cached["val_sex"], feature_names),
            test=CensusSplit(cached["test_X"], cached["test_y"], cached["test_sex"], feature_names),
            feature_names=feature_names,
        )

    raw = _fetch_raw()
    cleaned = _clean(raw)
    X, y, sex, feature_names = _encode(cleaned)

    train_idx, val_idx, test_idx = _split_indices(len(cleaned))
    train_X, val_X, test_X = _standardize(X[train_idx], X[val_idx], X[test_idx], n_numeric=len(NUMERIC_FEATURES))

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(
        CACHE_PATH,
        train_X=train_X.astype("float32"),
        train_y=y[train_idx].astype("float32"),
        train_sex=sex[train_idx],
        val_X=val_X.astype("float32"),
        val_y=y[val_idx].astype("float32"),
        val_sex=sex[val_idx],
        test_X=test_X.astype("float32"),
        test_y=y[test_idx].astype("float32"),
        test_sex=sex[test_idx],
        feature_names=np.array(feature_names, dtype=object),
    )
    return load_or_fetch()


if __name__ == "__main__":
    data = load_or_fetch()
    print(f"train: {data.train.X.shape}, val: {data.val.X.shape}, test: {data.test.X.shape}")
    print(f"{len(data.feature_names)} features")
    for name, split in [("train", data.train), ("val", data.val), ("test", data.test)]:
        female_rate = split.sex.mean()
        pos_rate = split.y.mean()
        print(f"  {name}: {split.X.shape[0]} rows, {female_rate:.1%} female, {pos_rate:.1%} income >$50K")
