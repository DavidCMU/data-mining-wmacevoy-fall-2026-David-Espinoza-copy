"""
bias_mitigation.py — the bias this project addresses, and two ways to
correct for it during training. Mirrors the role of ml/sevens.py in the
template: that module corrected for crossed 7s being under-represented
among 7s; this one corrects for high earners being under-represented
among women.

## The bias

In the cleaned training data, women are a much smaller share of the
>$50K-income examples than they are of the population overall — i.e.
P(female | income>50K) << P(female). A classifier trained on this data
with no correction learns that pattern and reproduces it: at test time its
recall (true positive rate) for "predict >50K when the person actually
earns >50K" is substantially lower for women than for men. That gap —
not overall accuracy — is the bias this project measures and mitigates.
(See `fairness_metrics` in runs.py / app.py for exactly how it's computed
on a trained model, and README.md for the measured before/after numbers.)

## The two mitigations

Selected via `--bias-mitigation` in train.py, same three-way shape as
sevens.py's `--seven-balance`:

- "none"       — no correction. The baseline a reader compares everything
                 else against.
- "oversample" — at the start of every epoch, rows where (sex == Female
                 AND label == >50K) are duplicated at random until that
                 group's share of the >50K-labeled training pool matches
                 women's overall share of the training population. Exactly
                 sevens.py's "repeated at random" strategy, applied to the
                 analogous group here.
- "reweight"   — Kamiran & Calders (2012) reweighing: every training
                 example gets a static per-sample loss weight so that, in
                 the weighted training distribution, sex and income are
                 statistically independent. Unlike oversampling this
                 doesn't change which rows exist, just how much each one
                 counts in the loss.
"""
from __future__ import annotations

import numpy as np

CHOICES = ("none", "oversample", "reweight")


def compute_sample_weights(y: np.ndarray, sex: np.ndarray) -> np.ndarray:
    """Kamiran & Calders reweighing: per-(sex, label) weight so the
    weighted joint distribution of (sex, label) factorizes into the
    product of its marginals — i.e. sex carries no information about the
    (re-weighted) label distribution.

    weight(s, y) = (P(sex=s) * P(label=y)) / P(sex=s, label=y)

    Returns one weight per row of `y`/`sex`, suitable as a per-sample loss
    weight (e.g. in a weighted BCE loss).
    """
    n = len(y)
    weights = np.ones(n, dtype="float64")
    for s in (0, 1):
        for label in (0.0, 1.0):
            group_mask = (sex == s) & (y == label)
            count = group_mask.sum()
            if count == 0:
                continue
            p_sex = (sex == s).mean()
            p_label = (y == label).mean()
            p_joint = count / n
            weights[group_mask] = (p_sex * p_label) / p_joint
    return weights.astype("float32")


def oversample_indices(y: np.ndarray, sex: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a (possibly larger) index array into (y, sex) for one epoch,
    with (sex == Female, label == >50K) rows duplicated at random until
    their share of the >50K pool matches women's overall population share.

    All other rows are included exactly once. Call this fresh each epoch
    (with the same `rng`, advancing) so repeats aren't identical across
    epochs — the same live, per-epoch resampling sevens.py does for 7s.
    """
    n = len(y)
    all_idx = np.arange(n)
    female_target_share = (sex == 1).mean()

    positive_mask = y == 1.0
    positive_idx = all_idx[positive_mask]
    female_positive_idx = all_idx[positive_mask & (sex == 1)]
    male_positive_idx = all_idx[positive_mask & (sex == 0)]

    current_female_share = (
        len(female_positive_idx) / len(positive_idx) if len(positive_idx) > 0 else 0.0
    )
    if current_female_share >= female_target_share or len(female_positive_idx) == 0:
        # Already at/above the target share (or no positive female examples
        # to draw from at all) — nothing to add.
        extra = np.array([], dtype=int)
    else:
        # Solve for how many extra female-positive draws bring their share
        # of the (now larger) positive pool up to female_target_share:
        #   (len(fem_pos) + k) / (len(pos) + k) = target
        target = female_target_share
        k = (target * len(positive_idx) - len(female_positive_idx)) / (1 - target)
        k = max(0, int(round(k)))
        extra = rng.choice(female_positive_idx, size=k, replace=True)

    return np.concatenate([all_idx, extra])


def describe_mitigation(method: str, y: np.ndarray, sex: np.ndarray) -> dict:
    """Small human-readable summary of what a mitigation setting does to
    this particular training set — used by train.py to log into
    config.json and by app.py to explain the run.
    """
    female_share = float((sex == 1).mean())
    female_share_of_positives = float((sex[y == 1] == 1).mean()) if (y == 1).any() else 0.0
    info = {
        "method": method,
        "female_share_overall": round(female_share, 4),
        "female_share_of_positives_before": round(female_share_of_positives, 4),
    }
    if method == "oversample":
        rng = np.random.default_rng(0)
        idx = oversample_indices(y, sex, rng)
        resampled_y, resampled_sex = y[idx], sex[idx]
        female_share_of_positives_after = (
            float((resampled_sex[resampled_y == 1] == 1).mean()) if (resampled_y == 1).any() else 0.0
        )
        info["rows_added_example_epoch"] = int(len(idx) - len(y))
        info["female_share_of_positives_after"] = round(female_share_of_positives_after, 4)
    elif method == "reweight":
        weights = compute_sample_weights(y, sex)
        info["weight_range"] = [round(float(weights.min()), 3), round(float(weights.max()), 3)]
        info["female_positive_weight"] = round(float(weights[(sex == 1) & (y == 1)].mean()), 3) if ((sex == 1) & (y == 1)).any() else None
    return info
