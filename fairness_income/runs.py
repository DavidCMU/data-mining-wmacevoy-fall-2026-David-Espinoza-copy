"""
runs.py — reads runs/<id>/ for app.py. No torch import, same as
ml/runs.py — the dashboard should be able to list and summarize runs even
on a machine that doesn't have (or is still installing) the training
dependencies.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # A run's status.json can be mid-write (train.py writes it
        # atomically via os.replace, but a read landing in the gap before
        # the first write is still possible); treat as "not available yet"
        # rather than crashing the dashboard.
        return None


def list_run_ids() -> list[str]:
    if not os.path.isdir(RUNS_DIR):
        return []
    ids = [
        name
        for name in os.listdir(RUNS_DIR)
        if os.path.isfile(os.path.join(RUNS_DIR, name, "config.json"))
    ]
    # Newest first, by the timestamp embedded in the run id's generated
    # names; falls back gracefully for custom --run-id values.
    return sorted(ids, reverse=True)


def read_metrics(run_id: str) -> pd.DataFrame:
    path = os.path.join(RUNS_DIR, run_id, "metrics.jsonl")
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def read_run(run_id: str) -> dict:
    run_dir = os.path.join(RUNS_DIR, run_id)
    return {
        "run_id": run_id,
        "config": _read_json(os.path.join(run_dir, "config.json")) or {},
        "status": _read_json(os.path.join(run_dir, "status.json")) or {"state": "unknown"},
        "test_report": _read_json(os.path.join(run_dir, "test_report.json")),
        "metrics": read_metrics(run_id),
    }


def read_test_arrays(run_id: str) -> dict | None:
    path = os.path.join(RUNS_DIR, run_id, "test.npz")
    if not os.path.exists(path):
        return None
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


def comparison_table(run_ids: list[str] | None = None) -> pd.DataFrame:
    """One row per finished run: the same kind of before/after table the
    README's crossed-7s section used, but for sex-based recall."""
    if run_ids is None:
        run_ids = list_run_ids()

    rows = []
    for run_id in run_ids:
        run = read_run(run_id)
        status = run["status"]
        if status.get("state") != "done":
            continue
        config = run["config"]
        rows.append(
            {
                "run_id": run_id,
                "model": config.get("model"),
                "bias_mitigation": config.get("bias_mitigation"),
                "seed": config.get("seed"),
                "test_accuracy": status.get("test_accuracy"),
                "male_recall": status.get("test_male_recall"),
                "female_recall": status.get("test_female_recall"),
                "tpr_gap": status.get("test_tpr_gap"),
                "selection_rate_gap": status.get("test_selection_rate_gap"),
            }
        )
    return pd.DataFrame(rows)
