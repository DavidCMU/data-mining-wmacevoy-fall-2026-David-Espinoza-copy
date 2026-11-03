"""
Tests for runs.py — the no-torch module app.py uses to read runs/<id>/.
Builds fake run directories by hand (config.json/status.json/metrics.jsonl/
test.npz) rather than actually training anything.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import runs


@pytest.fixture
def fake_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS_DIR", str(tmp_path))
    return tmp_path


def _write_run(runs_dir, run_id, *, state, model="mlp", mitigation="none", seed=0, extra_status=None):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)

    config = {
        "run_id": run_id,
        "model": model,
        "bias_mitigation": mitigation,
        "seed": seed,
        "mitigation_info": {"method": mitigation},
    }
    (run_dir / "config.json").write_text(json.dumps(config))

    status = {"state": state}
    if extra_status:
        status.update(extra_status)
    (run_dir / "status.json").write_text(json.dumps(status))

    metrics_lines = [
        {"epoch": 1, "train_loss": 0.5, "val_accuracy": 0.8, "val_tpr_gap": 0.3},
        {"epoch": 2, "train_loss": 0.4, "val_accuracy": 0.82, "val_tpr_gap": 0.25},
    ]
    with open(run_dir / "metrics.jsonl", "w") as f:
        for line in metrics_lines:
            f.write(json.dumps(line) + "\n")

    if state == "done":
        np.savez(
            run_dir / "test.npz",
            y_true=np.array([1, 0, 1, 0]),
            y_pred=np.array([1, 0, 0, 0]),
            probs=np.array([0.9, 0.1, 0.4, 0.2]),
            sex=np.array([0, 1, 1, 0]),
        )
        test_report = {
            "overall": {"accuracy": 0.82},
            "male": {"recall": 0.9},
            "female": {"recall": 0.6},
            "tpr_gap": 0.3,
            "selection_rate_gap": 0.1,
        }
        (run_dir / "test_report.json").write_text(json.dumps(test_report))

    return run_dir


def test_list_run_ids_only_includes_dirs_with_config(fake_runs_dir):
    _write_run(fake_runs_dir, "run-a", state="done")
    (fake_runs_dir / "not-a-run").mkdir()  # no config.json inside

    ids = runs.list_run_ids()
    assert ids == ["run-a"]


def test_list_run_ids_sorted_newest_first(fake_runs_dir):
    _write_run(fake_runs_dir, "mlp-none-20260101T000000-aaaaaa", state="done")
    _write_run(fake_runs_dir, "mlp-none-20260201T000000-bbbbbb", state="done")

    ids = runs.list_run_ids()
    assert ids[0] == "mlp-none-20260201T000000-bbbbbb"


def test_read_metrics_parses_jsonl(fake_runs_dir):
    _write_run(fake_runs_dir, "run-a", state="running")
    df = runs.read_metrics("run-a")
    assert list(df["epoch"]) == [1, 2]
    assert df["val_accuracy"].iloc[-1] == 0.82


def test_read_run_missing_directory_returns_safe_defaults(fake_runs_dir):
    run = runs.read_run("does-not-exist")
    assert run["config"] == {}
    assert run["status"] == {"state": "unknown"}
    assert run["test_report"] is None
    assert run["metrics"].empty


def test_read_run_for_finished_run(fake_runs_dir):
    _write_run(fake_runs_dir, "run-done", state="done")
    run = runs.read_run("run-done")
    assert run["config"]["model"] == "mlp"
    assert run["status"]["state"] == "done"
    assert run["test_report"]["tpr_gap"] == 0.3


def test_read_test_arrays(fake_runs_dir):
    _write_run(fake_runs_dir, "run-done", state="done")
    arrays = runs.read_test_arrays("run-done")
    assert arrays is not None
    assert list(arrays["y_true"]) == [1, 0, 1, 0]

    assert runs.read_test_arrays("run-without-test-npz") is None


def test_comparison_table_includes_only_done_runs(fake_runs_dir):
    _write_run(
        fake_runs_dir,
        "mlp-none-1",
        state="done",
        mitigation="none",
        extra_status={
            "test_accuracy": 0.82,
            "test_male_recall": 0.9,
            "test_female_recall": 0.65,
            "test_tpr_gap": 0.25,
            "test_selection_rate_gap": 0.1,
        },
    )
    _write_run(
        fake_runs_dir,
        "mlp-reweight-1",
        state="done",
        mitigation="reweight",
        extra_status={
            "test_accuracy": 0.80,
            "test_male_recall": 0.87,
            "test_female_recall": 0.83,
            "test_tpr_gap": 0.04,
            "test_selection_rate_gap": 0.02,
        },
    )
    _write_run(fake_runs_dir, "mlp-none-still-running", state="running")

    table = runs.comparison_table()
    assert set(table["run_id"]) == {"mlp-none-1", "mlp-reweight-1"}

    none_row = table[table["run_id"] == "mlp-none-1"].iloc[0]
    reweight_row = table[table["run_id"] == "mlp-reweight-1"].iloc[0]
    assert none_row["tpr_gap"] > reweight_row["tpr_gap"]


def test_read_json_handles_corrupt_file_gracefully(fake_runs_dir):
    run_dir = fake_runs_dir / "run-corrupt"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{not valid json")
    assert runs._read_json(str(run_dir / "config.json")) is None
