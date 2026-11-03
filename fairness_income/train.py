#!/usr/bin/env python3
"""
train.py — trains one model, writing everything to runs/<id>/, mirroring
ml/train.py exactly in structure: config.json, status.json (live progress,
polled by app.py), metrics.jsonl (one line per epoch), model.pt (best
checkpoint by validation accuracy), test.npz (that checkpoint's test-set
predictions). The one addition over the template is fairness metrics
(see fairness_metrics.py) computed on validation every epoch and on the
test set at the end, since measuring bias is this project's whole point.

Run standalone:
  python train.py --model mlp --bias-mitigation reweight --epochs 20
or via pixi from the repo root:
  pixi run -e fairness-income train -- --model mlp --bias-mitigation reweight
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

import bias_mitigation
import census_data
from fairness_metrics import compute_fairness_report
from model import MODEL_CHOICES, build_model

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an income classifier with optional bias mitigation.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="mlp")
    parser.add_argument("--bias-mitigation", choices=bias_mitigation.CHOICES, default="none")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64, help="MLP hidden width (ignored for logreg)")
    parser.add_argument("--dropout", type=float, default=0.2, help="MLP dropout (ignored for logreg)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None, help="Defaults to a generated id")
    return parser.parse_args(argv)


def _write_json(path: str, data: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, path)  # atomic, so app.py never reads a half-written file


def _run_id_for(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{args.model}-{args.bias_mitigation}-{ts}-{uuid.uuid4().hex[:6]}"


def _batches(n: int, batch_size: int, rng: np.random.Generator, shuffle_idx: np.ndarray):
    order = shuffle_idx.copy()
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def main(argv=None, run_id: str | None = None) -> str:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    run_id = run_id or _run_id_for(args)
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    data = census_data.load_or_fetch()
    n_features = data.train.X.shape[1]

    mitigation_info = bias_mitigation.describe_mitigation(args.bias_mitigation, data.train.y, data.train.sex)

    config = {
        "run_id": run_id,
        "model": args.model,
        "bias_mitigation": args.bias_mitigation,
        "mitigation_info": mitigation_info,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "seed": args.seed,
        "n_features": n_features,
        "feature_names": data.feature_names,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(os.path.join(run_dir, "config.json"), config)
    _write_json(
        os.path.join(run_dir, "status.json"),
        {"state": "running", "epoch": 0, "epochs": args.epochs, "updated_at": datetime.now(timezone.utc).isoformat()},
    )

    model = build_model(args.model, n_features, hidden=args.hidden, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    train_X = torch.from_numpy(data.train.X)
    train_y = torch.from_numpy(data.train.y)
    val_X = torch.from_numpy(data.val.X)
    val_y_np = data.val.y

    # For "reweight": a static per-sample weight, computed once from the
    # un-resampled training labels. For "none"/"oversample": uniform weight
    # (oversampling does its work by changing WHICH rows appear, not how
    # much each one counts).
    if args.bias_mitigation == "reweight":
        base_weights = torch.from_numpy(bias_mitigation.compute_sample_weights(data.train.y, data.train.sex))
    else:
        base_weights = torch.ones(len(data.train.y), dtype=torch.float32)

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    best_val_acc = -1.0
    best_epoch = -1
    model_path = os.path.join(run_dir, "model.pt")

    all_train_idx = np.arange(len(data.train.y))

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            if args.bias_mitigation == "oversample":
                epoch_idx = bias_mitigation.oversample_indices(data.train.y, data.train.sex, rng)
            else:
                epoch_idx = all_train_idx

            epoch_loss_total = 0.0
            epoch_examples = 0
            for batch_idx in _batches(len(epoch_idx), args.batch_size, rng, epoch_idx):
                xb = train_X[batch_idx]
                yb = train_y[batch_idx]
                wb = base_weights[batch_idx]

                optimizer.zero_grad()
                logits = model(xb)
                per_example_loss = loss_fn(logits, yb)
                loss = (per_example_loss * wb).mean()
                loss.backward()
                optimizer.step()

                epoch_loss_total += float(loss.item()) * len(batch_idx)
                epoch_examples += len(batch_idx)

            model.eval()
            with torch.no_grad():
                val_logits = model(val_X)
                val_probs = torch.sigmoid(val_logits).numpy()
            val_pred = (val_probs >= 0.5).astype("int64")
            val_report = compute_fairness_report(val_y_np.astype("int64"), val_pred, data.val.sex)

            metrics_line = {
                "epoch": epoch,
                "train_loss": epoch_loss_total / max(epoch_examples, 1),
                "val_accuracy": val_report["overall"]["accuracy"],
                "val_tpr_gap": val_report["tpr_gap"],
                "val_male_recall": val_report["male"]["recall"],
                "val_female_recall": val_report["female"]["recall"],
            }
            metrics_file.write(json.dumps(metrics_line) + "\n")
            metrics_file.flush()

            _write_json(
                os.path.join(run_dir, "status.json"),
                {
                    "state": "running",
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "val_accuracy": val_report["overall"]["accuracy"],
                    "val_tpr_gap": val_report["tpr_gap"],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            val_acc = val_report["overall"]["accuracy"] or 0.0
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)

    # Evaluate the BEST checkpoint (by validation accuracy) on the held-out
    # test set — scored once, here, at the very end, same discipline as
    # ml/train.py.
    best_model = build_model(args.model, n_features, hidden=args.hidden, dropout=args.dropout)
    best_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    best_model.eval()
    with torch.no_grad():
        test_logits = best_model(torch.from_numpy(data.test.X))
        test_probs = torch.sigmoid(test_logits).numpy()
    test_pred = (test_probs >= 0.5).astype("int64")
    test_report = compute_fairness_report(data.test.y.astype("int64"), test_pred, data.test.sex)

    np.savez(
        os.path.join(run_dir, "test.npz"),
        y_true=data.test.y.astype("int64"),
        y_pred=test_pred,
        probs=test_probs.astype("float32"),
        sex=data.test.sex,
    )

    _write_json(
        os.path.join(run_dir, "status.json"),
        {
            "state": "done",
            "epoch": args.epochs,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
            "test_accuracy": test_report["overall"]["accuracy"],
            "test_tpr_gap": test_report["tpr_gap"],
            "test_male_recall": test_report["male"]["recall"],
            "test_female_recall": test_report["female"]["recall"],
            "test_selection_rate_gap": test_report["selection_rate_gap"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_json(os.path.join(run_dir, "test_report.json"), test_report)

    return run_id


if __name__ == "__main__":
    # Computed once, here, and threaded through to main() — so if training
    # fails partway, the failure handler below writes status.json into the
    # SAME run directory main() already started writing to, not a freshly
    # (and differently) generated one.
    _cli_args = parse_args()
    _run_id = _run_id_for(_cli_args)

    try:
        finished_run_id = main(run_id=_run_id)
        print(f"done: {finished_run_id}")
    except Exception:
        # Record the failure where app.py (and a human re-reading the run
        # later) will actually see it, then re-raise so the process still
        # exits non-zero and the traceback still reaches stderr/the log file
        # app.py redirected this process's output to.
        import traceback

        try:
            failed_run_dir = os.path.join(RUNS_DIR, _run_id)
            os.makedirs(failed_run_dir, exist_ok=True)
            _write_json(
                os.path.join(failed_run_dir, "status.json"),
                {
                    "state": "failed",
                    "error": traceback.format_exc(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception:
            pass  # don't let failure-reporting itself hide the original error
        raise
