"""
Read, start, and stop training runs. Everything the dashboard knows about a
run comes from the files train.py writes; nothing here imports torch.

Liveness is judged by the files, not the process: a run whose status says
"running" but whose metrics.jsonl hasn't grown in STALE_SECONDS is reported as
"lost" (killed, laptop slept, ...). That works the same on every OS, which
probing a PID does not -- on Windows os.kill(pid, 0) *terminates* the process.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from train import RUNS_DIR

HERE = Path(__file__).parent
STALE_SECONDS = 120


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def read_metrics(run_dir: Path) -> list[dict]:
    """All complete lines of metrics.jsonl. A half-written last line is skipped."""
    rows = []
    try:
        text = (run_dir / "metrics.jsonl").read_text()
    except OSError:
        return rows
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def frames(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Split metric rows into (batch frame, epoch frame, sample snapshots)."""
    batch = pd.DataFrame([r for r in rows if r["kind"] == "batch"])
    epoch = pd.DataFrame([r for r in rows if r["kind"] == "epoch"])
    samples = [r for r in rows if r["kind"] == "samples"]
    return batch, epoch, samples


def state(run_dir: Path, now: float | None = None) -> str:
    status = read_json(run_dir / "status.json", {})
    s = status.get("state", "unknown")
    if s == "running":
        metrics = run_dir / "metrics.jsonl"
        last = metrics.stat().st_mtime if metrics.exists() else status.get("started", 0)
        if (now or time.time()) - last > STALE_SECONDS:
            return "lost"
    return s


def list_runs(root: Path = RUNS_DIR) -> list[Path]:
    """Run directories, newest first."""
    if not root.exists():
        return []
    return sorted((p for p in root.iterdir() if (p / "config.json").exists()),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def start(config: dict, root: Path = RUNS_DIR) -> Path:
    """Launch train.py as a detached process; return its run directory.

    Detached so that a Streamlit rerun, a closed tab, or a restarted server
    doesn't take the training down with it.
    """
    run_dir = root / f"{time.strftime('%Y%m%d-%H%M%S')}-{config['arch']}"
    run_dir.mkdir(parents=True)
    args = [sys.executable, str(HERE / "train.py"), "--run-dir", str(run_dir)]
    for k, v in config.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    detach = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32"
              else {"start_new_session": True})
    log = (run_dir / "stdout.log").open("w")
    subprocess.Popen(args, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, **detach)
    return run_dir


def stop(run_dir: Path) -> None:
    """Ask a run to stop; it notices at its next training step."""
    (run_dir / "STOP").touch()


def montage(images: np.ndarray, cols: int, scale: int = 1, pad: int = 1) -> np.ndarray:
    """Tile (n, h, w) weights into one uint8 image, each tile scaled to its own range.

    Per-tile scaling shows each filter's *pattern*; it hides how strong one
    filter is relative to another, which is fine for "what does it look for".
    """
    n, h, w = images.shape
    rows = -(-n // cols)
    lo = images.min(axis=(1, 2), keepdims=True)
    hi = images.max(axis=(1, 2), keepdims=True)
    tiles = (images - lo) / np.where(hi > lo, hi - lo, 1)
    sheet = np.ones((rows * (h + pad) + pad, cols * (w + pad) + pad))
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[pad + r * (h + pad):pad + r * (h + pad) + h,
              pad + c * (w + pad):pad + c * (w + pad) + w] = tile
    sheet = np.kron(sheet, np.ones((scale, scale)))
    return (sheet * 255).astype(np.uint8)
