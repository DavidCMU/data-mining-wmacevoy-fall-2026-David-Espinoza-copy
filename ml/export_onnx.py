"""
Export a trained run to ONNX, so it can run without Python or PyTorch: in a
browser (web/), on a phone, or from any language with an ONNX runtime.

    python export_onnx.py                       # newest finished run
    python export_onnx.py runs/20261013-104311-cnn
    python export_onnx.py --no-web              # don't copy to web/model.onnx

The exported graph is the *deployable* model, not the training one:

    input   "pixels"  float32 (batch, 1, 28, 28), 0 = background, 1 = ink
    output  "probs"   float32 (batch, 10), each row sums to 1

Standardization and softmax are baked in, so a caller only has to supply
pixels -- the MEAN/STD constants can't be forgotten or got wrong on the far
side. Dropout is switched off (eval mode) before export.

After exporting, the file is run through onnxruntime on real test images and
compared with PyTorch; the export fails loudly if they disagree.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

import mnist_data
import runs
from model import build

WEB_MODEL = Path(__file__).parent / "web" / "model.onnx"
OPSET = 18          # onnxruntime-web 1.20 (used by web/) supports up to 21
TOLERANCE = 1e-4    # max |onnx - torch| in probability


class Deployable(nn.Module):
    """Raw pixels in, probabilities out."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = (pixels - mnist_data.MEAN) / mnist_data.STD
        return self.model(x).softmax(dim=1)


def load_run(run_dir: Path) -> tuple[nn.Module, dict, dict]:
    config = runs.read_json(run_dir / "config.json", {})
    status = runs.read_json(run_dir / "status.json", {})
    if not (run_dir / "model.pt").exists():
        raise FileNotFoundError(f"{run_dir} has no model.pt (it never finished an epoch)")
    model = build(config["arch"], config.get("dropout", 0.2))
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    return model.eval(), config, status


def check(path: Path, model: nn.Module, pixels: np.ndarray) -> float:
    """Largest difference between onnxruntime and PyTorch on these pixels."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(["probs"], {"pixels": pixels})[0]
    with torch.no_grad():
        want = Deployable(model)(torch.from_numpy(pixels)).numpy()
    return float(np.abs(got - want).max())


def export(run_dir: Path, out: Path | None = None, sample: np.ndarray | None = None) -> dict:
    """Write run_dir/model.onnx (or `out`), verify it, and return a summary.

    `sample` is uint8 (n, 28, 28) images to verify on; by default the first
    256 test images.
    """
    model, config, status = load_run(run_dir)
    out = out or run_dir / "model.onnx"
    example = torch.rand(2, 1, 28, 28)
    torch.onnx.export(
        Deployable(model).eval(), (example,), str(out),
        input_names=["pixels"], output_names=["probs"],
        dynamic_shapes={"pixels": {0: torch.export.Dim("batch")}},
        opset_version=OPSET,
        external_data=False,       # one self-contained file, weights included
        dynamo=True,
    )

    # Record where the weights came from inside the file itself.
    m = onnx.load(str(out))
    meta = {"source": "data-mining ml/ MNIST pipeline", "run": run_dir.name,
            "arch": config["arch"], "best_epoch": str(status.get("best_epoch", "")),
            "test_acc": f"{status['test_acc']:.4f}" if "test_acc" in status else "",
            "input": "pixels float32 (N,1,28,28), 0=background 1=ink",
            "output": "probs float32 (N,10)"}
    onnx.helper.set_model_props(m, meta)
    onnx.save(m, str(out))
    onnx.checker.check_model(str(out))

    if sample is None:
        sample = mnist_data.load()["x_test"][:256]
    diff = check(out, model, sample.astype(np.float32)[:, None] / 255.0)
    if diff > TOLERANCE:
        raise RuntimeError(f"ONNX and PyTorch disagree by {diff:.2e} (> {TOLERANCE})")
    return {"path": str(out), "bytes": out.stat().st_size, "max_diff": diff, **meta}


def newest_finished() -> Path:
    for p in runs.list_runs():
        if (p / "model.pt").exists() and runs.state(p) in ("done", "stopped"):
            return p
    sys.exit("no finished run with a model.pt yet -- train one first")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", type=Path, nargs="?", help="default: newest finished run")
    p.add_argument("--no-web", action="store_true", help="don't copy to web/model.onnx")
    a = p.parse_args()
    info = export(a.run_dir or newest_finished())
    if not a.no_web:
        WEB_MODEL.parent.mkdir(exist_ok=True)
        shutil.copyfile(info["path"], WEB_MODEL)
        info["web"] = str(WEB_MODEL)
    print(json.dumps(info, indent=2))
