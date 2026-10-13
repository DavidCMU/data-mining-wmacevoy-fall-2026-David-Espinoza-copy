"""Offline tests: no download, a fake 'MNIST' of random digits, a few seconds total."""
import gzip
import json
import struct

import numpy as np
import pytest
import torch

import mnist_data
import runs
import train
from model import ARCHITECTURES, build, first_layer_filters


def idx_bytes(a: np.ndarray) -> bytes:
    return struct.pack(">HBB", 0, 0x08, a.ndim) + struct.pack(f">{a.ndim}I", *a.shape) + a.tobytes()


def test_parse_idx_round_trip():
    a = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    assert np.array_equal(mnist_data.parse_idx(idx_bytes(a)), a)
    assert np.array_equal(mnist_data.parse_idx(gzip.decompress(gzip.compress(idx_bytes(a)))), a)


def test_parse_idx_rejects_truncated_and_bad_magic():
    a = np.zeros((2, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        mnist_data.parse_idx(idx_bytes(a)[:-1])
    with pytest.raises(ValueError):
        mnist_data.parse_idx(b"\x00\x00\x0d\x01" + idx_bytes(a)[4:])   # float32 code


def test_split_is_disjoint_and_reproducible():
    x, y = np.arange(100)[:, None], np.arange(100)
    (xt, yt), (xv, yv) = mnist_data.split(x, y, val_size=20, seed=3)
    assert len(yv) == 20 and len(yt) == 80
    assert set(yt).isdisjoint(yv) and set(yt) | set(yv) == set(range(100))
    assert np.array_equal(mnist_data.split(x, y, 20, seed=3)[1][1], yv)


def test_standardize_shape_and_scale():
    x = np.full((3, 28, 28), 255, dtype=np.uint8)
    out = mnist_data.standardize(x)
    assert out.shape == (3, 1, 28, 28) and out.dtype == np.float32
    assert out[0, 0, 0, 0] == pytest.approx((1 - mnist_data.MEAN) / mnist_data.STD)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_models_map_images_to_ten_logits(arch):
    model = build(arch)
    assert model(torch.zeros(4, 1, 28, 28)).shape == (4, 10)
    f = first_layer_filters(model)
    assert f.shape[1:] == ((3, 3) if arch == "cnn" else (28, 28))


@pytest.fixture
def fake_mnist(monkeypatch):
    """Digits whose label is drawn as a bright row, so a model can actually learn them."""
    rng = np.random.default_rng(0)

    def make(n):
        y = rng.integers(0, 10, n).astype(np.uint8)
        x = rng.integers(0, 40, (n, 28, 28)).astype(np.uint8)
        x[np.arange(n), 2 + 2 * y, :] = 255
        return x, y

    (xtr, ytr), (xte, yte) = make(600), make(200)
    monkeypatch.setattr(mnist_data, "load",
                        lambda path=None: {"x_train": xtr, "y_train": ytr,
                                           "x_test": xte, "y_test": yte})


def test_train_writes_a_complete_run(tmp_path, fake_mnist):
    args = train.parse_args(["--epochs", "2", "--val-size", "100", "--batch-size", "50",
                             "--log-every", "2", "--run-dir", str(tmp_path / "r")])
    run_dir = train.train(args)

    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "done" and status["test_acc"] > 0.5
    assert runs.state(run_dir) == "done"

    batch, epoch, samples = runs.frames(runs.read_metrics(run_dir))
    assert list(epoch["epoch"]) == [0, 1, 2]
    assert batch["step"].is_monotonic_increasing
    assert [len(s["pred"]) for s in samples] == [train.N_SAMPLES] * 3
    assert np.load(run_dir / "filters.npy").shape == (3, 16, 3, 3)
    t = np.load(run_dir / "test.npz")
    assert t["probs"].shape == (200, 10)


def test_stop_file_ends_run_as_stopped(tmp_path, fake_mnist):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    runs.stop(run_dir)
    train.train(train.parse_args(["--epochs", "3", "--val-size", "100",
                                  "--run-dir", str(run_dir)]))
    assert runs.state(run_dir) == "stopped"
    assert not (run_dir / "STOP").exists()


def test_read_metrics_skips_half_written_line(tmp_path):
    (tmp_path / "metrics.jsonl").write_text('{"kind": "batch", "step": 1}\n{"kind": "ba')
    assert runs.read_metrics(tmp_path) == [{"kind": "batch", "step": 1}]


def test_running_run_goes_lost_when_metrics_stop_growing(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"state": "running", "started": 0}))
    (tmp_path / "metrics.jsonl").write_text("")
    mtime = (tmp_path / "metrics.jsonl").stat().st_mtime
    assert runs.state(tmp_path, now=mtime + 1) == "running"
    assert runs.state(tmp_path, now=mtime + runs.STALE_SECONDS + 1) == "lost"


def test_montage_tiles_and_scales():
    sheet = runs.montage(np.random.default_rng(0).normal(size=(5, 3, 3)), cols=4, scale=2)
    assert sheet.shape == ((2 * 4 + 1) * 2, (4 * 4 + 1) * 2) and sheet.dtype == np.uint8
    assert runs.montage(np.zeros((1, 3, 3)), cols=1).max() <= 255      # flat tile: no NaN


def test_onnx_export_matches_pytorch(tmp_path, fake_mnist):
    import onnxruntime as ort

    import export_onnx

    run_dir = train.train(train.parse_args(["--epochs", "1", "--val-size", "100",
                                            "--run-dir", str(tmp_path / "r")]))
    info = export_onnx.export(run_dir)
    assert info["max_diff"] < export_onnx.TOLERANCE and info["arch"] == "cnn"

    # Raw [0, 1] pixels in, probabilities out, any batch size.
    session = ort.InferenceSession(info["path"], providers=["CPUExecutionProvider"])
    probs = session.run(["probs"], {"pixels": np.random.rand(5, 1, 28, 28).astype(np.float32)})[0]
    assert probs.shape == (5, 10)
    assert np.allclose(probs.sum(1), 1, atol=1e-5)
