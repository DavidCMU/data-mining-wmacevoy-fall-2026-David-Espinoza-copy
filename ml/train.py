"""
Train one model on MNIST and record everything to runs/<run-id>/.

    python train.py --arch cnn --epochs 5
    python train.py --arch mlp --lr 0.01 --optimizer sgd --train-size 10000
    python train.py --seven-balance augment   # see sevens.py

A run directory holds:

    config.json     the arguments, so a run can be reproduced
    status.json     running | done | stopped | failed, plus the test score
    metrics.jsonl   one JSON object per line, appended as training goes:
                      {"kind": "batch", ...}     every --log-every steps
                      {"kind": "epoch", ...}     train and validation scores
                      {"kind": "samples", ...}   predictions on 32 fixed images
    filters.npy     first-layer weights after each epoch, (epochs, n, h, w)
    model.pt        weights from the epoch with the best validation accuracy
    test.npz        that model's test-set labels, predictions, probabilities

The Streamlit app reads these files while they grow; it never imports the
model in-process. Touching runs/<run-id>/STOP asks the run to stop cleanly at
the next step.
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

import mnist_data
import sevens
from model import build, first_layer_filters, n_params

RUNS_DIR = Path(__file__).parent / "runs"
N_SAMPLES = 32          # fixed validation images whose predictions we track


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--arch", choices=["cnn", "mlp"], default="cnn")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--train-size", type=int, default=0,
                   help="use only this many training images (0 = all 55,000)")
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seven-balance", choices=sevens.BALANCE, default="none",
                   help="add crossed 7s each epoch until the 7 pool is 50:50 (see sevens.py)")
    p.add_argument("--log-every", type=int, default=20, help="steps between batch logs")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="where to write (default runs/<timestamp>-<arch>)")
    return p.parse_args(argv)


def write_json(path: Path, obj) -> None:
    """Write via a temp file so a reader never sees half a file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1000):
    """Mean loss, accuracy, and class probabilities over a whole dataset."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total, probs = 0.0, []
    for i in range(0, len(x), batch_size):
        logits = model(x[i:i + batch_size])
        total += loss_fn(logits, y[i:i + batch_size]).item()
        probs.append(logits.softmax(1))
    probs = torch.cat(probs)
    acc = (probs.argmax(1) == y).float().mean().item()
    return total / len(x), acc, probs


def train(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir or RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{args.arch}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    write_json(run_dir / "config.json", config)
    status = {"state": "running", "pid": os.getpid(), "started": time.time()}
    write_json(run_dir / "status.json", status)
    metrics = (run_dir / "metrics.jsonl").open("a", buffering=1)   # line-buffered
    stop_file = run_dir / "STOP"

    def log(**row):
        metrics.write(json.dumps({**row, "t": round(time.time() - status["started"], 2)}) + "\n")

    try:
        torch.manual_seed(args.seed)
        d = mnist_data.load()
        # Split positions rather than images, so tags can follow the images.
        (i_tr, _), (i_va, _) = mnist_data.split(
            np.arange(len(d["y_train"])), d["y_train"], args.val_size, args.seed)
        if args.train_size:
            i_tr = i_tr[:args.train_size]
        x_tr, y_tr = d["x_train"][i_tr], d["y_train"][i_tr]
        x_va, y_va = d["x_train"][i_va], d["y_train"][i_va]
        if len(x_tr) == 0:
            raise ValueError(f"no training images left after a {args.val_size}-image validation set")

        def tensors(x, y):
            return torch.from_numpy(mnist_data.standardize(x)), torch.from_numpy(y.astype(np.int64))

        x_tr, y_tr = tensors(x_tr, y_tr)
        x_va, y_va = tensors(x_va, y_va)
        x_te, y_te = tensors(d["x_test"], d["y_test"])

        n_extra, x_crossed = 0, None
        if args.seven_balance != "none":
            crossed = sevens.crossed_mask(i_tr, len(d["y_train"])) & (y_tr.numpy() == 7)
            n_extra = sevens.n_extra(y_tr.numpy(), crossed)
            x_crossed = x_tr[torch.from_numpy(crossed)]

        model = build(args.arch, args.dropout)
        opt = (torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == "adam"
               else torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9))
        loss_fn = nn.CrossEntropyLoss()
        status.update(n_params=n_params(model), n_train=len(x_tr), n_val=len(x_va),
                      n_extra_crossed7=n_extra)
        write_json(run_dir / "status.json", status)

        steps_per_epoch = (len(x_tr) + n_extra + args.batch_size - 1) // args.batch_size
        filters, best_acc, step = [], -1.0, 0
        gen = torch.Generator().manual_seed(args.seed)

        # Epoch 0 = the untrained network, so every chart starts from chance.
        val_loss, val_acc, val_probs = evaluate(model, x_va, y_va)
        log(kind="epoch", epoch=0, step=0, val_loss=val_loss, val_acc=val_acc)
        log(kind="samples", epoch=0, pred=val_probs[:N_SAMPLES].argmax(1).tolist(),
            conf=val_probs[:N_SAMPLES].max(1).values.tolist())
        filters.append(first_layer_filters(model).numpy())

        for epoch in range(1, args.epochs + 1):
            model.train()
            x_ep, y_ep = x_tr, y_tr
            if n_extra:        # fresh extras every epoch
                x_ep = torch.cat([x_tr, sevens.extras(x_crossed, n_extra, args.seven_balance, gen)])
                y_ep = torch.cat([y_tr, torch.full((n_extra,), 7)])
            order = torch.randperm(len(x_ep), generator=gen)
            run_loss = run_correct = run_n = 0
            for b in range(steps_per_epoch):
                if stop_file.exists():
                    raise KeyboardInterrupt
                idx = order[b * args.batch_size:(b + 1) * args.batch_size]
                xb, yb = x_ep[idx], y_ep[idx]
                logits = model(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
                run_loss += loss.item() * len(yb)
                run_correct += (logits.argmax(1) == yb).sum().item()
                run_n += len(yb)
                if step % args.log_every == 0:
                    log(kind="batch", epoch=epoch - 1 + (b + 1) / steps_per_epoch, step=step,
                        loss=loss.item(), acc=(logits.argmax(1) == yb).float().mean().item())

            val_loss, val_acc, val_probs = evaluate(model, x_va, y_va)
            log(kind="epoch", epoch=epoch, step=step, train_loss=run_loss / run_n,
                train_acc=run_correct / run_n, val_loss=val_loss, val_acc=val_acc)
            log(kind="samples", epoch=epoch, pred=val_probs[:N_SAMPLES].argmax(1).tolist(),
                conf=val_probs[:N_SAMPLES].max(1).values.tolist())
            filters.append(first_layer_filters(model).numpy())
            np.save(run_dir / "filters.npy", np.stack(filters))
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), run_dir / "model.pt")
                status.update(best_epoch=epoch, best_val_acc=val_acc)
            print(f"epoch {epoch}/{args.epochs}  train_loss {run_loss / run_n:.4f}  "
                  f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}", flush=True)

        status["state"] = "done"
    except KeyboardInterrupt:
        status["state"] = "stopped"
    except Exception as e:
        status.update(state="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        # Score the best checkpoint on the held-out test set, even for a run
        # that was stopped early, as long as it finished at least one epoch.
        if status["state"] != "failed" and (run_dir / "model.pt").exists():
            model.load_state_dict(torch.load(run_dir / "model.pt"))
            test_loss, test_acc, test_probs = evaluate(model, x_te, y_te)
            np.savez_compressed(run_dir / "test.npz", y=y_te.numpy(),
                                pred=test_probs.argmax(1).numpy(),
                                probs=test_probs.numpy().astype(np.float16))
            status.update(test_loss=test_loss, test_acc=test_acc)
            print(f"test_acc {test_acc:.4f} (epoch {status.get('best_epoch')} checkpoint)")
        status["finished"] = time.time()
        write_json(run_dir / "status.json", status)
        metrics.close()
        stop_file.unlink(missing_ok=True)
    return run_dir


if __name__ == "__main__":
    print(train(parse_args()))
