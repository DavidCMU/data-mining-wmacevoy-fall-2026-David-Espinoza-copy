# MNIST Training Pipeline

Train a small neural network to read handwritten digits, and watch it learn.

```
mnist_data.py   download → parse IDX → cache data/mnist.npz → split, standardize
model.py        cnn (two conv layers) | mlp (fully connected)
train.py        train one model, writing everything to runs/<id>/
app.py          Streamlit: start runs, watch them live, inspect the test set
runs.py         reads runs/<id>/ for the app (no torch)
```

Training runs as its own process and only talks to the dashboard through
files, so you can reload the page, close it, or start a run from the command
line and watch it in the browser.

## Running it

From the **repository root**:

```bash
pixi run -e ml app
```

| Task | What it does |
| --- | --- |
| `pixi run -e ml app` | Launch the dashboard |
| `pixi run -e ml data` | Download MNIST into `data/mnist.npz` (the first run does this anyway) |
| `pixi run -e ml train` | Train from the command line (`--help` for the options) |
| `pixi run -e ml test` | Run the tests (offline, a few seconds) |
| `pixi run -e ml lint` | Check code style |
| `pixi run -e ml clean` | Delete `data/` and `runs/` |

## What a run records

| File | Contents |
| --- | --- |
| `config.json` | the arguments, so the run can be reproduced |
| `status.json` | `running` / `done` / `stopped` / `failed`, sizes, best epoch, test score |
| `metrics.jsonl` | one line per logged batch, per epoch, and per prediction snapshot |
| `filters.npy` | first-layer weights after every epoch |
| `model.pt` | weights from the epoch with the best **validation** accuracy |
| `test.npz` | that checkpoint's predictions on the 10,000 test images |

The 60,000 training images are split once (seeded) into 55,000 for training
and 5,000 for validation. The 10,000 test images are scored once, at the end.
Picking the checkpoint by validation accuracy means the test score is still
an honest estimate.

## Things to try

- **cnn vs. mlp** at the same epochs. The CNN has fewer weights and makes
  about half the mistakes.
- **1,000 training images, dropout 0**: training accuracy reaches 100% and
  validation stalls. That gap is overfitting.
- **sgd, learning rate 0.1** vs. **0.001**: one diverges or bounces, and the
  other barely moves in 5 epochs.
- The **Most confident mistakes** gallery: many of them would fool a person too.
