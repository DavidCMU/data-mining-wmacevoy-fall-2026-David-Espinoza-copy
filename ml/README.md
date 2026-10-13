# MNIST Training Pipeline

Train a small neural network to read handwritten digits, and watch it learn.

```
mnist_data.py   download → parse IDX → cache data/mnist.npz → split, standardize
model.py        cnn (two conv layers) | mlp (fully connected)
train.py        train one model, writing everything to runs/<id>/
app.py          Streamlit: start runs, watch them live, inspect the test set
runs.py         reads runs/<id>/ for the app (no torch)
export_onnx.py  best checkpoint → model.onnx, verified against PyTorch
web/            draw a digit; the ONNX model reads it in the browser
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
| `pixi run -e ml export` | Export the newest finished run to ONNX and copy it to `web/model.onnx` |
| `pixi run -e ml web` | Serve `web/` at http://localhost:8000 |
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

## From model to app: ONNX

A `model.pt` file is only useful to PyTorch. **ONNX** is a framework-neutral
format for the same computation graph and weights, and runtimes exist for
browsers, phones, C++, C#, Java and more. `export_onnx.py` exports the
*deployable* model rather than the training one:

| | |
| --- | --- |
| input `pixels` | float32 `(batch, 1, 28, 28)`, 0 = background, 1 = ink |
| output `probs` | float32 `(batch, 10)`, each row sums to 1 |

Standardization (the MNIST mean and std) and softmax are part of the graph,
so whoever calls the model can't forget them or get them wrong. After it
writes the file, the exporter runs it through onnxruntime on 256 test images
and fails if any probability differs from PyTorch's by more than 1e-4.

Export from the dashboard (**Test set → Export**, then **Use in the web
app**) or with `pixi run -e ml export`, then:

```bash
pixi run -e ml web       # open http://localhost:8000
```

`web/index.html` is one static page. It loads `model.onnx` with
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) (WebAssembly)
and reads what you draw as you draw it. Nothing leaves the browser. **Load
.onnx…** tries any other export. The page prepares a drawing the way MNIST
prepared its digits: crop to the ink, scale the longer side to 20 px, then
center by center of mass in a 28×28 frame. The small black square shows the
result. Without that step, a well-drawn digit off in a corner looks like
nothing the network has seen.

`web/model.onnx` is git-ignored like the rest of `runs/`: it is built from a
run, not written by hand.

## Things to try

- **cnn vs. mlp** at the same epochs. The CNN has fewer weights and makes
  about half the mistakes.
- **1,000 training images, dropout 0**: training accuracy reaches 100% and
  validation stalls. That gap is overfitting.
- **sgd, learning rate 0.1** vs. **0.001**: one diverges or bounces, and the
  other barely moves in 5 epochs.
- The **Most confident mistakes** gallery: many of them would fool a person too.
- In the web app, draw a 1 with a serif and a base, a 7 with a crossbar, or a
  digit that fills only half the pad. Then compare an MLP export with a CNN export.
