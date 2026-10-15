"""
MNIST training dashboard (Streamlit)

What this app does:
- Starts a training run (train.py, as its own process) from the sidebar.
- Watches it learn, live: loss and accuracy per batch and per epoch, the
  predictions on 32 fixed validation digits, and the first layer's weights.
- When a run finishes: its test-set confusion matrix, per-digit accuracy, and
  the mistakes it was most sure of.
- Compares validation accuracy across runs.

The pipeline in one line: mnist_data (download, split, standardize) ->
model (cnn | mlp) -> train (writes runs/<id>/) -> this app (reads runs/<id>/).
"""
import shutil

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import mnist_data
import runs
from train import N_SAMPLES

st.set_page_config(page_title="MNIST training", layout="wide")
st.title("MNIST training")

# Reference categorical palette, fixed order (blue, orange, aqua, ...).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
TRAIN, VAL = PALETTE[0], PALETTE[1]
SERIES = alt.Scale(domain=["train", "validation"], range=[TRAIN, VAL])


@st.cache_data
def validation_images(val_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The same validation split train.py used, so sample i means the same digit."""
    d = mnist_data.load()
    _, (x_va, y_va) = mnist_data.split(d["x_train"], d["y_train"], val_size, seed)
    return x_va, y_va


@st.cache_data
def test_images() -> np.ndarray:
    return mnist_data.load()["x_test"]


# ========== Sidebar: start a run ==========
with st.sidebar:
    st.header("New run")
    st.caption("Pick settings and start training. Hover the ? on each one for what it does.")
    with st.form("new_run"):
        arch = st.radio("Network", ["cnn", "mlp"], horizontal=True,
                        help="cnn: two conv layers that look at small patches of the image, then a "
                             "classifier. mlp: fully connected, every pixel wired to every "
                             "neuron, no convolutions. The cnn usually wins on images.")
        epochs = st.slider("Epochs", 1, 20, 5,
                           help="How many full passes over the training images. More epochs "
                                "learn more, until validation accuracy stops improving and "
                                "the model starts memorizing (overfitting).")
        optimizer = st.radio("Optimizer", ["adam", "sgd"], horizontal=True,
                             help="How the weights are updated from each batch's gradient. "
                                  "adam: adapts the step size per weight, forgiving of the "
                                  "learning rate. sgd: plain gradient descent with momentum "
                                  "0.9, usually wants a larger learning rate.")
        lr = st.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
                              value=1e-3, format_func=lambda v: f"{v:g}",
                              help="How big a step each update takes. Too small learns "
                                   "slowly; too large makes the loss jump around or blow up. "
                                   "Try 0.001 for adam, 0.01-0.1 for sgd.")
        batch_size = st.select_slider("Batch size", [16, 32, 64, 128, 256, 512], value=128,
                                      help="Images per weight update. Small batches give more, "
                                           "noisier updates per epoch; large batches give fewer, "
                                           "smoother ones and run faster per image.")
        dropout = st.slider("Dropout", 0.0, 0.7, 0.2, 0.05,
                            help="Fraction of hidden neurons randomly switched off on each "
                                 "training batch, so the network can't lean on any one of "
                                 "them. Fights overfitting; 0 turns it off. Off at test time.")
        train_size = st.select_slider(
            "Training images", [1000, 5000, 10000, 25000, 55000], value=55000,
            help="How many of the 55,000 training images to use (the other 5,000 are "
                 "held out for validation). Fewer images train faster and make "
                 "overfitting easy to see.")
        if st.form_submit_button("Start training", type="primary"):
            new = runs.start({
                "arch": arch, "epochs": epochs, "optimizer": optimizer, "lr": lr,
                "batch_size": batch_size, "dropout": dropout,
                "train_size": 0 if train_size == 55000 else train_size,
            })
            st.session_state["run"] = new.name
    st.caption("Training runs in the background; it shows up under Runs below.")

    all_runs = runs.list_runs()
    if not all_runs:
        st.info("No runs yet. Start one above, or `pixi run -e ml train`.")
        st.stop()

    st.header("Runs")
    st.caption("Every run, newest first: its id, its state (running, done, stopped, "
               "failed, or lost if it stopped reporting), and its test accuracy once "
               "finished. Pick one to view it on the right.")
    names = [p.name for p in all_runs]
    if st.session_state.get("run") not in names:
        st.session_state["run"] = names[0]

    def label(name: str) -> str:
        p = runs.RUNS_DIR / name
        s = runs.read_json(p / "status.json", {})
        acc = f" · test {s['test_acc']:.2%}" if "test_acc" in s else ""
        return f"{name} · {runs.state(p)}{acc}"

    st.radio("Show", names, key="run", format_func=label, label_visibility="collapsed")


run_dir = runs.RUNS_DIR / st.session_state["run"]
config = runs.read_json(run_dir / "config.json", {})
running = runs.state(run_dir) == "running"


# ========== Charts ==========
def curve(batch: pd.DataFrame, epoch: pd.DataFrame, metric: str, title: str) -> alt.Chart:
    """Training per batch (thin, noisy) under validation per epoch (line + points)."""
    parts = []
    if not batch.empty:
        parts.append(pd.DataFrame({"epoch": batch["epoch"], "value": batch[metric],
                                   "series": "train"}))
    v = epoch.dropna(subset=[f"val_{metric}"]) if not epoch.empty else epoch
    if not v.empty:
        parts.append(pd.DataFrame({"epoch": v["epoch"], "value": v[f"val_{metric}"],
                                   "series": "validation"}))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["epoch", "value", "series"])

    y = (alt.Y("value:Q", title=title, scale=alt.Scale(type="log"))
         if metric == "loss" else
         alt.Y("value:Q", title=title, scale=alt.Scale(zero=False), axis=alt.Axis(format="%")))
    fmt = ".4f" if metric == "loss" else ".2%"
    base = alt.Chart(df).encode(
        x=alt.X("epoch:Q", title="epoch", scale=alt.Scale(domainMin=0)),
        y=y,
        color=alt.Color("series:N", scale=SERIES, legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("series:N"), alt.Tooltip("epoch:Q", format=".2f"),
                 alt.Tooltip("value:Q", title=title, format=fmt)],
    )
    train_line = base.transform_filter(alt.datum.series == "train").mark_line(
        strokeWidth=1.5, opacity=0.6)
    val_line = base.transform_filter(alt.datum.series == "validation").mark_line(
        strokeWidth=2, point=alt.OverlayMarkDef(size=80, filled=True))
    return (train_line + val_line).properties(height=280)


def sample_grid(x: np.ndarray, y: np.ndarray, snap: dict, cols: int = 8) -> None:
    for start in range(0, N_SAMPLES, cols):
        for col, i in zip(st.columns(cols), range(start, start + cols)):
            pred, conf = snap["pred"][i], snap["conf"][i]
            mark = "✓" if pred == y[i] else f"✗ (is {y[i]})"
            col.image(x[i], width=56)
            col.caption(f"**{pred}** {mark} · {conf:.0%}")


def confusion_chart(y: np.ndarray, pred: np.ndarray) -> alt.Chart:
    cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="predicted"))
    cm = cm.reindex(index=range(10), columns=range(10), fill_value=0)
    df = cm.stack().rename("count").reset_index()
    df["off"] = np.where(df["true"] == df["predicted"], 0, df["count"])
    base = alt.Chart(df).encode(
        x=alt.X("predicted:O", title="predicted"),
        y=alt.Y("true:O", title="true digit"),
        tooltip=["true:O", "predicted:O", "count:Q"],
    )
    # Colour only the mistakes: the diagonal (~1000 each) would otherwise wash
    # every off-diagonal cell to the same pale blue.
    cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color("off:Q", title="mistakes",
                        scale=alt.Scale(range=["#f4f8fd", "#104281"])))
    text = base.mark_text(fontSize=11).encode(
        text="count:Q",
        color=alt.condition(alt.datum.off > df["off"].max() / 2,
                            alt.value("white"), alt.value("#52514e")))
    return (cells + text).properties(height=360)


# ========== The selected run ==========
def show_run() -> None:
    status = runs.read_json(run_dir / "status.json", {})
    st_now = runs.state(run_dir)
    if st.session_state.get("_seen_state") not in (None, st_now) and st_now != "running":
        st.session_state["_seen_state"] = st_now
        st.rerun()                     # finished: redraw the whole page without polling
    st.session_state["_seen_state"] = st_now

    batch, epoch, samples = runs.frames(runs.read_metrics(run_dir))
    done_epochs = int(epoch["epoch"].max()) if not epoch.empty else 0

    c = st.columns(5)
    c[0].metric("State", st_now)
    c[1].metric("Epoch", f"{done_epochs} / {config.get('epochs', '?')}")
    c[2].metric("Validation acc", f"{epoch['val_acc'].iloc[-1]:.2%}" if not epoch.empty else "–")
    c[3].metric("Test acc", f"{status['test_acc']:.2%}" if "test_acc" in status else "–",
                help="Scored once, at the end, with the best-validation checkpoint.")
    c[4].metric("Weights", f"{status['n_params']:,}" if "n_params" in status else "–")
    st.caption(
        f"{config.get('arch')} · {config.get('optimizer')} lr {config.get('lr')} · "
        f"batch {config.get('batch_size')} · dropout {config.get('dropout')} · "
        f"{status.get('n_train', '?'):,} training images"
        if "n_train" in status else "starting…")
    if st_now == "failed":
        st.error(status.get("error", "failed"))
    if st_now == "lost":
        st.warning("This run stopped reporting without finishing (killed, or the machine slept).")
    if st_now in ("failed", "lost") and (run_dir / "stdout.log").exists():
        with st.expander("stdout.log"):
            st.code((run_dir / "stdout.log").read_text()[-4000:])

    left, right = st.columns(2)
    left.altair_chart(curve(batch, epoch, "loss", "cross-entropy loss"),
                      width="stretch")
    right.altair_chart(curve(batch, epoch, "acc", "accuracy"), width="stretch")
    if done_epochs >= 2 and "train_acc" in epoch:
        gap = epoch["train_acc"].iloc[-1] - epoch["val_acc"].iloc[-1]
        if gap > 0.02:
            st.caption(f"Training accuracy is {gap:.1%} above validation: the network is "
                       "starting to memorize its training images (overfitting).")

    if samples:
        st.subheader("32 validation digits, as the network sees them")
        epoch_pick = (st.select_slider("After epoch", [s["epoch"] for s in samples],
                                       value=samples[-1]["epoch"], key="sample_epoch")
                      if len(samples) > 1 else 0)
        snap = next(s for s in samples if s["epoch"] == epoch_pick)
        x_va, y_va = validation_images(config["val_size"], config["seed"])
        right_n = sum(p == t for p, t in zip(snap["pred"], y_va[:N_SAMPLES]))
        st.caption(f"Epoch {epoch_pick}: {right_n}/{N_SAMPLES} right. "
                   "Each caption is the prediction, ✓ or ✗, and the network's confidence.")
        sample_grid(x_va, y_va, snap)

    filters_path = run_dir / "filters.npy"
    if filters_path.exists():
        st.subheader("What the first layer looks for")
        filters = np.load(filters_path)
        k = st.select_slider("After epoch ", list(range(len(filters))), value=len(filters) - 1,
                             key="filter_epoch") if len(filters) > 1 else 0
        f = filters[k]
        if f.shape[-1] == 3:
            st.caption("The 16 3×3 kernels of the first convolution, light = positive weight. "
                       "Many become edge and stroke detectors.")
            st.image(runs.montage(f, cols=16, scale=16), width=16 * 4 * 16)
        else:
            st.caption("Each hidden unit has one weight per pixel, so its weights form a 28×28 "
                       "picture (first 64 of 256). Light = this pixel pushes the unit up.")
            st.image(runs.montage(f[:64], cols=16, scale=3), width=16 * 29 * 3)


live = st.fragment(run_every=2 if running else None)(show_run)

tab_run, tab_test, tab_compare = st.tabs(["Training", "Test set", "Compare runs"])
with tab_run:
    if running and st.button("Stop this run"):
        runs.stop(run_dir)
        st.toast("Stopping after the current step…")
    live()

with tab_test:
    test_path = run_dir / "test.npz"
    if not test_path.exists():
        st.info("The test set is scored once, when the run ends.")
    else:
        t = np.load(test_path)
        y, pred, probs = t["y"], t["pred"], t["probs"].astype(np.float32)
        wrong = np.flatnonzero(pred != y)
        st.markdown(f"**{len(y) - len(wrong):,} of {len(y):,}** test digits right "
                    f"({1 - len(wrong) / len(y):.2%}); {len(wrong)} mistakes.")
        a, b = st.columns([3, 2])
        with a:
            st.altair_chart(confusion_chart(y, pred), width="stretch")
        with b:
            per = pd.DataFrame({"digit": range(10),
                                "accuracy": [(pred[y == d] == d).mean() for d in range(10)]})
            st.altair_chart(
                alt.Chart(per).mark_bar(color=TRAIN, cornerRadiusEnd=4).encode(
                    x=alt.X("accuracy:Q", scale=alt.Scale(zero=False),
                            axis=alt.Axis(format="%")),
                    y=alt.Y("digit:O"),
                    tooltip=["digit:O", alt.Tooltip("accuracy:Q", format=".2%")],
                ).properties(height=360, title="Accuracy by true digit"),
                width="stretch")
        st.subheader("Most confident mistakes")
        st.caption("Wrong, and sure of it. Some are genuinely ambiguous; some are mislabeled.")
        x_te = test_images()
        worst = wrong[np.argsort(-probs[wrong].max(1))][:16]
        for row in (worst[:8], worst[8:]):
            for col, i in zip(st.columns(8), row):
                col.image(x_te[i], width=56)
                col.caption(f"said **{pred[i]}** ({probs[i].max():.0%}), is {y[i]}")

        st.subheader("Export")
        st.caption("ONNX is a framework-neutral model file: this run's best checkpoint, with "
                   "standardization and softmax built in, runnable without Python or PyTorch. "
                   "The export is checked against PyTorch on 256 test images.")
        onnx_path = run_dir / "model.onnx"
        if st.button("Export to ONNX", type="primary"):
            # Pulls in torch and onnx, so load it only when asked.
            import export_onnx

            with st.spinner("Exporting…"):
                try:
                    info = export_onnx.export(run_dir)
                    st.session_state["onnx_info"] = (run_dir.name, info)
                except Exception as e:     # show any exporter failure, don't crash the page
                    st.error(f"Export failed: {type(e).__name__}: {e}")
        info = st.session_state.get("onnx_info")
        if info and info[0] == run_dir.name:
            st.success(f"Exported {info[1]['bytes'] / 1024:,.0f} KB. Largest difference from "
                       f"PyTorch: {info[1]['max_diff']:.1e}.")
        if onnx_path.exists():
            a, b = st.columns(2)
            a.download_button("Download model.onnx", onnx_path.read_bytes(),
                              file_name=f"mnist-{run_dir.name}.onnx",
                              mime="application/octet-stream")
            if b.button("Use in the web app"):
                web_model = runs.HERE / "web" / "model.onnx"
                shutil.copyfile(onnx_path, web_model)
                b.success("Copied to web/model.onnx.")
            st.caption("Then `pixi run -e ml web` and open http://localhost:8000 to draw digits "
                       "for it.")

with tab_compare:
    rows = []
    for p in all_runs[:len(PALETTE)]:      # one fixed colour per run, never cycled
        _, e, _ = runs.frames(runs.read_metrics(p))
        if not e.empty:
            rows.append(e.assign(run=p.name)[["run", "epoch", "val_acc"]])
    if rows:
        df = pd.concat(rows)
        order = sorted(df["run"].unique())
        st.altair_chart(
            alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64)).encode(
                x=alt.X("epoch:Q"),
                y=alt.Y("val_acc:Q", title="validation accuracy", scale=alt.Scale(zero=False),
                        axis=alt.Axis(format="%")),
                color=alt.Color("run:N", scale=alt.Scale(domain=order,
                                                         range=PALETTE[:len(order)]),
                                legend=alt.Legend(orient="bottom", columns=2)),
                tooltip=["run:N", "epoch:Q", alt.Tooltip("val_acc:Q", format=".2%")],
            ).properties(height=380),
            width="stretch")
        table = []
        for p in all_runs[:len(PALETTE)]:
            c, s = runs.read_json(p / "config.json", {}), runs.read_json(p / "status.json", {})
            table.append({"run": p.name, "state": runs.state(p), "arch": c.get("arch"),
                          "optimizer": c.get("optimizer"), "lr": c.get("lr"),
                          "batch": c.get("batch_size"), "dropout": c.get("dropout"),
                          "train images": s.get("n_train"), "best val": s.get("best_val_acc"),
                          "test": s.get("test_acc")})
        st.dataframe(pd.DataFrame(table), hide_index=True, column_config={
            "best val": st.column_config.NumberColumn(format="percent"),
            "test": st.column_config.NumberColumn(format="percent")})
        if len(all_runs) > len(PALETTE):
            st.caption(f"Showing the {len(PALETTE)} newest of {len(all_runs)} runs.")
