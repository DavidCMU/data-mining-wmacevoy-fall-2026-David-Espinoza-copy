"""
Streamlit dashboard — start training runs, watch them live, inspect the
test set and compare bias-mitigation strategies. Mirrors ml/app.py's
architecture: this process never imports torch or train.py directly
(train.py runs as its own subprocess, exactly like the template's
description of its own app.py); everything here is read from runs/<id>/
files via runs.py, which is itself torch-free. That split is what lets the
dashboard start instantly and keep working even while a run is training.
"""
from __future__ import annotations

import subprocess
import sys
import uuid
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import streamlit as st

import bias_mitigation
import runs
from choices import MODEL_CHOICES

st.set_page_config(page_title="Income Bias Mitigation", layout="wide")
st.title("Adult Income — Measuring and Mitigating Sex Bias")

st.markdown(
    """
**The question:** a classifier trained to predict income (>$50K/year) from
the UCI Adult census data learns to flag qualified women as high earners
far less often than equally qualified men. This dashboard trains that
classifier under three settings — no correction, oversampling, and
reweighting (see `bias_mitigation.py`) — and measures whether each one
actually closes the gap, and what (if anything) it costs in accuracy.
"""
)


def _generate_run_id(model: str, mitigation: str) -> str:
    # Mirrors train.py's own `_run_id_for` for the no-explicit-id case, so
    # the id we pass via --run-id looks identical to one train.py would
    # have generated itself.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{model}-{mitigation}-{ts}-{uuid.uuid4().hex[:6]}"


# ========== Sidebar: launch a new run ==========
st.sidebar.header("Start a new run")
model_choice = st.sidebar.selectbox("Model", MODEL_CHOICES, index=MODEL_CHOICES.index("mlp"))
mitigation_choice = st.sidebar.selectbox("Bias mitigation", bias_mitigation.CHOICES, index=0)
epochs = st.sidebar.slider("Epochs", 1, 50, 20)
batch_size = st.sidebar.select_slider("Batch size", options=[32, 64, 128, 256, 512], value=128)
lr = st.sidebar.select_slider("Learning rate", options=[0.01, 0.003, 0.001, 0.0003, 0.0001], value=0.001)
seed = st.sidebar.number_input("Seed", min_value=0, max_value=10_000, value=0, step=1)

if st.sidebar.button("Start run", type="primary"):
    new_run_id = _generate_run_id(model_choice, mitigation_choice)
    cmd = [
        sys.executable,
        "train.py",
        "--model", model_choice,
        "--bias-mitigation", mitigation_choice,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--seed", str(seed),
        "--run-id", new_run_id,
    ]
    # Detached subprocess: training keeps going even if this Streamlit
    # session reloads or the browser tab is closed, exactly the property
    # the README promises ("start a run from the command line and watch it
    # in the browser" implies the reverse holds too).
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    st.session_state["selected_run_id"] = new_run_id
    st.sidebar.success(f"Started {new_run_id}")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "First run downloads the Adult dataset (~3-4 MB) from OpenML and caches it "
    "to data/adult.npz — later runs reuse the cache."
)

# ========== Run picker ==========
run_ids = runs.list_run_ids()
if not run_ids:
    st.info("No runs yet — configure one in the sidebar and click **Start run**.")
    st.stop()

default_idx = 0
if "selected_run_id" in st.session_state and st.session_state["selected_run_id"] in run_ids:
    default_idx = run_ids.index(st.session_state["selected_run_id"])

selected_run_id = st.selectbox("Run", run_ids, index=default_idx)
st.session_state["selected_run_id"] = selected_run_id

tab_progress, tab_test, tab_compare = st.tabs(["Training progress", "Test set", "Compare runs"])


@st.fragment(run_every=2)
def _live_status(run_id: str) -> None:
    """Auto-refreshing block (every 2s) — only meaningfully "live" while
    the run is still `running`; otherwise it just redraws the same final
    state, which is harmless."""
    run = runs.read_run(run_id)
    status = run["status"]
    state = status.get("state", "unknown")

    cols = st.columns(4)
    cols[0].metric("State", state)
    cols[1].metric("Epoch", f"{status.get('epoch', '?')}/{status.get('epochs', '?')}")
    val_acc = status.get("val_accuracy") or status.get("best_val_accuracy")
    cols[2].metric("Val accuracy", f"{val_acc:.1%}" if val_acc is not None else "—")
    tpr_gap = status.get("val_tpr_gap") or status.get("test_tpr_gap")
    cols[3].metric("TPR gap (M−F)", f"{tpr_gap:+.1%}" if tpr_gap is not None else "—")

    if state == "failed":
        st.error("This run failed. Error:")
        st.code(status.get("error", "(no traceback recorded)"))
        return

    metrics_df = run["metrics"]
    if not metrics_df.empty:
        loss_chart = (
            alt.Chart(metrics_df)
            .mark_line(color="#1f77b4")
            .encode(x=alt.X("epoch:Q", title="Epoch"), y=alt.Y("train_loss:Q", title="Train loss"))
            .properties(height=220, title="Training loss")
        )
        st.altair_chart(loss_chart, use_container_width=True)

        fairness_long = metrics_df.melt(
            id_vars=["epoch"],
            value_vars=["val_male_recall", "val_female_recall"],
            var_name="group",
            value_name="recall",
        )
        fairness_long["group"] = fairness_long["group"].map(
            {"val_male_recall": "male", "val_female_recall": "female"}
        )
        recall_chart = (
            alt.Chart(fairness_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("epoch:Q", title="Epoch"),
                y=alt.Y("recall:Q", title="Validation recall", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("group:N", title="Group"),
            )
            .properties(height=260, title="Recall by sex, over training (the gap this project targets)")
        )
        st.altair_chart(recall_chart, use_container_width=True)


with tab_progress:
    run = runs.read_run(selected_run_id)
    st.json(run["config"].get("mitigation_info", {}), expanded=False)
    _live_status(selected_run_id)

with tab_test:
    run = runs.read_run(selected_run_id)
    if run["status"].get("state") != "done":
        st.info("This run hasn't finished yet — the test set is only scored once, at the end.")
    else:
        report = run["test_report"]
        if report is None:
            st.warning("test_report.json not found for this run.")
        else:
            st.subheader("Test-set fairness report")
            report_df = pd.DataFrame(
                {
                    "overall": report["overall"],
                    "male": report["male"],
                    "female": report["female"],
                }
            ).T
            st.dataframe(report_df, use_container_width=True)
            c1, c2 = st.columns(2)
            c1.metric("TPR gap (male − female recall)", f"{report['tpr_gap']:+.1%}")
            c2.metric("Selection-rate gap", f"{report['selection_rate_gap']:+.1%}")
            st.caption(
                "TPR gap near 0 means men and women who actually earn >$50K are "
                "flagged at roughly the same rate — the goal of this project's mitigations."
            )

with tab_compare:
    st.subheader("Every finished run, side by side")
    table = runs.comparison_table()
    if table.empty:
        st.info("No finished runs yet.")
    else:
        display_table = table.copy()
        for col in ["test_accuracy", "male_recall", "female_recall", "tpr_gap", "selection_rate_gap"]:
            display_table[col] = display_table[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
        st.dataframe(display_table, use_container_width=True)
        st.caption(
            "Compare `tpr_gap` across bias_mitigation settings (none / oversample / reweight) "
            "at matched model/epochs/seed to see whether a mitigation actually helped, and "
            "`test_accuracy` to see what it cost."
        )
