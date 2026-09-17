"""
Streamlit app — Blue Mesa Reservoir storage vs. Colorado avg. temperature

The problem this explores:
  Does Blue Mesa Reservoir's storage level track statewide temperature on a
  monthly basis? Storage should mostly follow spring snowmelt timing and
  downstream release schedules, while temperature is a broad climate signal —
  so a naive "hotter month -> lower storage" story might not hold once you
  actually look at the numbers side by side.

Data (static CSV downloads, see data/raw/ and README.md for sources):
  - Blue Mesa Reservoir daily storage (acre-feet), from Reclamation's RISE.
  - Colorado statewide average temperature (°F), monthly.

Both series are put on the same monthly time scale (storage is resampled
from daily to monthly mean) so they can be compared directly.
"""
import os
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from data_loader import (
    PROCESSED_DIR,
    export_cleaned_data,
    load_monthly_merged,
    load_reservoir_storage,
    load_temperature,
)

st.set_page_config(page_title="Blue Mesa Storage vs. CO Temperature", layout="wide")
st.title("Blue Mesa Reservoir Storage vs. Colorado Avg. Temperature")

st.markdown(
    """
**The question:** does reservoir storage at Blue Mesa move together with
statewide average temperature, month to month?

Reservoir storage is recorded *daily*; temperature is only available
*monthly*. To compare them, storage is resampled to a monthly mean (any
month without near-complete daily coverage — such as a month a CSV was
downloaded partway through — is dropped rather than averaged from a partial
sample).
"""
)

merged = load_monthly_merged()

# ---- Monthly overview (each series on its own axis, since af and °F are on
# wildly different scales) ----
st.subheader("Monthly storage and temperature, side by side")

base = merged.reset_index()
storage_line = (
    alt.Chart(base)
    .mark_line(color="#1f77b4", point=True)
    .encode(
        x=alt.X("month:T", title="Month"),
        y=alt.Y("storage_af:Q", title="Storage (acre-feet)", axis=alt.Axis(titleColor="#1f77b4")),
    )
)
temp_line = (
    alt.Chart(base)
    .mark_line(color="#d62728", point=True)
    .encode(
        x=alt.X("month:T", title="Month"),
        y=alt.Y("temp_f:Q", title="Avg. Temp (°F)", axis=alt.Axis(titleColor="#d62728")),
    )
)
st.altair_chart(
    alt.layer(storage_line, temp_line).resolve_scale(y="independent").properties(height=320),
    use_container_width=True,
)
st.caption("Blue = storage (acre-feet, left axis). Red = avg. temperature (°F, right axis).")

# ---- Correlation ----
st.subheader("Do the two series correlate?")

corr = merged["storage_af"].corr(merged["temp_f"])
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Pearson correlation (r)", f"{corr:.3f}")
    st.caption(f"n = {len(merged)} overlapping months")
with col2:
    scatter = (
        alt.Chart(base)
        .mark_circle(size=80, opacity=0.7)
        .encode(
            x=alt.X("temp_f:Q", title="Avg. Temp (°F)"),
            y=alt.Y("storage_af:Q", title="Storage (acre-feet)"),
            tooltip=["month:T", "temp_f:Q", "storage_af:Q"],
        )
    )
    trend = scatter.transform_regression("temp_f", "storage_af").mark_line(color="black")
    st.altair_chart((scatter + trend).properties(height=320), use_container_width=True)

st.markdown(
    f"""
A correlation of **r = {corr:.2f}** is weak-to-moderate — temperature alone
does not explain most of the month-to-month change in storage. That fits the
underlying mechanism: storage is driven mainly by snowmelt inflow timing and
dam operations (a fixed seasonal shape that peaks in early summer), while
temperature is a more continuous, less seasonally-lagged signal. A stronger
relationship would likely show up if temperature were lagged by a month or
two, or compared against inflow/release volumes directly instead of storage
level.
"""
)

# ---- Export cleaned data ----
st.subheader("Export cleaned data")
st.markdown(
    """
The chart and table above are built from one cleaned, monthly-merged
table — daily storage resampled to monthly, incomplete months dropped, then
joined against monthly temperature. The button below writes that exact
table to disk as both **Parquet** (compact, keeps dtypes — the format
you'd hand to another pandas/Spark job) and **Excel** (the format you'd
hand to someone opening it by hand).
"""
)

if st.button("Export to Parquet + Excel"):
    exported_paths = export_cleaned_data()
    st.success(f"Exported {len(exported_paths)} files to `data/processed/`.")

if os.path.isdir(PROCESSED_DIR) and os.listdir(PROCESSED_DIR):
    st.markdown("**Files currently in `data/processed/`:**")
    rows = []
    for filename in sorted(os.listdir(PROCESSED_DIR)):
        full_path = os.path.join(PROCESSED_DIR, filename)
        rows.append(
            {
                "file": filename,
                "size_kb": round(os.path.getsize(full_path) / 1024, 1),
                "modified": datetime.fromtimestamp(os.path.getmtime(full_path)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    download_cols = st.columns(len(rows))
    for col, row in zip(download_cols, rows):
        full_path = os.path.join(PROCESSED_DIR, row["file"])
        with open(full_path, "rb") as f:
            col.download_button(f"Download {row['file']}", f.read(), file_name=row["file"])
else:
    st.info("No exported files yet — click the button above to create them.")

# ---- Raw-ish monthly table + underlying daily/monthly source data ----
with st.expander("Monthly merged data (what the charts above are built from)"):
    st.dataframe(merged.reset_index())

with st.expander("Source data — daily reservoir storage"):
    st.dataframe(load_reservoir_storage().reset_index().tail(30))

with st.expander("Source data — monthly avg. temperature"):
    st.dataframe(load_temperature().reset_index())

st.caption(
    "Sources: Bureau of Reclamation RISE (Blue Mesa Reservoir Dam and Powerplant, "
    "Lake/Reservoir Storage, daily) and a monthly Colorado statewide average "
    "temperature series. See README.md for details."
)
