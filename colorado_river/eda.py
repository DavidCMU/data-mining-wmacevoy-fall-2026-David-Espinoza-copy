"""
EDA helpers (beginner-friendly walkthrough)

In data mining with pandas, it's common to:
- Resample high-frequency data (like 15-min IV) into daily aggregates.
- Compute simple diagnostics (e.g., how often values are missing?).
- Derive features and simple anomaly scores to help spot unusual days.

Key tips:
- If your DataFrame is indexed by time, pandas can resample using strings
  like "1D" for one day, "15min" for fifteen minutes, etc.
- Timezones are tricky. To simplify math, we convert to UTC and drop tz where
  appropriate.
"""
import numpy as np
import pandas as pd

DENVER_TZ = "America/Denver"


def to_local(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with times converted to local wall-clock (America/Denver).

    Why do this?
    - USGS data come in UTC or with offsets. For human plots, local time is
      easier to read.
    - Streamlit/Arrow prefer timezone-naive datetimes, so after converting
      to local, we drop the timezone.
    """
    if df.index.tz is None:
        return df
    # Convert to local time then drop tz for easier plotting/Arrow compat
    return df.tz_convert(DENVER_TZ).tz_localize(None)


def summarize_gaps(df: pd.DataFrame) -> pd.Series:
    """Calculate simple sampling-gap metrics for 15-min data.

    Returns a pandas Series (a labeled 1D vector) with:
    - count: number of rows (samples)
    - start/end: first and last timestamps
    - median_step_sec: typical spacing between samples
    - max_gap_sec: largest spacing (big gaps indicate outages)
    - pct_missing_vs_15min: rough percent of missing samples compared to an ideal
      15-minute cadence over the covered time span
    """
    if df.empty:
        return pd.Series(dtype=float)
    idx = df.index
    # Normalize to naive UTC for delta math (ignore DST complexities here)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    deltas = idx.to_series().diff().dropna()
    ideal = (idx.max() - idx.min()).total_seconds() / (15 * 60)
    pct_missing = float(1 - (len(df) / ideal)) if ideal > 0 else np.nan
    return pd.Series({
        "count": float(len(df)),
        "start": df.index.min(),
        "end": df.index.max(),
        "median_step_sec": float(deltas.median().total_seconds()) if not deltas.empty else np.nan,
        "max_gap_sec": float(deltas.max().total_seconds()) if not deltas.empty else np.nan,
        "pct_missing_vs_15min": pct_missing,
    })


def daily_features(df_iv: pd.DataFrame) -> pd.DataFrame:
    """Resample IV (15-min) into daily features (mean, max, min).

    Output columns look like:
    - discharge_cfs_mean, discharge_cfs_max, discharge_cfs_min
    - stage_ft_mean, stage_ft_max, stage_ft_min (if stage is available)
    """
    if df_iv.empty:
        return df_iv
    # Resampling prefers timezone-naive timestamps; normalize gently
    base = df_iv
    if df_iv.index.tz is not None:
        base = df_iv.tz_convert("UTC").tz_localize(None)
    agg = {
        "discharge_cfs": ["mean", "max", "min"],
        "stage_ft": ["mean", "max", "min"],
    }
    avail = {k: v for k, v in agg.items() if k in base.columns}
    out = base.resample("1D").agg(avail)
    # Flatten MultiIndex columns to simple snake_case
    out.columns = ["_".join(filter(None, col)) for col in out.columns]
    # Ensure index is named 'date' for downstream consumers
    out.index.name = "date"
    return out


def rolling_anoms(daily: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Compute simple rolling z-scores for each numeric column.

    Concept: z = (x - mean) / std computed over a moving window.
    We use `min_periods=7` to avoid dividing by zero early in the series.
    """
    if daily.empty:
        return daily
    out = daily.copy()
    for col in out.columns:
        if out[col].dtype.kind not in "fc":
            continue
        mu = out[col].rolling(window, min_periods=7).mean()
        sigma = out[col].rolling(window, min_periods=7).std()
        out[f"{col}_z"] = (out[col] - mu) / sigma
    return out

