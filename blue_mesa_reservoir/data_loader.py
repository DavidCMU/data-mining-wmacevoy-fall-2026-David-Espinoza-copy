"""
data_loader.py — Blue Mesa Reservoir storage vs. Colorado avg. temperature

Both inputs here are static CSV downloads (not live APIs), so this module is
much simpler than colorado_river/usgs.py: no HTTP client, no Parquet cache —
just "read the file, shape it into a tidy monthly DataFrame."

Two source files (see data/raw/):
- blue_mesa_storage.csv — RISE (Reclamation Information Sharing Environment)
  export. Daily reservoir storage in acre-feet (af) for Blue Mesa Reservoir.
  The file has a handful of metadata rows before the real header row
  ("Location","Parameter","Result",...), so we skip down to it explicitly
  instead of guessing a fixed row count.
- co_avg_temp.csv — Monthly average Colorado temperature (°F). Two comment
  lines (starting with "#") then a normal "Date,Value" header, where Date is
  an integer like 202301 meaning January 2023.

Both are resampled/parsed to a monthly period (`YYYY-MM`) so they can be
joined and compared on the same time scale.
"""
from __future__ import annotations

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
STORAGE_CSV = os.path.join(DATA_DIR, "blue_mesa_storage.csv")
TEMP_CSV = os.path.join(DATA_DIR, "co_avg_temp.csv")

# Where exported (cleaned) data lands. Kept separate from data/raw/ so it's
# obvious which files are the original downloads and which are generated.
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
PARQUET_FILENAME = "blue_mesa_monthly.parquet"
EXCEL_FILENAME = "blue_mesa_monthly.xlsx"


def load_reservoir_storage(path: str = STORAGE_CSV) -> pd.DataFrame:
    """Load the RISE daily reservoir-storage export into a tidy daily frame.

    Returns a DataFrame indexed by date with one column: storage_af.
    """
    # Find the real header row (starts with "Location") rather than assuming
    # a fixed skiprows count — RISE's metadata block has changed length
    # between exports before.
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header_idx = next(
        i for i, line in enumerate(lines) if line.startswith('"Location","Parameter"')
    )

    df = pd.read_csv(path, skiprows=header_idx)
    df = df.rename(columns={"Result": "storage_af", "Datetime (UTC)": "datetime"})
    df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df = df[["date", "storage_af"]].dropna().sort_values("date")
    df = df.set_index("date")
    return df


def load_temperature(path: str = TEMP_CSV) -> pd.DataFrame:
    """Load the monthly average-temperature CSV into a tidy monthly frame.

    Returns a DataFrame indexed by month-start date with one column: temp_f.
    The source `Date` column is an integer like 202301 (YYYYMM).
    """
    df = pd.read_csv(path, comment="#")
    df["date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m")
    df = df.rename(columns={"Value": "temp_f"})
    df = df[["date", "temp_f"]].dropna().sort_values("date")
    df = df.set_index("date")
    return df


def storage_to_monthly(daily_storage: pd.DataFrame) -> pd.DataFrame:
    """Resample daily storage to monthly mean, dropping any incomplete month.

    A month is "incomplete" if the data does not cover essentially the whole
    month (fewer than 25 daily readings) — this matters for whatever month a
    fresh RISE export was downloaded mid-month, which would otherwise pull
    the last point's average down/up misleadingly.
    """
    counts = daily_storage["storage_af"].resample("MS").count()
    monthly = daily_storage["storage_af"].resample("MS").mean().to_frame("storage_af")
    complete = counts >= 25
    return monthly[complete]


def load_monthly_merged() -> pd.DataFrame:
    """Load both sources and return one monthly-indexed DataFrame.

    Columns: storage_af (mean monthly reservoir storage, acre-feet),
    temp_f (mean monthly Colorado temperature, °F). Only months present in
    both series are kept (inner join).
    """
    storage_monthly = storage_to_monthly(load_reservoir_storage())
    temp_monthly = load_temperature()
    merged = storage_monthly.join(temp_monthly, how="inner")
    merged.index.name = "month"
    return merged


def export_cleaned_data(output_dir: str = PROCESSED_DIR) -> dict[str, str]:
    """Write the cleaned, monthly-merged dataset to disk as Parquet + Excel.

    This is the "cleaned data" for the assignment: the same monthly,
    inner-joined table the app charts and correlates (see
    `load_monthly_merged`) — not the raw daily/monthly source files, and not
    any one-off filtered slice. Writing it out is what makes the cleaning
    step (parsing two differently-shaped CSVs, resampling to a common
    monthly scale, dropping incomplete months) reusable outside this app.

    Returns a dict mapping format name -> the file path written, e.g.
    {"parquet": ".../blue_mesa_monthly.parquet", "excel": ".../blue_mesa_monthly.xlsx"}.
    """
    os.makedirs(output_dir, exist_ok=True)
    merged = load_monthly_merged().reset_index()

    parquet_path = os.path.join(output_dir, PARQUET_FILENAME)
    excel_path = os.path.join(output_dir, EXCEL_FILENAME)

    merged.to_parquet(parquet_path, index=False)
    merged.to_excel(excel_path, index=False, sheet_name="monthly_merged")

    return {"parquet": parquet_path, "excel": excel_path}
