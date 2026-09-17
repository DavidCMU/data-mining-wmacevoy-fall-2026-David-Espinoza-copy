import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import (  # noqa: E402
    export_cleaned_data,
    load_monthly_merged,
    load_reservoir_storage,
    load_temperature,
    storage_to_monthly,
)


def test_load_reservoir_storage_is_daily_and_positive():
    df = load_reservoir_storage()
    assert not df.empty
    assert "storage_af" in df.columns
    assert (df["storage_af"] > 0).all()
    assert df.index.is_monotonic_increasing


def test_load_temperature_is_monthly():
    df = load_temperature()
    assert not df.empty
    assert "temp_f" in df.columns
    # Monthly data should land on the first of the month.
    assert (df.index.day == 1).all()


def test_storage_to_monthly_drops_incomplete_months():
    daily = load_reservoir_storage()
    monthly = storage_to_monthly(daily)
    # The most recent month in this export only has a handful of days, so it
    # should be dropped rather than averaged from a partial sample.
    last_daily_month = daily.index.max().to_period("M")
    last_monthly = monthly.index.max().to_period("M")
    assert last_monthly <= last_daily_month


def test_load_monthly_merged_joins_on_overlap():
    merged = load_monthly_merged()
    assert not merged.empty
    assert list(merged.columns) == ["storage_af", "temp_f"]
    assert merged.index.is_monotonic_increasing
    assert merged.isna().sum().sum() == 0


def test_export_cleaned_data_writes_parquet_and_excel(tmp_path):
    paths = export_cleaned_data(output_dir=str(tmp_path))
    assert set(paths) == {"parquet", "excel"}
    for path in paths.values():
        assert os.path.exists(path)

    merged = load_monthly_merged().reset_index()
    parquet_df = pd.read_parquet(paths["parquet"])
    excel_df = pd.read_excel(paths["excel"])

    assert list(parquet_df.columns) == list(merged.columns)
    assert len(parquet_df) == len(merged)
    assert len(excel_df) == len(merged)
