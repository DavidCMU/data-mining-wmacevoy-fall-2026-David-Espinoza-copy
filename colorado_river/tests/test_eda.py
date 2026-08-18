import pandas as pd

from eda import daily_features, rolling_anoms


def test_daily_features_and_anoms():
    # Build a tiny IV dataframe at 15-min cadence
    idx = pd.date_range("2025-01-01 00:00", periods=96, freq="15min", tz="UTC")
    df_iv = pd.DataFrame({
        "discharge_cfs": 1000 + (pd.Series(range(len(idx))) % 10).astype(float),
    }, index=idx)

    feats = daily_features(df_iv)
    assert not feats.empty
    assert any(c.startswith("discharge_cfs_") for c in feats.columns)
    assert feats.index.name == "date"

    anoms = rolling_anoms(feats, window=7)
    assert anoms.shape[0] == feats.shape[0]

