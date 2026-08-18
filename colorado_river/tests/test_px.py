import pandas as pd


def test_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=3, freq="D"),
        "value": [1.0, 2.0, 3.0],
    }).set_index("date")
    p = tmp_path / "sample.parquet"
    df.to_parquet(p)
    loaded = pd.read_parquet(p)
    assert list(loaded.columns) == ["value"]
    assert len(loaded) == 3


