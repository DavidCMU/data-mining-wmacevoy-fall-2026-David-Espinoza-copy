"""
EDA helpers for occupancy events.

An event is (location, ts, delta) with delta = +1 (someone entered) or -1
(someone left). Nothing stores "the count": occupancy at any moment is the
running sum of the deltas up to that moment. Everything here is built on that.

Times arrive in UTC. People think about rooms in local time ("busy at 10am on
Tuesdays"), so every "by hour" or "by day" question is asked in
America/Denver. Daylight-saving days have 23 or 25 hours; pandas handles that
as long as we convert with tz_convert and never by adding a fixed offset.
"""
import numpy as np
import pandas as pd

DENVER_TZ = "America/Denver"
MOCK_PREFIX = "mock-"


def local_naive(s: pd.Series) -> pd.Series:
    """UTC timestamps -> Denver wall-clock time, timezone dropped (Streamlit/Arrow prefer that)."""
    return s.dt.tz_convert(DENVER_TZ).dt.tz_localize(None)


def is_synthetic(location: str) -> bool:
    """Mock locations start with 'mock-'. Mixing them into real analysis is the classic mistake."""
    return str(location).startswith(MOCK_PREFIX)


def occupancy_series(events: pd.DataFrame) -> pd.DataFrame:
    """One location's events in time order with the running count.

    Sorted by ts, then row_id so that two events in the same second keep the
    order the server received them in. Columns: ts (UTC), delta, occupancy.
    """
    if events.empty:
        return pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns, UTC]"),
                             "delta": pd.Series(dtype="int64"),
                             "occupancy": pd.Series(dtype="int64")})
    ordered = events.sort_values(["ts", "row_id"])
    out = ordered[["ts", "delta"]].astype({"delta": "int64"}).reset_index(drop=True)
    out["occupancy"] = out["delta"].cumsum()
    return out


def location_summary(events: pd.DataFrame) -> pd.DataFrame:
    """One row per location: how much data, over what span, and whether it adds up.

    - count_now: the sum of all deltas, the current occupancy.
    - lowest_count: the lowest the running count ever went. Below zero means
      more people left than entered: the sensor missed some entries.
    - median_lag_s: median seconds from ts (when it happened) to received_at
      (when the server got it). Big lags mean store-and-forward at work.
    """
    rows = []
    for location, group in events.groupby("location", sort=True):
        series = occupancy_series(group)
        lag = (group["received_at"] - group["ts"]).dt.total_seconds()
        rows.append({
            "location": location,
            "synthetic": is_synthetic(location),
            "events": len(group),
            "enters": int((group["delta"] > 0).sum()),
            "leaves": int((group["delta"] < 0).sum()),
            "count_now": int(series["occupancy"].iloc[-1]),
            "lowest_count": int(series["occupancy"].min()),
            "first": group["ts"].min(),
            "last": group["ts"].max(),
            "median_lag_s": float(lag.median()),
        })
    return pd.DataFrame(rows)


def bucketed(events: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """One location's activity per local time bucket.

    Columns: enters, leaves, net, and occupancy (the running count at the end
    of the bucket; carried forward through empty buckets, because the count
    doesn't change when nobody moves).
    """
    series = occupancy_series(events)
    if series.empty:
        return pd.DataFrame(columns=["enters", "leaves", "net", "occupancy"])
    local = series.set_index(series["ts"].dt.tz_convert(DENVER_TZ))
    out = pd.DataFrame({
        "enters": (local["delta"] > 0).resample(freq).sum(),
        "leaves": (local["delta"] < 0).resample(freq).sum(),
        "net": local["delta"].resample(freq).sum(),
        "occupancy": local["occupancy"].resample(freq).last().ffill(),
    })
    out.index = out.index.tz_localize(None)
    out.index.name = "time"
    return out.astype({"enters": "int64", "leaves": "int64", "net": "int64"})


def weekday_hour_profile(events: pd.DataFrame) -> pd.DataFrame:
    """Average entries per (weekday, local hour), counting the hours when nobody came.

    Averaging only the hours that have events would overstate quiet times, so
    we first build every hour in the span (zeros included) and then average.
    Long format: weekday (0=Mon), weekday_name, hour, mean_enters.
    """
    hourly = bucketed(events, "1h")
    if hourly.empty:
        return pd.DataFrame(columns=["weekday", "weekday_name", "hour", "mean_enters"])
    frame = pd.DataFrame({
        "weekday": hourly.index.dayofweek,
        "weekday_name": hourly.index.day_name().str[:3],
        "hour": hourly.index.hour,
        "enters": hourly["enters"].to_numpy(),
    })
    profile = (frame.groupby(["weekday", "weekday_name", "hour"], as_index=False)["enters"]
               .mean().rename(columns={"enters": "mean_enters"}))
    return profile


def daily_drift(events: pd.DataFrame) -> pd.DataFrame:
    """Net change per local day, and the running count at the end of each day.

    A real room empties at night, so every day's net should be 0. A day that
    doesn't net to zero is a counting error (two people side by side, a missed
    exit); the end-of-day count drifting away from zero is those errors adding up.
    For a mock location the simulator decides, so this is a check on it instead.
    """
    daily = bucketed(events, "1D")
    if daily.empty:
        return pd.DataFrame(columns=["date", "net", "end_of_day"])
    return pd.DataFrame({
        "date": daily.index,
        "net": daily["net"].to_numpy(),
        "end_of_day": daily["occupancy"].to_numpy(),
    })


def upload_lag(events: pd.DataFrame) -> pd.DataFrame:
    """Seconds between when each event happened (ts) and when the server stored it.

    The mock posts right away (lag ~ 0, or negative if the browser clock is
    ahead). The Pi uploads every 2 minutes, and after a network outage events
    arrive hours late. A backfill has lags of days. Same table, different stories.
    """
    if events.empty:
        return pd.DataFrame(columns=["location", "ts", "lag_s"])
    return pd.DataFrame({
        "location": events["location"].to_numpy(),
        "ts": events["ts"].to_numpy(),
        "lag_s": (events["received_at"] - events["ts"]).dt.total_seconds().to_numpy(),
    })


def late_arrivals(events: pd.DataFrame) -> int:
    """How many events reached the server after a later-timestamped event did.

    Any analysis that assumes rows arrive in time order is wrong by this many
    rows. Sorting by ts (as occupancy_series does) fixes it.
    """
    if events.empty:
        return 0
    by_arrival = events.sort_values("row_id")
    newest_so_far = by_arrival["ts"].cummax().shift()
    return int(np.sum(by_arrival["ts"].to_numpy() < newest_so_far.to_numpy()))
