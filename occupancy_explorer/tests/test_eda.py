import pandas as pd

from eda import (
    bucketed,
    daily_drift,
    late_arrivals,
    location_summary,
    occupancy_series,
    weekday_hour_profile,
)


def make_events(rows):
    """rows: (row_id, location, local time string, delta[, seconds of upload lag])"""
    out = []
    for r in rows:
        row_id, location, local, delta = r[:4]
        lag = r[4] if len(r) > 4 else 1
        ts = pd.Timestamp(local, tz="America/Denver").tz_convert("UTC")
        out.append({"row_id": row_id, "location": location, "ts": ts, "delta": delta,
                    "event_id": f"e{row_id}", "received_at": ts + pd.to_timedelta(lag, unit="s")})
    return pd.DataFrame(out)


def test_occupancy_is_running_sum_in_time_order():
    # Row 3 arrived last but happened first: the count must follow ts, not arrival.
    ev = make_events([(1, "lab", "2026-10-05 09:10", +1),
                      (2, "lab", "2026-10-05 09:20", -1),
                      (3, "lab", "2026-10-05 09:00", +1, 3600)])
    s = occupancy_series(ev)
    assert s["occupancy"].tolist() == [1, 2, 1]
    assert late_arrivals(ev) == 1


def test_summary_flags_negative_counts_and_mock():
    ev = make_events([(1, "lab", "2026-10-05 09:00", -1),
                      (2, "lab", "2026-10-05 09:05", +1),
                      (3, "mock-x", "2026-10-05 09:00", +1)])
    s = location_summary(ev).set_index("location")
    assert s.loc["lab", "lowest_count"] == -1
    assert s.loc["lab", "count_now"] == 0
    assert bool(s.loc["mock-x", "synthetic"]) and not bool(s.loc["lab", "synthetic"])


def test_bucketed_carries_occupancy_through_quiet_hours():
    ev = make_events([(1, "lab", "2026-10-05 09:05", +1),
                      (2, "lab", "2026-10-05 09:10", +1),
                      (3, "lab", "2026-10-05 12:30", -1)])
    b = bucketed(ev, "1h")
    assert b.loc["2026-10-05 09:00", "enters"] == 2
    assert b.loc["2026-10-05 10:00", "occupancy"] == 2      # nobody moved, still 2 inside
    assert b.loc["2026-10-05 12:00", "occupancy"] == 1
    assert b["net"].sum() == 1


def test_daily_drift_uses_local_days():
    # 23:30 Denver on Oct 5 is already Oct 6 in UTC; it must count toward Oct 5.
    ev = make_events([(1, "lab", "2026-10-05 08:00", +1),
                      (2, "lab", "2026-10-05 23:30", -1),
                      (3, "lab", "2026-10-06 08:00", +1)])
    d = daily_drift(ev).set_index("date")
    assert d.loc["2026-10-05", "net"] == 0
    assert d.loc["2026-10-06", "net"] == 1


def test_profile_averages_include_empty_hours():
    # Two Mondays; only the first has an entry at 9am, so the mean is 0.5, not 1.
    ev = make_events([(1, "lab", "2026-10-05 09:15", +1),
                      (2, "lab", "2026-10-12 10:15", -1)])
    p = weekday_hour_profile(ev)
    mon9 = p[(p["weekday_name"] == "Mon") & (p["hour"] == 9)]["mean_enters"].item()
    assert mon9 == 0.5
