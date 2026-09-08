# Blue Mesa Reservoir Storage vs. Colorado Avg. Temperature

A small Streamlit dashboard comparing two static CSV downloads on a common
monthly time scale, unlike `../colorado_river` this project does not call a
live API — the data is downloaded once and read from disk, which is a
common and perfectly valid way to start a data-mining assignment.

## The problem

**Does Blue Mesa Reservoir's storage level track Colorado's statewide
average temperature, month to month?** Storage is recorded daily; the
temperature series is only available monthly, so the two can't be compared
directly without first putting them on the same time scale. This app
resamples daily storage to a monthly mean, joins it against the monthly
temperature series, and asks how correlated the two are.

## Data sources

| Series | File | Granularity | Source |
| --- | --- | --- | --- |
| Blue Mesa Reservoir storage (acre-feet) | `data/raw/blue_mesa_storage.csv` | daily | [Bureau of Reclamation RISE](https://data.usbr.gov/rise) |
| Colorado avg. temperature (°F) | `data/raw/co_avg_temp.csv` | monthly | statewide monthly average |

Both are committed as static CSVs rather than fetched live — see
`data_loader.py` for exactly how each is parsed.

## Running it

From the **repository root**:

```bash
pixi run reservoir-climate
```

Then click the `Local URL:` that Streamlit prints.

| Task | What it does |
| --- | --- |
| `pixi run reservoir-climate` | Launch the dashboard |
| `pixi run reservoir-climate-test` | Run the tests |

## Layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI: time-series comparison, correlation scatter + Pearson r |
| `data_loader.py` | Parses both CSVs, resamples storage to monthly, joins them |
| `data/raw/` | The two source CSVs, committed as-is |
| `.streamlit/config.toml` | `headless = true` so Streamlit doesn't block on the first-run email prompt (see `../colorado_river/README.md` for why this matters) |
| `tests/` | Unit tests for `data_loader.py` |

## How the monthly resampling works

- Temperature is already monthly (`YYYYMM` → month-start date).
- Storage is daily and gets resampled with `.resample("MS").mean()`. A
  month is only kept if it has at least 25 daily readings — this drops
  whatever partial month sits at the end of a CSV that was downloaded
  mid-month, so that month isn't silently averaged from 5–6 days instead of
  ~30.
- The two monthly series are then joined with an inner join, so only months
  present in *both* sources appear in the correlation.

## What the correlation shows (and why it's not stronger)

Pearson r between monthly storage and monthly temperature comes out weak-to
-moderate (see the app for the live number). That's expected: reservoir
storage is driven mostly by snowmelt inflow timing and dam-operator release
decisions — a fixed seasonal shape that peaks in early summer — while
temperature is a broader, more continuous climate signal. A tighter
relationship would more likely show up against inflow volume, or against
temperature lagged by a month or two (this month's heat driving next
month's snowmelt runoff), which would be natural follow-up analysis.

## License

MIT — see [`../LICENSE`](../LICENSE).
