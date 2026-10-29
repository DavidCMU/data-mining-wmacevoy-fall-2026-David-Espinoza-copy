# Occupancy Explorer

A Streamlit dashboard over the occupancy events that people-counting sensors
send in: the Raspberry Pi over a doorway (`ingress-pi`) and the simulator
(`ingress-mock`). It reads them from the read-only GraphQL API
(`source-graphql`), keeps a local Parquet cache that only ever fetches what's
new, and shows how each room fills and empties, and where the count goes
wrong.

It is the same shape as [`colorado_river`](../colorado_river/README.md):
remote API → local cache → features → dashboard. The difference is that you
can see, and change, every step that produces this data. See the
[project overview](../docs/occupancy/index.md).

## Running it

From the **repository root**:

```bash
pixi run -e occupancy app
```

Then click the `Local URL:` that Streamlit prints.

| Task | What it does |
| --- | --- |
| `pixi run -e occupancy app` | Launch the dashboard (it syncs once when a browser session opens) |
| `pixi run -e occupancy sync` | Fetch new events into `data/events.parquet` (`--full` re-reads everything) |
| `pixi run -e occupancy test` | Run the tests |
| `pixi run -e occupancy lint` | Check code style |
| `pixi run -e occupancy clean` | Delete `data/` |

## The API key

The API URL is in `config.json`. The key is a secret, so it isn't. Use either:

- an environment variable, `OCCUPANCY_GRAPHQL_KEY`, or
- `occupancy_explorer/.streamlit/secrets.toml` (git-ignored) containing
  `graphql_key = "..."`.

If you deployed the backend yourself, the key is in SSM. This puts it in the
environment without printing it:

```bash
export OCCUPANCY_GRAPHQL_KEY=$(aws ssm get-parameter --region us-west-2 \
  --name /occupancy-graphql/api-key --with-decryption --query Parameter.Value --output text)
```

Otherwise, ask your instructor for it. Without a key, the app still shows
whatever is already in the cache.

## Layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI: all-locations summary, one-location charts, a GraphQL query tab |
| `occupancy_api.py` | GraphQL client (retries with backoff) and the incremental Parquet cache |
| `eda.py` | Running count, time buckets, weekday × hour profile, daily drift, upload lag |
| `config.json` | The GraphQL endpoint |
| `.streamlit/config.toml` | Streamlit settings (headless, no telemetry). Tracked on purpose |
| `tests/` | Unit tests for `eda` and for the sync (against a fake server) |

## The data

One row per event: `row_id`, `location`, `ts` (when it happened), `delta`
(+1 entered, −1 left), `event_id`, and `received_at` (when the server stored
it). `(location, event_id)` is the record key, so a sensor that retries an
upload never counts the same person twice.

**Nothing stores the count.** Occupancy is the running sum of `delta`, in `ts`
order. Every chart in the app is computed from that one idea.

Times are stored in UTC and shown in `America/Denver`. "By day" and "by hour"
always mean local days and hours. An event at 23:30 on a Monday belongs to
Monday, even though in UTC it is already Tuesday.

## Why the cache is safe

The store is insert-only: events are never edited or deleted. So a cached row
never goes stale, and a sync only asks for rows with a larger `row_id` than the
newest cached one, paging by cursor so nothing is skipped or repeated.

There is one catch. `row_id` comes from a database sequence, and two inserts
that run at the same moment can commit out of order. Row 41 can become visible
just after you've already synced past row 42. Each sync therefore re-reads the
last 500 ids and de-duplicates on `row_id`. `tests/test_occupancy_api.py`
reproduces that case.

## What to look for

These are the lessons the data is built to teach:

1. **Counts that don't add up.** `lowest_count` below zero means more people
   left than entered: the sensor missed entries (two people side by side make
   one warm blob). The *Daily drift* chart shows every day whose entries and
   exits don't cancel out. A room that empties at night should net to zero.
   A sensor with every event reversed shows up as a count that only falls:
   that means `enter_direction` is set the wrong way round.
2. **Synthetic vs real.** Locations starting with `mock-` come from the
   simulator. The app marks them, and you can hide them.
3. **Arrival order isn't time order.** *Late arrivals* counts events the
   server got after a later-timestamped one. The Pi queues events during a
   network outage and sends them afterwards, and a backfill sends a month of
   history at once. *Upload lag* shows which of those you're looking at.
4. **Quiet hours count.** The weekday × hour profile averages over *every*
   hour, including empty ones. Averaging only the hours that have events
   makes every room look busy.

## Known rough edges

1. **The whole table is cached.** Fine for a classroom's worth of events
   (a month of mock backfill is tens of thousands of rows). A real deployment
   would sync one location or a time window.
2. **No dedupe across `event_id` schemes.** If a sensor changes how it makes
   ids, the same visit can arrive twice under two keys, and nothing here
   catches it. That's a property of the data, not of the app.
3. **The cache trusts the server.** If you suspect the cache is wrong, use
   *Re-read all* (or `sync --full`). The server is the system of record.
