"""
GraphQL client for the occupancy data, with a local Parquet cache.

The occupancy store (source-graphql in occupancy-backend) is read-only and
*insert-only*: events are added, never edited or deleted. That makes caching
easy and safe. We keep every event we have seen in `data/events.parquet`, and
a sync only asks the server for rows newer than the newest one we have.

Two details make that sync correct, not just fast:

- Pagination is by cursor (`first` / `after`), ordered by `rowId`, so a page
  boundary never skips or repeats a row even while new events arrive.
- `rowId` comes from a database sequence. Two inserts running at the same
  moment can *commit* out of order, so a row with a smaller id can appear just
  after we synced past it. Each sync therefore re-reads a small overlap below
  the newest id we hold and drops the duplicates by `rowId`.

Configuration:
- `OCCUPANCY_GRAPHQL_URL` (or `graphql_url` in config.json)
- `OCCUPANCY_GRAPHQL_KEY` (or `graphql_key` in .streamlit/secrets.toml, which
  is git-ignored). The key is a secret: never put it in config.json.
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "events.parquet"
PAGE_SIZE = 5000        # the API's maximum page size
OVERLAP = 500           # re-read this many ids below the newest cached one (see above)

COLUMNS = ["row_id", "location", "ts", "delta", "event_id", "received_at"]

EVENTS_QUERY = """
query Events($after: Cursor, $first: Int!, $minRowId: BigInt!) {
  occupancyEvents(
    first: $first
    after: $after
    orderBy: [ROW_ID_ASC]
    filter: { rowId: { greaterThan: $minRowId } }
  ) {
    totalCount
    pageInfo { hasNextPage endCursor }
    nodes { rowId location ts delta eventId receivedAt }
  }
}
"""


class GraphQLError(RuntimeError):
    """The server answered, but with errors (or not at all)."""


def load_config() -> dict:
    """config.json next to this file; an empty dict if it is missing or malformed."""
    try:
        return json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def graphql(url: str, key: str, query: str, variables: dict | None = None,
            retries: int = 4, timeout: float = 30) -> dict:
    """POST one GraphQL request and return its `data`.

    Throttling (429) and server errors (5xx) are retried with exponential
    backoff; anything else raises GraphQLError with the server's message.
    """
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, method="POST", headers={
            "content-type": "application/json", "x-api-key": key})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise GraphQLError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e
        except urllib.error.URLError as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise GraphQLError(f"cannot reach {url}: {e.reason}") from e
        if payload.get("errors"):
            raise GraphQLError("; ".join(err.get("message", "?") for err in payload["errors"]))
        return payload["data"]
    raise GraphQLError("unreachable")


def to_frame(nodes: list[dict]) -> pd.DataFrame:
    """GraphQL nodes -> a typed DataFrame (times are tz-aware UTC)."""
    df = pd.DataFrame(nodes, columns=["rowId", "location", "ts", "delta", "eventId", "receivedAt"])
    df = df.rename(columns={"rowId": "row_id", "eventId": "event_id", "receivedAt": "received_at"})
    df["row_id"] = df["row_id"].astype("int64")          # BigInt arrives as a string
    df["delta"] = df["delta"].astype("int8")
    df["location"] = df["location"].astype("string")
    df["event_id"] = df["event_id"].astype("string")
    for col in ("ts", "received_at"):
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    return df[COLUMNS]


def fetch_since(url: str, key: str, min_row_id: int, post=graphql) -> pd.DataFrame:
    """Every event with rowId > min_row_id, page by page. `post` is swappable for tests."""
    nodes, after = [], None
    while True:
        page = post(url, key, EVENTS_QUERY, {
            "after": after, "first": PAGE_SIZE, "minRowId": str(min_row_id)})["occupancyEvents"]
        nodes.extend(page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            return to_frame(nodes)
        after = page["pageInfo"]["endCursor"]


def read_cache(path: Path = CACHE) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return to_frame([])


def merge(cached: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Union of the two, one row per rowId, in rowId order. Fresh rows win on a tie."""
    frames = [f for f in (cached, fresh) if not f.empty]
    if not frames:
        return to_frame([])
    both = pd.concat(frames, ignore_index=True)
    both = both.drop_duplicates("row_id", keep="last").sort_values("row_id")
    return both.reset_index(drop=True)


def sync(url: str, key: str, path: Path = CACHE, full: bool = False, post=graphql) -> dict:
    """Bring the cache up to date. Returns counts that describe what happened.

    full=True ignores the cache and re-reads everything (use it if you suspect
    the cache is wrong; the server is the system of record).
    """
    cached = to_frame([]) if full else read_cache(path)
    newest = int(cached["row_id"].max()) if not cached.empty else 0
    fresh = fetch_since(url, key, max(0, newest - OVERLAP), post=post)
    merged = merge(cached, fresh)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(path)                     # atomic: a crash never leaves half a file
    return {
        "cached_before": len(cached),
        "fetched": len(fresh),
        "new": len(merged) - len(cached),
        "total": len(merged),
    }


if __name__ == "__main__":
    # pixi run -e occupancy sync [--full]
    import sys

    cfg = load_config()
    url = os.environ.get("OCCUPANCY_GRAPHQL_URL") or cfg.get("graphql_url")
    key = os.environ.get("OCCUPANCY_GRAPHQL_KEY")
    if not key:
        secrets = HERE / ".streamlit" / "secrets.toml"
        if secrets.exists():
            import tomllib
            key = tomllib.loads(secrets.read_text(encoding="utf-8")).get("graphql_key")
    if not (url and key):
        sys.exit("Set OCCUPANCY_GRAPHQL_URL and OCCUPANCY_GRAPHQL_KEY (see README.md).")
    stats = sync(url, key, full="--full" in sys.argv)
    print(f"{stats['new']} new events; {stats['total']} cached in {CACHE.relative_to(HERE)}")
