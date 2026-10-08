import occupancy_api as gql
import pandas as pd


def node(i, location="lab"):
    return {"rowId": str(i), "location": location, "ts": f"2026-10-05T15:{i % 60:02d}:00+00:00",
            "delta": 1, "eventId": f"e{i}", "receivedAt": "2026-10-05T16:00:00.123456+00:00"}


class FakeServer:
    """Answers EVENTS_QUERY from a list of nodes, in pages, like the real API."""

    def __init__(self, nodes, page_size=3):
        self.nodes, self.page_size, self.calls = nodes, page_size, []

    def __call__(self, url, key, query, variables):
        self.calls.append(variables)
        rows = [n for n in self.nodes if int(n["rowId"]) > int(variables["minRowId"])]
        rows.sort(key=lambda n: int(n["rowId"]))
        start = int(variables["after"] or 0)
        page = rows[start:start + self.page_size]
        more = start + self.page_size < len(rows)
        return {"occupancyEvents": {"totalCount": len(rows),
                                    "pageInfo": {"hasNextPage": more,
                                                 "endCursor": str(start + self.page_size)},
                                    "nodes": page}}


def test_types():
    df = gql.to_frame([node(1)])
    assert df["row_id"].dtype == "int64"
    assert str(df["ts"].dt.tz) == "UTC"
    assert list(df.columns) == gql.COLUMNS


def test_sync_pages_then_fetches_only_new(tmp_path, monkeypatch):
    monkeypatch.setattr(gql, "OVERLAP", 2)
    path = tmp_path / "events.parquet"
    server = FakeServer([node(i) for i in range(1, 8)])
    first = gql.sync("u", "k", path=path, post=server)
    assert first == {"cached_before": 0, "fetched": 7, "new": 7, "total": 7}
    assert len(server.calls) == 3                       # 7 rows in pages of 3

    server.nodes.append(node(8))
    second = gql.sync("u", "k", path=path, post=server)
    assert server.calls[-1]["minRowId"] == "5"          # newest 7 minus overlap 2
    assert second["new"] == 1 and second["total"] == 8


def test_sync_picks_up_a_late_commit_inside_the_overlap(tmp_path, monkeypatch):
    # Row 4 committed after row 5 was already synced: the overlap re-read catches it.
    monkeypatch.setattr(gql, "OVERLAP", 3)
    path = tmp_path / "events.parquet"
    server = FakeServer([node(i) for i in (1, 2, 3, 5)])
    gql.sync("u", "k", path=path, post=server)
    server.nodes.append(node(4))
    stats = gql.sync("u", "k", path=path, post=server)
    assert stats["new"] == 1
    assert pd.read_parquet(path)["row_id"].tolist() == [1, 2, 3, 4, 5]
