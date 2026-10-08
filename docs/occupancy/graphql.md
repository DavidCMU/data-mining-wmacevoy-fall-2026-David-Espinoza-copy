---
title: Occupancy — GraphQL tutorial
---

# GraphQL tutorial: asking the occupancy data questions

[Course home](../) · [Overview](./) · [Student project](project.html) · **GraphQL tutorial**

This tutorial teaches GraphQL by using it. Every query below runs against the course's occupancy API, which holds about eight weeks of synthetic traffic for five rooms. By the end you will be able to write your own queries, page through large results correctly, and pull data into pandas.

GraphQL is a query language for APIs. A REST API gives you fixed responses at fixed URLs: `GET /occupancy/mock-lobby` returns whatever its author decided to return. A GraphQL API has **one** URL and a **schema** that describes everything you can ask for. You send a query that says exactly which fields you want, and the answer comes back in the same shape as the query.

## The practice data

Five synthetic rooms, made by `ingress-mock`'s backfill. All names start with `mock-`, because none of this is real.

| Location | Shape | History | What to look for |
|---|---|---|---|
| `mock-lobby` | office: morning rush, lunch bump | 8 weeks | quiet weekends |
| `mock-cafe` | breakfast, a big lunch peak, short visits | 8 weeks | busier weekends than the lobby |
| `mock-classroom-101` | bursts of arrivals just before each hour, 8 am to 5 pm | 6 weeks | **nobody** on weekends |
| `mock-library` | steady all day until 10 pm, long visits | 6 weeks | busy weekends |
| `mock-gym` | early-morning and after-work peaks | 4 weeks | weekends busier than weekdays |

Each event is one person entering (`delta: 1`) or leaving (`delta: -1`). Nothing stores the count: occupancy is the running sum of `delta`. See the [overview](./) for why.

## Where to run queries

You need the API key: ask your instructor. Then pick one of these:

1. **Ruru**, the API's built-in explorer: <https://b8nhkd7i8l.execute-api.us-west-2.amazonaws.com/>. Open **Headers** at the bottom and enter `{"x-api-key": "<the key>"}`. Ruru autocompletes as you type (Ctrl-Space), and its **Docs** panel describes every type. This is the best place to start.
2. **The explorer's GraphQL tab**: `pixi run -e occupancy app` from the course repository (see [`occupancy_explorer/README.md`](https://github.com/wmacevoy/data-mining-wmacevoy-fall-2026/blob/data-science-principles/occupancy_explorer/README.md)). It also shows results as a table.
3. **curl** or **Python**, covered in [section 11](#11-from-python).

---

## 1. Your first query

```graphql
query {
  occupancyEvents(first: 5) {
    nodes {
      location
      ts
      delta
    }
  }
}
```

Read it from the outside in. `query` means read, the only kind of operation this API allows. `occupancyEvents` is a **field** on the root `Query` type, and `first: 5` is an **argument** to it. `nodes` is the list of results, and inside it you name the three fields you want from each event.

The answer has the same shape:

```json
{
  "data": {
    "occupancyEvents": {
      "nodes": [
        { "location": "lab", "ts": "2026-10-01T16:41:46.000000+00:00", "delta": 1 },
        ...
      ]
    }
  }
}
```

Times are in UTC (the `+00:00`). A room in Grand Junction is on Denver time, 6 hours behind UTC in summer (MDT) and 7 in winter (MST).

**Try it:** add `eventId` and `receivedAt` to the fields. Remove `location`. The answer changes to match. That is the core idea of GraphQL: **you choose the fields**.

## 2. What can I ask for? The schema

Every GraphQL API describes itself. Ruru's Docs panel reads this description, and you can query it yourself:

```graphql
query {
  __type(name: "OccupancyEvent") {
    fields {
      name
      type { name kind ofType { name } }
    }
  }
}
```

An `OccupancyEvent` has:

| Field | Type | Meaning |
|---|---|---|
| `rowId` | `BigInt!` | The database's id, in insertion order |
| `location` | `String!` | Which room |
| `ts` | `Datetime!` | When it happened, as the sensor reported it |
| `delta` | `Int!` | `1` entered, `-1` left |
| `eventId` | `String!` | The sender's id; `(location, eventId)` is the record key |
| `receivedAt` | `Datetime!` | When the server stored it |
| `id` | `ID!` | A global id for this object, which every type here has |

The `!` means *never null*. `BigInt` arrives as a **string** (`"4021"`), because JSON numbers lose precision above 2⁵³. Convert it yourself.

## 3. Ordering and counting

```graphql
query {
  occupancyEvents(first: 3, orderBy: [TS_DESC]) {
    totalCount
    nodes {
      location
      ts
      delta
    }
  }
}
```

- `orderBy` takes a **list**, so you can break ties: `[LOCATION_ASC, TS_ASC]`. The default is `PRIMARY_KEY_ASC` (by `rowId`), which is the order the server *received* events, not the order they *happened*. Those differ whenever a sensor uploads late.
- `totalCount` is the number of matching rows, however many `first` returns. It costs the server a count, so ask for it only when you need it.

## 4. Filtering

`filter` narrows the rows. You can filter on `rowId`, `location`, `ts` and `eventId`, the indexed columns, so every filter here stays fast. The operators are `equalTo`, `notEqualTo`, `in`, `lessThan`, `lessThanOrEqualTo`, `greaterThan` and `greaterThanOrEqualTo`.

One room on one day, in Denver time:

```graphql
query {
  occupancyEvents(
    filter: {
      location: { equalTo: "mock-classroom-101" }
      ts: { greaterThanOrEqualTo: "2026-09-29T00:00:00-06:00", lessThan: "2026-09-30T00:00:00-06:00" }
    }
    orderBy: [TS_ASC]
  ) {
    totalCount
    nodes {
      ts
      delta
    }
  }
}
```

- Fields inside one filter object are **and**ed together.
- Write the day boundaries **with the Denver offset** (`-06:00` in summer, `-07:00` in winter). Without it, `"2026-09-29"` means midnight UTC, which is 6 pm the evening before in Denver.
- Use `>=` start and `<` end (*half-open*). Then consecutive days never share or drop an event that falls exactly on midnight.

Several rooms at once, and `or`:

```graphql
query {
  occupancyEvents(
    first: 5
    filter: {
      or: [
        { location: { equalTo: "mock-gym" } }
        { location: { in: ["mock-cafe", "mock-library"] } }
      ]
    }
  ) {
    totalCount
    nodes {
      location
      ts
    }
  }
}
```

**There is no `delta` filter.** You can't ask the server for "only the entries". Fetch the events and filter them yourself (section 11). This API only filters columns that have an index, so that no query can force a slow scan of the whole table.

**A trap: comparing text.** There's no `startsWith`, but a prefix is a range: every string that starts with `mock` sorts at or after `"mock"` and before `"mocl"`.

```graphql
query {
  occupancyEvents(filter: { location: { greaterThanOrEqualTo: "mock", lessThan: "mocl" } }) {
    totalCount
  }
}
```

The obvious version, `"mock-"` to `"mock-~"`, returns **zero** rows. The database sorts text by English language rules, not character codes, and those rules skip punctuation such as `-` on the first pass. So `"mock-test"` compares like `"mocktest"`, and that sorts after `"mock-~"`. Range filters on text are only safe with letters and digits.

## 5. Variables

Pasting values into the query text gets old, and in a program it invites injection bugs. Declare **variables** instead, with their types, and send their values separately:

```graphql
query OneDay($location: String!, $from: Datetime!, $to: Datetime!) {
  occupancyEvents(
    filter: { location: { equalTo: $location }, ts: { greaterThanOrEqualTo: $from, lessThan: $to } }
    orderBy: [TS_ASC]
  ) {
    totalCount
    nodes {
      ts
      delta
    }
  }
}
```

```json
{ "location": "mock-gym", "from": "2026-10-03T00:00:00-06:00", "to": "2026-10-04T00:00:00-06:00" }
```

In Ruru, the values go in the **Variables** pane. `OneDay` is the **operation name**. It's optional, but it shows up in the server's logs and makes errors easier to trace. Name every query you put in a program.

## 6. Aliases: several questions in one request

You can't ask for the same field twice under one name, but you can **alias** each copy. This counts a week of events in three rooms with one request:

```graphql
query WeekCounts {
  lobby: occupancyEvents(filter: { location: { equalTo: "mock-lobby" }, ts: { greaterThanOrEqualTo: "2026-09-28T00:00:00-06:00", lessThan: "2026-10-05T00:00:00-06:00" } }) {
    totalCount
  }
  cafe: occupancyEvents(filter: { location: { equalTo: "mock-cafe" }, ts: { greaterThanOrEqualTo: "2026-09-28T00:00:00-06:00", lessThan: "2026-10-05T00:00:00-06:00" } }) {
    totalCount
  }
  gym: occupancyEvents(filter: { location: { equalTo: "mock-gym" }, ts: { greaterThanOrEqualTo: "2026-09-28T00:00:00-06:00", lessThan: "2026-10-05T00:00:00-06:00" } }) {
    totalCount
  }
}
```

The answer uses your names: `{"lobby": {"totalCount": ...}, "cafe": ..., "gym": ...}`.

## 7. Fragments: reuse a set of fields

When several parts of a query want the same fields, name the set once:

```graphql
fragment EventFields on OccupancyEvent {
  ts
  delta
  eventId
}

query FirstAndLast {
  first: occupancyEvents(first: 1, orderBy: [TS_ASC], filter: { location: { equalTo: "mock-library" } }) {
    nodes { ...EventFields }
  }
  last: occupancyEvents(first: 1, orderBy: [TS_DESC], filter: { location: { equalTo: "mock-library" } }) {
    nodes { ...EventFields }
  }
}
```

This one query tells you the span of a room's data, which is the first thing to check about any dataset.

## 8. Pagination: getting *all* the rows

A page holds at most **5000** rows. Ask for more and the server gives you 5000. Most rooms here have more than that, so a program has to fetch page after page. Ask for `pageInfo` and pass `endCursor` back as `after`:

```graphql
query Page($after: Cursor) {
  occupancyEvents(first: 1000, after: $after, orderBy: [TS_ASC, ROW_ID_ASC], filter: { location: { equalTo: "mock-cafe" } }) {
    totalCount
    pageInfo {
      hasNextPage
      endCursor
    }
    nodes {
      rowId
      ts
      delta
    }
  }
}
```

```json
{ "after": null }
```

Run it, copy `endCursor` into `after`, and run it again. Stop when `hasNextPage` is `false`.

- A **cursor** marks a position in the ordering. "After this row" stays correct even if new events arrive while you page. With `offset: 1000` instead, an event inserted earlier in the order would shift every later row by one, and you'd see one row twice.
- **Make the order total.** `TS_ASC` alone has ties (two people in the same second), and tied rows can come back in a different order on the next request. Adding `ROW_ID_ASC`, which is unique, settles every tie.
- **Check your work.** At the end you should have `totalCount` rows and no `rowId` twice. The course's pull script refuses to save the file if either check fails.

## 9. Looking up one event

Two fields return a single event instead of a list. One looks it up by the database id:

```graphql
query {
  occupancyEvent(rowId: "1") {
    location
    ts
    eventId
  }
}
```

The other looks it up by the **record key** `(location, eventId)`. This is how a sender can check whether one particular event arrived:

```graphql
query {
  occupancyEventByLocationAndEventId(location: "rollout-test-2026-10-08", eventId: "rollout-check-1") {
    rowId
    ts
    delta
    receivedAt
  }
}
```

A key that doesn't exist returns `null`, not an error.

## 10. The guard rails

This API is for **reading**. Try:

```graphql
mutation {
  __typename
}
```

> Schema is not configured to execute mutation operation (this API is read-only)

The schema has no `Mutation` type at all. On top of that, the server logs in to the database as a role that can only `SELECT`. Also:

- **Depth:** at most 5 levels of nested fields, so no query can blow up.
- **Page size:** `first` above 5000 is lowered to 5000.
- **Check `errors`, not just the status code.** A mistake in the query itself (a typo, an unknown field, a mutation) comes back as HTTP 400 with an `errors` list. But an error *while running* a valid query comes back as HTTP **200**, with an `errors` list next to whatever `data` did succeed. A wrong key is HTTP 401.

## 11. From Python

Everything above is a `POST` of JSON: `{"query": ..., "variables": ...}`. This function pages through one room and returns a pandas DataFrame:

```python
import json, os, urllib.error, urllib.request
import pandas as pd

URL = "https://b8nhkd7i8l.execute-api.us-west-2.amazonaws.com/graphql"
KEY = os.environ["OCCUPANCY_GRAPHQL_KEY"]          # never paste the key into code

QUERY = """
query Room($location: String!, $after: Cursor) {
  occupancyEvents(first: 5000, after: $after, orderBy: [TS_ASC, ROW_ID_ASC],
                  filter: { location: { equalTo: $location } }) {
    totalCount
    pageInfo { hasNextPage endCursor }
    nodes { rowId ts delta }
  }
}
"""

def graphql(query, variables):
    body = json.dumps({"query": query, "variables": variables}).encode()
    request = urllib.request.Request(URL, data=body, headers={
        "content-type": "application/json", "x-api-key": KEY})
    try:
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as e:
        if e.code != 400:                           # 401: wrong key; 5xx: try again later
            raise
        payload = json.load(e)                      # 400: the query itself is wrong
    if payload.get("errors"):                       # can come with HTTP 200 too
        raise RuntimeError(payload["errors"])
    return payload["data"]

def room(location):
    rows, after = [], None
    while True:
        page = graphql(QUERY, {"location": location, "after": after})["occupancyEvents"]
        rows += page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            break
        after = page["pageInfo"]["endCursor"]
    df = pd.DataFrame(rows)
    assert len(df) == page["totalCount"] and df["rowId"].is_unique
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601").dt.tz_convert("America/Denver")
    return df

cafe = room("mock-cafe")
cafe["occupancy"] = cafe["delta"].cumsum()        # the running count
print(cafe.set_index("ts")["occupancy"].resample("1h").max().tail(24))
```

From the shell, `curl` does the same thing:

```bash
curl -s https://b8nhkd7i8l.execute-api.us-west-2.amazonaws.com/graphql \
  -H "x-api-key: $OCCUPANCY_GRAPHQL_KEY" -H 'content-type: application/json' \
  -d '{"query": "query { occupancyEvents(first: 2) { nodes { location ts delta } } }"}'
```

For real work, use the course's tools rather than rewriting this. The explorer's `occupancy_api.py` keeps a local Parquet copy and fetches only new events, and `source-graphql/examples/pull_events.py` saves one room and date range to Parquet.

---

## Exercises

Write each one before you open the answer. Check counts against the explorer where you can.

**1.** How many events does `mock-gym` have in total? How many did it have before 1 October (Denver time)?

<details markdown="1"><summary>Answer</summary>

```graphql
query GymCounts {
  all: occupancyEvents(filter: { location: { equalTo: "mock-gym" } }) {
    totalCount
  }
  beforeOct: occupancyEvents(filter: { location: { equalTo: "mock-gym" }, ts: { lessThan: "2026-10-01T00:00:00-06:00" } }) {
    totalCount
  }
}
```

Aliases (section 6) let you ask both in one request.
</details>

**2.** What were the first and the last event at `mock-classroom-101` on Tuesday 29 September? Do they fit a class schedule from 8 am to 5 pm?

<details markdown="1"><summary>Answer</summary>

```graphql
query ClassDay {
  first: occupancyEvents(first: 1, orderBy: [TS_ASC, ROW_ID_ASC], filter: { location: { equalTo: "mock-classroom-101" }, ts: { greaterThanOrEqualTo: "2026-09-29T00:00:00-06:00", lessThan: "2026-09-30T00:00:00-06:00" } }) {
    nodes { ts delta }
  }
  last: occupancyEvents(first: 1, orderBy: [TS_DESC, ROW_ID_DESC], filter: { location: { equalTo: "mock-classroom-101" }, ts: { greaterThanOrEqualTo: "2026-09-29T00:00:00-06:00", lessThan: "2026-09-30T00:00:00-06:00" } }) {
    nodes { ts delta }
  }
}
```

Convert from UTC first (subtract 6 hours). The first event is an entry shortly after 8 am (about 14:20 UTC): the simulation has nobody arriving before 8. The last is a leave in the evening, well after 5 pm. Arrivals stop at 5, but a few people stay a long time. Visit lengths in this simulation have a long tail, as they often do in real rooms.
</details>

**3.** Does `mock-classroom-101` have *any* events on a weekend? Check Saturday 3 and Sunday 4 October in one request.

<details markdown="1"><summary>Answer</summary>

```graphql
query Weekend {
  occupancyEvents(filter: { location: { equalTo: "mock-classroom-101" }, ts: { greaterThanOrEqualTo: "2026-10-03T00:00:00-06:00", lessThan: "2026-10-05T00:00:00-06:00" } }) {
    totalCount
  }
}
```

It should be 0, or a few leaves just after midnight from people still inside on Friday. The simulation turns weekend traffic off for this room. Compare `mock-gym` over the same two days.
</details>

**4.** Which rooms does the data hold? There's no "list the locations" query. Find a way anyway, and say why it's expensive.

<details markdown="1"><summary>Answer</summary>

Ask for one page ordered by location, and use a `greaterThan` filter to jump past each location you've already seen:

```graphql
query NextLocation($after: String!) {
  occupancyEvents(first: 1, orderBy: [LOCATION_ASC], filter: { location: { greaterThan: $after } }) {
    nodes { location }
  }
}
```

Start with `{"after": ""}`. Feed each answer back in, and stop when `nodes` is empty. Each step is one fast indexed lookup, so N locations take N+1 requests instead of downloading every event. (Databases call this a *skip scan*.) A real API would add a `locations` field. This one is generated from the table, so it doesn't have one.
</details>

**5.** Using the Python function above, find the busiest hour of the week at `mock-cafe`. On average, how many people are inside at noon on a weekday? On a Saturday?

<details markdown="1"><summary>Hint</summary>

The server can't do this for you. Pull every event and do the arithmetic in pandas. Occupancy at a moment is the running sum of `delta`. Resample with `"1h"` and `.last()`, fill the empty hours forward, then group by `index.dayofweek` and `index.hour`. Keep the times in Denver time, or "noon" means the wrong hour.
</details>

**6.** Fetch `mock-lobby` twice: once ordered by `[TS_ASC]` only, once by `[TS_ASC, ROW_ID_ASC]`, both in pages of 500. Do you get the same rows? Is that guaranteed?

<details markdown="1"><summary>Answer</summary>

This time you probably get the same rows. But with `TS_ASC` alone, two events in the same second can come back in either order, and if a page boundary falls between them, one page might repeat one of them while another page misses the other. Usually it works, and nothing tells you when it doesn't. That's the worst kind of bug. The unique tiebreaker makes it impossible.
</details>

---

## Going further

- The [student project](project.html) has you build this API yourself (Part 3), from a spec, with Claude.
- The GraphQL specification and guides: <https://graphql.org/learn/>.
- The server is generated from the database table by PostGraphile: <https://postgraphile.org/>. Filtering comes from its connection-filter plugin, restricted here to the operators above.
