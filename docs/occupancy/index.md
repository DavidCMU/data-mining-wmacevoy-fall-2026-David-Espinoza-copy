---
title: Occupancy — overview
---

# Occupancy: a data source you build yourself

[Course home](../) · **Overview** · [Student project](project.html) · [GraphQL tutorial](graphql.html)

Most datasets in this course arrive finished. This one you build. A sensor over a doorway counts people walking in and out, a small cloud service keeps the record, and you pull the data out to analyse it. Building it yourself lets you see every place a number can go quietly wrong: a sensor that misses people, a clock that is off by a day, a retry that counts someone twice. Those are the same failures you will have to find, without the source code, in every dataset someone hands you later.

## The pieces

Each component is named for the job it does with the data:

| Component | Role | What it is |
|---|---|---|
| **ingress-mock** | writes data in | A web page that simulates a sensor: click Enter/Leave, run a live simulation, or backfill a month of synthetic traffic. |
| **ingress-pi** | writes data in | A Raspberry Pi 4 with a thermal camera (Grid-EYE) over a real doorway. It keeps everything in SQLite and uploads on a cron job. |
| **state-aws** | system of record | A REST API on AWS Lambda with PostgreSQL (RDS) behind it. Insert-only, so history is never rewritten. |
| **source-graphql** | reads data out | A read-only GraphQL API over the same database, plus a script that pulls events into Parquet for pandas. |
| **occupancy_explorer** | looks at it | A Streamlit dashboard in the course repository. It keeps a local Parquet copy of every event, synced from source-graphql, and charts each room: occupancy over time, a typical week, days whose count doesn't add up, and upload delays. It has a tab for running your own GraphQL queries. |

```
 ingress-mock ──┐
  (simulated)   │   POST /enter|leave/{location}/{ts}?id={event_id}
                ├──────────────────────────────────────────────────▶ state-aws ──▶ source-graphql ──▶ explorer / notebook
 ingress-pi  ───┘          the contract every ingress follows          (RDS)        (read-only)
  (real sensor)
```

## The one idea: the contract

Everything hangs on one small agreement between the things that write and the thing that stores:

- An **event** is a `location`, a time `ts`, and a `delta` of **+1** (someone entered) or **−1** (someone left).
- Occupancy at any moment is the **sum of the deltas** up to that moment. Nothing stores "the count"; it is always computed from the events.
- Every event carries an **`event_id`** chosen by the sender. **`(location, event_id)` is the record key**: the server stores each pair once. A sender that is unsure whether an event arrived simply sends it again. That is what makes retrying safe: an event may be delivered more than once, but it is only ever counted once.
- The store is **insert-only**. Events are never edited or deleted. A mistake is fixed by adding a correcting event, and the history shows that it happened.

Any program that follows the contract is an ingress, so the mock and the Pi can be swapped freely, and you could write a third one.

## The real sensor

The Pi watches the doorway with a Panasonic AMG8833 "Grid-EYE": an 8×8 grid of temperatures, 10 times a second. A person is a warm blob. Software learns what the empty doorway looks like, finds the warm blobs, follows each one from frame to frame, and when one disappears decides whether it crossed the doorway, and in which direction.

```
 Grid-EYE ─I2C─▶ sensor (systemd, always on) ─▶ SQLite ─▶ upload (cron, every 2 min) ─▶ state-aws
                 count people; keep frames       frames,    send what hasn't been sent;
                                                 events     retry later if the network is down
```

- **Two processes.** The sensor runs all the time, because cron can't start a job 10 times a second. Uploading is a cron job, so a network outage never stops the counting: events wait in SQLite and go out on a later run. This pattern is called *store and forward*.
- **The raw frames are kept** (for a week by default). They are the real data-mining material. The people counter is a model, the frames are its input, and you can replay them to measure and improve it.
- **The Pi has no battery-backed clock.** Until it syncs the time over the network, the time can be days off, so the sensor refuses to record until the clock is synced. Timestamps are data too.

## Where the data goes wrong (on purpose)

These are features of the project, not bugs to hide:

1. **Counting errors.** Two people walking side by side merge into one blob. The Pi then misses an enter or a leave, and the count drifts: it never quite returns to zero at night, or it goes negative. The database accepts this, because the sensor really did report it.
2. **Synthetic vs real.** Mock locations start with `mock-`. Mixing them up is the classic way a model ends up learning from fake data.
3. **Ground truth is expensive.** To know how wrong the Pi is, someone has to count by hand. Have them click Enter and Leave on the mock page under a `truth-<room>` location while the Pi records `<room>`, then compare the two.
4. **Duplicates and retries.** The record key makes resending harmless, but only if every sender keeps its ids stable. Change the id scheme and the same visit is counted twice.

## Looking at the data

From the course repository root:

```bash
pixi run -e occupancy app
```

You need the read API key (ask your instructor, or see the explorer's README if you deployed your own). The explorer marks `mock-` locations, shows a count that goes below zero as missed entries, and shows each day whose entries and exits don't cancel out. Those are the problems listed above, made visible. See [`occupancy_explorer/README.md`](https://github.com/wmacevoy/data-mining-wmacevoy-fall-2026/blob/data-science-principles/occupancy_explorer/README.md).

## Cost

Everything except the database scales to zero when idle and fits in the AWS free tier. The database is a `db.t4g.micro` PostgreSQL instance with 20 GB of storage, which is free-tier eligible. Check what your own account's free tier covers before you deploy, set a budget alert, and run the teardown at the end of the semester. The [project page](project.html) walks through this.

## Next

The **[student project](project.html)** builds all four components in your own AWS account, in order, using Claude Code from a written spec at each step.
