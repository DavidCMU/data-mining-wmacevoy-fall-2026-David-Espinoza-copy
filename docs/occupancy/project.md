---
title: Occupancy — student project
---

# Student project: build the occupancy data source

[Course home](../) · [Overview](./) · **Student project**

You will build the four components from the [overview](./) in your own AWS account, in the order data flows through them, then use what you built to answer a question about a real room. You will build each one with **Claude Code**, from a written spec. The goal is not that you type less. The goal is that you learn to say exactly what you want, and then check that you got it.

| Part | You build | You learn |
|---|---|---|
| 0 | nothing yet: read the contract | what an event is; why the record key is `(location, event_id)` |
| 1 | **state-aws**: the API and database | private databases, IAM auth, insert-only records, idempotency |
| 2 | **ingress-mock**: the simulated sensor | Poisson arrivals, synthetic data, reproducible backfills |
| 3 | **source-graphql**: the read side | read-only access, pagination, getting data into pandas |
| 4 | **ingress-pi**: the real sensor | I2C hardware, store and forward, cron vs services, a model that counts people |
| 5 | the analysis | measuring how wrong the sensor is before you trust it |

The reference implementation and its specs are in the course repository under `source/occupancy-backend/`, one folder per component, each with a `requirements.md`. You can use those specs as they are, or write your own with Claude and compare.

---

## Before you start

**Accounts and tools**

1. An **AWS account** of your own. Work in region **us-west-2** unless told otherwise.
2. **Set a budget alert first.** Console → Billing and Cost Management → Budgets → *Create budget* → *Monthly cost budget*, $5, alert at 80%. Then check *Free Tier* on the same page to see what your account's free tier covers (AWS changed the free-tier rules in 2025, and new accounts get credits rather than the old 12-month allowance). The only part of this project that costs anything outside the free tier is the database: a `db.t4g.micro` with 20 GB of storage, roughly $15 a month at list price.
3. AWS credentials on your laptop: `aws login` (or `aws configure sso`). Check them with `aws sts get-caller-identity`.
4. [pixi](https://pixi.sh), git and [Claude Code](https://claude.com/claude-code). Every component has its own `pixi.toml`, so `pixi run <task>` is all you ever type to build, test or deploy.
5. Part 4 needs the hardware kit: a Raspberry Pi 4, SparkFun Qwiic HAT, SparkFun Qwiic Grid-EYE and a Qwiic cable. If you don't have one yet, do Part 4 in simulation (`--fake`), which is how it is developed anyway.

**Repository layout.** Make a repository with one folder per component, named exactly as below, so each spec's relative links (`../state-aws/...`) work:

```
occupancy/
  state-aws/        ingress-mock/        source-graphql/        ingress-pi/        analysis/
```

Commit after every step that works. A commit is your undo button when Claude goes in a direction you didn't want.

---

## How to work with Claude

Every component goes through the same loop. Learn the loop. The specific prompts below are just this loop filled in.

```
 spec ──▶ plan ──▶ build ──▶ verify ──▶ commit
  ▲                             │
  └──── fix the spec, not ──────┘
        just the code
```

1. **Spec.** A `requirements.md` that says what to build, what already exists and must not change, and how you'll know it's done (acceptance criteria). Claude builds what the spec says, so a vague spec gets you a vague result.
2. **Plan.** Ask for questions and a plan *before* any code. This is where you catch misunderstandings cheaply. (In Claude Code, *plan mode* (Shift+Tab) keeps it from editing files while you agree on the plan.)
3. **Build.** In small steps you can check, not one giant prompt.
4. **Verify.** Make Claude *show evidence* for each acceptance criterion: the command it ran and its output. "It should work" is not evidence.
5. **Commit**, then the next step.

**The four prompts you will reuse** (replace `<component>`):

**Plan**
```text
Read <component>/requirements.md. Don't write code yet. First list every decision the spec
leaves open or ambiguous and ask me about each, one at a time. Then propose a file-by-file
plan and tell me how you will test each piece without deploying.
```

**Build**
```text
Build <component> as planned, following requirements.md exactly. Work in small steps and
run the tests after each one. Don't deploy anything and don't touch other folders.
```

**Verify**
```text
Go through every acceptance criterion in <component>/requirements.md. For each, run the
command that checks it, show me the output, and mark it PASS or FAIL. Don't fix anything
in this pass; just report.
```

**Review**
```text
Review the code in <component> against requirements.md as a skeptical reviewer. List
requirements that are missing or only partly met, anything that could leak a secret, and
anything that would cost money while idle. Rank by severity. Don't change code yet.
```

**Rules that apply to every part**

- **Never paste a secret into the chat.** The deploy scripts keep keys in AWS SSM Parameter Store. Ask Claude to read them from there into a variable, never to print them.
- **Ask about cost before every deploy:** *"List every AWS resource this will create and what each costs per month outside the free tier."*
- **Deploying is your decision.** Read the plan, then run `pixi run deploy` yourself or tell Claude to.
- When something fails, paste the **exact error** and ask for the cause before the fix: *"Explain why this happened before changing anything."*

---

## Part 0: Read the contract

Get the reference repository and read `state-aws/requirements.md`. It's short, and everything else depends on it.

```text
Read state-aws/requirements.md. Explain to me, as if I'm new to databases:
(1) why there is no "count" column, only events;
(2) what goes wrong if a sensor retries a request and there is no event_id;
(3) why (location, event_id) is the key rather than event_id alone;
(4) why the table is insert-only. Then quiz me with three questions.
```

**Check:** you can answer, without notes, what happens when the same `POST /enter/lab/1790000000?id=a` arrives twice.

---

## Part 1: state-aws (the system of record)

**Goal:** your own API and private PostgreSQL database, created from nothing by one command.

Use the reference spec as it is. This part is mostly infrastructure, and the interesting decisions are already written down. Copy `state-aws/requirements.md` into your `state-aws/` folder, then:

1. **Plan** prompt. Expect questions about the VPC, the database size and the API key.
2. **Build** prompt, then one more that tests the code locally:
   ```text
   Start a local PostgreSQL with pixi, apply schema.sql, and exercise app.py's handler
   directly (no AWS): a new enter, the same id again, the same id at another location,
   a leave without an id, a future timestamp. Show me each response.
   ```
3. Cost question, then deploy (the first deploy creates the database and takes about 10 minutes):
   ```text
   List every AWS resource template.yaml creates and its monthly cost outside the free
   tier. Then run pixi run deploy and show me the output, without printing any secrets.
   ```
4. **Verify** prompt.

**Check yourself**

- [ ] `curl -X POST -H "x-api-key: $KEY" $URL/enter/test` returns 201, then `GET /occupancy/test` returns 1.
- [ ] The same `?id=` twice gives 201 then 200 with `"duplicate": true`.
- [ ] RDS console: the database is **not** publicly accessible.
- [ ] `git grep` finds no key or password in your repository.

**Think about it:** the API accepts timestamps in the past. What could a broken sensor do to your history with that? What would you add to detect it?

---

## Part 2: ingress-mock (a simulated sensor)

**Goal:** a web page that fills your database with realistic synthetic traffic, so Parts 3 and 5 have data before any hardware exists.

This time, **write the spec yourself**, then compare it with the reference:

```text
I'm building ingress-mock: a web page plus a small proxy Lambda that simulates an occupancy
sensor against the API in ../state-aws (read its requirements.md and README.md). It needs
manual Enter/Leave buttons, a live simulation, and a backfill of past days that can be
re-run without creating duplicates. Interview me one question at a time to write
ingress-mock/requirements.md in the same style as ../state-aws/requirements.md. Ask me
especially about: how arrivals are modelled, how event_ids are made so a re-run dedupes,
and how the API key stays out of the browser. Don't write code.
```

Then:

```text
Compare my ingress-mock/requirements.md with the reference spec at <path>. List what mine
is missing, where it differs, and which differences would change the data. Don't edit yet.
```

Decide which differences to adopt. **Plan → Build → Verify**, deploy, then backfill 30 days for two locations with different traffic profiles.

**Check yourself**

- [ ] Re-running the same backfill reports every event as a duplicate.
- [ ] The history chart shows the weekday/weekend pattern of the profile you chose.
- [ ] The API key is not in the page source or the browser's network tab.
- [ ] Every synthetic location starts with `mock-`.

**Think about it:** your backfill uses a Poisson process with an hourly rate. Name one way real arrivals at a classroom door are *not* Poisson. How would that show up in the data?

---

## Part 3: source-graphql (the read side)

**Goal:** a read-only GraphQL API over your database, and a script that pulls events into a Parquet file.

```text
Interview me to write source-graphql/requirements.md: a read-only GraphQL API over the
occupancy_event table, generated from the schema with PostGraphile V5, running on Lambda
next to the database from ../state-aws (read the stack outputs it publishes). It must log
in as the read-only occupancy_reader role. Include limits (page size, query depth), how
mutations are made impossible, and an example Python client that pages through all events
for a location and writes Parquet. Don't write code.
```

Compare with the reference spec, then **Plan → Build → Verify**. Ask for a local version first:

```text
Before anything touches AWS, get this running against a local PostgreSQL with seed data
(pixi run db / pixi run dev) and show me a query for the latest 5 events at one location.
```

**Check yourself**

- [ ] A mutation is rejected, and the schema has no `Mutation` type.
- [ ] `pixi run pull` fetches more than 5,000 events with no gaps or repeats (check `id`).
- [ ] The count from GraphQL matches `GET /occupancy/{location}` from Part 1.
- [ ] The explorer (`pixi run -e occupancy app` in the course repository, with `graphql_url` in its `config.json` pointed at *your* API) shows your mock locations, and its *Count now* matches too.

**Think about it:** why read through a separate role and API at all, instead of letting analysts connect to the database directly?

---

## Part 4: ingress-pi (the real sensor)

**Goal:** a Raspberry Pi that counts people through a real doorway, keeps everything in SQLite, and uploads on a cron job.

This is the biggest part, so build it in **four separate steps**, each with its own Plan → Build → Verify → commit. First write the spec:

```text
Interview me to write ingress-pi/requirements.md. It's a Raspberry Pi 4 with a SparkFun
Qwiic HAT and Qwiic Grid-EYE (AMG8833, I2C address 0x69) over a doorway. Two processes:
a sensor that runs all the time under systemd, reads 10 frames a second, counts people,
and stores frames and enter/leave events in SQLite; and an uploader that cron runs every
2 minutes to send unsent events to the API in ../state-aws with stable event ids. Ask me
about: what happens with no network, the Pi having no battery clock, where the API key is
kept, how long to keep raw frames, and how I'll develop it on a laptop without the
sensor. Don't write code.
```

Compare with the reference spec, then build in steps:

**Step 1: the sensor reader and storage, with a fake sensor**
```text
Build only the AMG8833 reader, the SQLite storage and a --fake mode that generates
realistic frames (warm people walking across a cool room), so I can run everything on my
laptop. Put the byte decoding in pure functions with unit tests. No counting yet.
```

**Step 2: the uploader**
```text
Build the uploader. Test it against a fake HTTP server running in the test itself:
success, duplicate (200), server error then retry on the next run, network down, a 400
that should not block the queue, a wrong key, and two uploads started at once.
```

**Step 3: the people counter**
```text
Build the people counter as a pure function: frames in, enter/leave events out, no I/O.
Use a learned background, warm blobs, and tracks that cross a midline. Test it on
synthetic walks: one enter, one leave, two people in a row, noise only, someone who
stops at the line and turns back. Then measure its accuracy on 30 minutes of --fake
traffic and tell me which situations it gets wrong.
```

**Step 4: on the Pi**
```text
Write install.sh for a fresh Raspberry Pi OS (64-bit): check I2C and the i2c group,
create the config and a 0600 key file, install a systemd service for the sensor and a
crontab line for the uploader. It must be safe to run twice. Then write the README steps
I follow from flashing the SD card to seeing my first event in the API.
```

Then on the Pi: `i2cdetect -y 1` should show `69`. Run `bash install.sh`, set `location` and the API URL, put the key in the key file, and walk through the doorway.

**Check yourself**

- [ ] `pixi run test` passes on your laptop.
- [ ] Unplug the Pi's network for 10 minutes while walking in and out, then plug it back in: every event arrives, none twice.
- [ ] `sudo reboot`: the sensor comes back by itself and waits for the clock before recording.
- [ ] The key is only in `/etc/ingress-pi/api-key` (mode 0600), not in the repo or the logs.

**Think about it:** the counter decides only when a track *ends*. What does that cost you in latency, and what does it buy in accuracy?

---

## Part 5: the analysis

**Goal:** a claim about a real room that you would be willing to act on, with the sensor's error measured rather than assumed.

1. **Collect ground truth.** For at least two one-hour sessions, someone sits by the door and clicks Enter/Leave on the mock page under `truth-<room>` while the Pi records `<room>`.
2. **Pull both** with the source-graphql client into Parquet. Look at them in the explorer first: the *Daily drift* and *Upload lag* charts often show what went wrong before any analysis does.
3. Analyse in a notebook or script in `analysis/`:

```text
Load events for <room> and truth-<room> from these Parquet files. For each session, compare
the two count curves minute by minute and report: missed enters, missed leaves, extra
events, and how far the count drifts by the end. Show where in time the errors happen.
Describe the error; don't correct the data yet.
```

```text
Using the raw frames stored on the Pi, replay the sessions where the counter was wrong
(pixi run replay). Find what those frames have in common. Propose one change to the
counter, test it with replay on ALL sessions, and report whether it helps overall or only
on the cases I picked.
```

**Questions to answer in your write-up**

- Over a day, how far does the Pi's count drift from the truth? Is it biased (always too high or too low), or just noisy?
- Is the error worse at busy times? Why would it be?
- Your backfilled `mock-` data and the real room: how is the real daily pattern different from the profile you simulated?
- If facilities asked "is this room over capacity at 10 am?", what would you answer, and how confident are you? (See *worth acting on* in `DATA-SCIENCE.md`.)

---

## Finishing

- **Hand in:** your repository (every `requirements.md`, the code, the analysis), your write-up, and a short log of the three prompts that most changed your result and why.
- **Tear down** at the end of the semester so nothing keeps billing: `pixi run teardown` in `state-aws/` deletes all the stacks and parameters and lists the final database snapshot (export what you need first, then delete it). Check the Billing console a few days later to make sure it says $0.
