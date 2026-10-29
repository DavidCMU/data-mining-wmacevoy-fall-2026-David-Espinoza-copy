# A RAG Lab over the Software Engineering Guide

Turn [wmacevoy/software-engineering](https://github.com/wmacevoy/software-engineering)
into a searchable database, then build a mentor bot on top of it, and tune both
by measurement. The lab has two phases:

1. **Retrieval** (local, free): which embedding model, search mode and chunk
   size find the right section most often?
2. **Conversation** (Claude API): Zen Coder, a mentor for novice developers,
   knows the guide. Which system prompt, and how many retrieved sections per
   turn, make it a better mentor, as scored by a judge model?

```
app.py            Streamlit dashboard: both phases, and a chat with Zen Coder
vendor.py         copy the guide into vendor/ (see below)
chunks.py         Markdown → one chunk per heading (split further only if long)
models.py         the candidate embedding models, and their query/passage prefixes
store.py          LanceDB: one table per model; vector, keyword (fts), or hybrid search
index.py          chunk → embed → write data/lancedb/chunks_<model>
ask.py            search, or print the prompt a RAG system would send an LLM
compare.py        score every model × search mode on questions.jsonl
questions.jsonl   40 questions with the sections that answer them
zen/              phase 2: Zen Coder, its test conversations, and the judge (zen/README.md)
```

**RAG** (retrieval-augmented generation) means: before asking an LLM a
question, look up the passages that answer it and paste them into the prompt.
Phase 1 is the retrieval half. `ask.py --prompt` prints the finished prompt,
so you can paste it into any chat model and watch the whole loop. Phase 2 is
the generation half, measured.

## The dashboard

```bash
pixi run -e llm app          # opens http://localhost:8501
```

| Page | What you do there |
| --- | --- |
| **1 · Retrieval** | Build an index per model and chunk size; search three ways side by side; score a setup on the test set and see every miss; compare scores over time; edit the test set |
| **2 · Conversation** | Start a run of test conversations (prompts × k × scenarios × repeats); watch it live; read transcripts with what Zen retrieved each turn; compare scores by prompt and k, and across runs; edit prompts and scenarios |
| **Chat with Zen Coder** | Talk to it yourself, see what it retrieved, and have the judge score your conversation |

Phase 2 and Chat call the Claude API, which needs a key from
[console.anthropic.com](https://console.anthropic.com). Put it in `llm/.env`,
which git ignores:

```
ANTHROPIC_API_KEY=sk-ant-...
```

or export it before starting the app. A conversation costs about $0.15 with
the default models (Sonnet for Zen Coder, Haiku for the novice, Opus for the
judge), so a 12-conversation run is about $2. The dashboard shows an estimate
before you start and the running bill after; the prices it uses are in
`zen/coder.py`.

## Running it

From the **repository root**:

```bash
pixi run -e llm index                      # first run downloads e5-small (~130 MB)
pixi run -e llm ask "how do I handle a flaky bug I cannot reproduce"
```

| Task | What it does |
| --- | --- |
| `pixi run -e llm index` | Build the table for the default model (`--model all` for every model) |
| `pixi run -e llm app` | Launch the dashboard |
| `pixi run -e llm ask "<question>"` | Top 5 sections (`--mode fts\|hybrid`, `--model`, `-k`, `--prompt`) |
| `pixi run -e llm compare` | Index every model and score it on `questions.jsonl` |
| `pixi run -e llm test` | Run the tests (offline, about a second) |
| `pixi run -e llm lint` | Check code style |
| `pixi run -e llm zen-run --run <name>` | A phase 2 run from the command line (`--help`) |
| `pixi run -e llm zen-score <name>` | Summarize a phase 2 run |
| `pixi run -e llm vendor` | Update the vendored guide to upstream `main` (`--ref v1.1` to pin a tag, `--check` to compare only) |
| `pixi run -e llm clean` | Delete `data/` (the database and results) |

The guide is **vendored**: a copy lives in `vendor/software-engineering/`, and
`vendor/software-engineering.json` records the commit it came from. A fresh
clone indexes the same text every time, with no network and no second
checkout. Updating is a deliberate step: `pixi run -e llm vendor` prints which
files changed; re-index, run the tests (one checks that `zen/guide-summary.md`
still lists every principle), and commit. To work against a live checkout
instead, set `$SE_GUIDE` or pass `--corpus`. Models are cached in
`~/.cache/huggingface`, so after the first download everything runs offline.

> Pixi's task runner can't parse an apostrophe inside a quoted argument
> (`"I can't"`). Rephrase, or call the script directly:
> `pixi run -e llm python llm/ask.py "..."`.

## Why LanceDB

It is an **embedded** database, like SQLite: `lancedb.connect("data/lancedb")`
opens a folder, and there is no server to install or keep running. It stores
vectors next to ordinary columns (`ref`, `heading`, `principle`), so one query
can mix similarity with a SQL-style `where` filter. It also has BM25 keyword
search built in, which matters for a guide full of exact terms and IDs.

With 246 chunks, LanceDB simply compares the question against every vector
(about a millisecond). An approximate index such as IVF_PQ or HNSW is for
around 100,000 rows and up.

## Chunking

The guide is organized as numbered principles, one heading each, and a
heading is what a person would cite. So each `##`/`###` section is a chunk,
and only sections longer than `--max-chars` (default 1500, roughly 350 tokens)
are split by paragraph, then by line. Glossary tables repeat their header row
in every piece. Each chunk is embedded as its heading path plus its body:

```
7. Operating > SE1/O8. Secure by default

Deny by default. ...
```

Two cleanups changed the results, and both are worth knowing about:

- **Link targets are removed.** `[SE1/O8](07-operating.md#o8)` becomes
  `[SE1/O8]`. With the targets left in, keyword search counted every
  cross-reference as an extra mention of `o8`, and four other sections
  outranked the one that defines O8.
- **Table padding is removed.** The spaces that line columns up in an editor
  cost tokens and carry no meaning.

## Search modes, and asking for a principle by ID

| Mode | Good at | Bad at |
| --- | --- | --- |
| `vector` | paraphrases: "a bug that only shows up sometimes" → T9 | exact tokens: to a model, "O8" means nothing |
| `fts` | exact words and IDs, rare terms | synonyms; any question phrased differently from the text |
| `hybrid` | merges both rankings (reciprocal rank fusion) | gives keyword noise as much weight as the vector match |

When a question names a principle (`SE1/O8`, or a bare capitalized `D10`),
`store.pin_principles` looks it up with an exact `where principle IN (...)`
filter and puts it first. That handles IDs reliably in every mode, which
neither embeddings nor BM25 did on their own. `compare.py --pin` turns it on;
without the flag you see how each raw mode copes with IDs.

## Which model? The experiment

All five candidates produce 384-number vectors, so the comparison is about
the model itself, not the vector size:

| Key | Model | Size | Prefixes |
| --- | --- | --- | --- |
| `minilm` | all-MiniLM-L6-v2 | 22M | none; **reads only 256 tokens** |
| `bge-small` | BAAI/bge-small-en-v1.5 | 33M | instruction on queries |
| `e5-small` | intfloat/e5-small-v2 | 33M | `query: ` / `passage: ` |
| `arctic-s` | Snowflake/snowflake-arctic-embed-s | 33M | instruction on queries |
| `granite-small` | ibm-granite/granite-embedding-small-english-r2 | 47M | none; reads 8,192 tokens |

Getting the prefixes wrong produces no error, just worse vectors. To see how
much they matter, edit `models.py` to remove them and rerun `compare`.

`pixi run -e llm compare --pin` on an M-series Mac, October 2026, 1500-char chunks:

| Model | Mode | hit@1 | hit@5 | MRR | paraphrase hit@5 | keyword hit@5 |
| --- | --- | --- | --- | --- | --- | --- |
| **e5-small** | **vector** | **0.65** | **0.95** | **0.76** | 0.93 | 1.00 |
| arctic-s | vector | 0.65 | 0.85 | 0.74 | 0.83 | 0.83 |
| arctic-s | hybrid | 0.65 | 0.85 | 0.73 | 0.80 | 1.00 |
| e5-small | hybrid | 0.65 | 0.82 | 0.72 | 0.77 | 1.00 |
| bge-small | vector | 0.57 | 0.93 | 0.72 | 0.97 | 0.67 |
| minilm | vector | 0.60 | 0.88 | 0.70 | 0.87 | 0.83 |
| granite-small | vector | 0.57 | 0.85 | 0.68 | 0.83 | 0.83 |
| (any) | fts | 0.50 | 0.75 | 0.62 | 0.67 | 1.00 |

hit@k is the fraction of questions answered within the top k sections; MRR is
the mean of 1/rank. What the table shows:

- **e5-small with vector search is the default.** It has the best MRR and
  hit@5, and it is no slower than the others (about 10 ms per query).
  bge-small is a close second and the best on paraphrases alone.
- **Hybrid search did not help once IDs were pinned.** Equal-weight fusion
  lets keyword matches on words like "client" or "project" push out the right
  section. It does win on keyword questions.
- **MiniLM truncates 115 of 246 chunks** (anything past 256 tokens is
  ignored), and it still isn't last. On text this clean, model size matters
  less than you might expect.
- **Chunk size barely mattered.** At 800 and 2500 characters, e5-small scored
  MRR 0.74 and 0.72, against 0.76 at 1500.

Forty questions is a small test set: each question is 2.5 points of hit@1, so
treat gaps of a few points as ties. `--details` lists every miss. Several
questions are missed by every setup ("Production is down. What do I do
first?"), which says more about the question set or the chunking than about
any one model.

## Experiments to try

Each run is appended to `data/compare.csv`, so you can compare runs later.

1. **Grow the test set.** Add questions in your own words to
   `questions.jsonl`. A test checks that every expected `ref` exists.
2. **Weighted hybrid.** Replace RRF with
   `lancedb.rerankers.LinearCombinationReranker(weight=0.8)` in `store.search`
   and find the weight that beats plain vector search.
3. **Prefixes off.** Strip `query_prefix`/`doc_prefix` from e5-small and
   measure the drop.
4. **A reranker.** Retrieve 20 chunks, then reorder them with a cross-encoder
   (`cross-encoder/ms-marco-MiniLM-L6-v2`), and see whether hit@1 rises.
5. **Index pages.** README's "Principle index" chunks appear in many result
   lists. Exclude them and see whether anything gets worse.
