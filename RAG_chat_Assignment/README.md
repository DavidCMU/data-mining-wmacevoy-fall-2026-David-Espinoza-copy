# A RAG Lab over The Rust Programming Language

Turn [rust-lang/book](https://github.com/rust-lang/book) (the free, open-source
"Rust Book") into a searchable vector database, then chat with it as a
knowledge base, tuning both by measurement. Mirrors the `llm/` example's
architecture — vendor the corpus, chunk it, embed it, index it, measure
retrieval, then build a conversation on top — applied to a different corpus
and a plain knowledge-base chat instead of `llm/`'s mentor-bot-plus-judge
system.

```
app.py            Streamlit dashboard: retrieval, and a chat with the book
vendor.py         copy the book's source into vendor/ (see below)
chunks.py         Markdown -> one chunk per heading (split further only if long)
models.py         the candidate embedding models, and their query/passage prefixes
store.py          LanceDB: one table per model; vector, keyword (fts), or hybrid search
index.py          chunk -> embed -> write data/lancedb/chunks_<model>
ask.py            search, or print the prompt a RAG system would send an LLM
compare.py        score every model x search mode on questions.jsonl
chat.py           one chat exchange: retrieve, then ask Claude to answer only from that
questions.jsonl   45 questions with the sections that answer them
```

**RAG** (retrieval-augmented generation) means: before asking an LLM a
question, look up the passages that answer it and paste them into the
prompt. The retrieval half is measured directly (`compare.py`); the
generation half is what `chat.py` and the dashboard's Chat page do with it.
`ask.py --prompt` prints the finished prompt, so you can paste it into any
chat model and watch the whole loop by hand.

## The dashboard

```bash
pixi run -e rag app          # opens http://localhost:8501
```

| Page | What you do there |
| --- | --- |
| **Retrieval** | Build an index per model and chunk size; search three ways side by side; score a setup on the test set and see every miss; compare scores over time; edit the test set |
| **Chat** | Ask the book questions; each reply cites the sections it was given, and an expander under the reply shows exactly what was retrieved |

Chat calls the Claude API, which needs a key from
[console.anthropic.com](https://console.anthropic.com). Put it in `.env`
(git-ignored):

```
ANTHROPIC_API_KEY=sk-ant-...
```

or export it before starting the app. A chat message costs well under a
cent with the default model (Sonnet); the dashboard shows the running bill.

## Running it

From the **repository root**:

```bash
pixi run -e rag index                      # first run downloads e5-small (~130 MB)
pixi run -e rag ask "how do I grow a list and read its elements back out"
```

| Task | What it does |
| --- | --- |
| `pixi run -e rag index` | Build the table for the default model (`--model all` for every model) |
| `pixi run -e rag app` | Launch the dashboard |
| `pixi run -e rag ask "<question>"` | Top 5 sections (`--mode fts\|hybrid`, `--model`, `-k`, `--prompt`) |
| `pixi run -e rag compare` | Index every model and score it on `questions.jsonl` |
| `pixi run -e rag test` | Run the tests (offline, about a second) |
| `pixi run -e rag lint` | Check code style |
| `pixi run -e rag vendor` | Update the vendored book to upstream `main` (`--ref <tag>` to pin a release, `--check` to compare only) |
| `pixi run -e rag clean` | Delete `data/` (the database and results) |

The book is **vendored**: a copy of its `src/` directory lives in
`vendor/rust-book/src/`, and `vendor/rust-book.json` records the commit it
came from (committed in this repo at
[rust-lang/book@2771df1](https://github.com/rust-lang/book/commit/2771df165f092a4fbb9f46b5b2e7450f19010adb),
2026-10-28). A fresh clone indexes the same text every time, with no network
and no second checkout. Updating is a deliberate step: `pixi run -e rag
vendor` prints which files changed; re-index, run the tests (one checks
that every `questions.jsonl` answer still exists), and commit. To work
against a live checkout instead, set `$RUST_BOOK` or pass `--corpus`. Models
are cached in `~/.cache/huggingface`, so after the first download everything
runs offline.

## Why LanceDB

It is an **embedded** database, like SQLite: `lancedb.connect("data/lancedb")`
opens a folder, and there is no server to install or keep running. It stores
vectors next to ordinary columns (`ref`, `heading`, `file`), so one query
can mix similarity with a SQL-style `where` filter. It also has BM25 keyword
search built in, which matters for a book full of exact terms (`Rc<T>`,
`RefCell`, `?` operator).

With about 1,100 chunks, LanceDB simply compares the question against every
vector (a few milliseconds). An approximate index such as IVF_PQ or HNSW is
for around 100,000 rows and up.

## Chunking

The book is one Markdown page per section, each opening with a `#` title and
then `##`/`###` subsections — e.g. `ch08-01-vectors.md` opens with `##
Storing Lists of Values with Vectors` and then `### Creating a New Vector`.
A heading is what a person would cite, so each heading is a chunk, and only
sections longer than `--max-chars` (default 1500, roughly 350 tokens) are
split by paragraph, then by line. Glossary-style tables repeat their header
row in every piece. Each chunk is embedded as its heading path plus its
body:

```
Storing Lists of Values with Vectors > Creating a New Vector

Use Vec::new() to make an empty vector. ...
```

Three cleanups changed the results, and all are worth knowing about:

- **Link targets are removed.** `[other installation methods][otherinstall]`
  (and the plain-URL `(url)` form) becomes `[other installation methods]`,
  and the dangling reference-definition line (`[otherinstall]: https://...`)
  is dropped entirely. Left in, link targets are pure noise to an embedding
  and inflate keyword-search token counts for no benefit.
- **`<!-- ignore -->` comments are removed.** mdbook (the book's own build
  tool) uses these to mark a code block that shouldn't be compiled as a
  doctest; a model gets nothing from them.
- **Table padding is removed.** The spaces that line columns up in an editor
  cost tokens and carry no meaning.

Unlike `llm/`'s software-engineering guide, the Rust book has no numbered
principle-ID scheme (no `SE1/T3`-style citations to pin by exact match), so
there is no `store.pin_principles` equivalent here — every question is
answered by vector, fts, or hybrid search alone.

## Search modes

| Mode | Good at | Bad at |
| --- | --- | --- |
| `vector` | paraphrases: "a way to grow a list" -> *Storing Lists of Values with Vectors* | exact tokens: to a model, "`Rc<T>`" means little |
| `fts` | exact words and identifiers, rare terms | synonyms; any question phrased differently from the text |
| `hybrid` | merges both rankings (reciprocal rank fusion) | gives keyword noise as much weight as the vector match |

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

Run `pixi run -e rag compare` yourself and fill in this table (the
software-engineering guide's own numbers, from `llm/README.md`, are **not**
a safe prediction here — this corpus is ~4.5x larger, has no principle-ID
scheme to pin, and is full of code-identifier-style terms like `Vec<T>` that
a general-purpose embedding model has never seen written that way):

| Model | Mode | hit@1 | hit@5 | MRR |
| --- | --- | --- | --- | --- |
| | | | | |
| | | | | |
| | | | | |

hit@k is the fraction of questions answered within the top k sections; MRR
is the mean of 1/rank. 45 questions is still a small test set, so treat gaps
of a few points as ties, and read `--details` before believing a winner.

## Experiments to try

Each run is appended to `data/compare.csv`, so you can compare runs later.

1. **Grow the test set.** Add questions in your own words to
   `questions.jsonl` (the **Test set** tab in the dashboard, or by hand). A
   test checks that every expected `ref` exists.
2. **Chunk size.** Rebuild at 800 and 2500 characters and compare to 1500.
3. **Prefixes off.** Strip `query_prefix`/`doc_prefix` from e5-small in
   `models.py` and measure the drop.
4. **A reranker.** Retrieve 20 chunks, then reorder them with a
   cross-encoder (`cross-encoder/ms-marco-MiniLM-L6-v2`), and see whether
   hit@1 rises.
5. **k = 0 as a control, in Chat.** Set k to 0 so Claude answers with no
   retrieved context at all. Ask something specific to this edition of the
   book (a function name, an exact code example) and see whether it
   confabulates versus says it doesn't know.
