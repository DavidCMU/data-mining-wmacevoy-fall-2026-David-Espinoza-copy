# Zen Coder test harness

Zen Coder is a toy mentor bot: it finds out what a novice developer wants to
build and recommends good ways to build it, and why. This harness measures
how two choices change its conversations:

- **the system prompt** (`prompts.json`: `baseline`, `socratic`, `principle`, `door`)
- **k**, how many guide sections are retrieved into each turn (0 is the no-RAG control)

```
guide-summary.md   the whole guide condensed to ~2,800 tokens: what Zen Coder knows
prompts.json       the system prompts under test (edit them on the dashboard)
scenarios.json     the simulated developers: opening, persona, hidden facts, relevant sections
coder.py           one Zen turn, the simulated novice, the judge, the metrics, the bill
run.py             a batch of conversations, judged, written to data/zen/<run>/
score.py           summarize a run by prompt and k
retrieve.py        the retrieval step as a command (used by the workflow)
zen.workflow.js    the same harness as a Claude Code workflow, for running without an API key
```

## What Zen Coder knows, and what it shows

Zen Coder reads `guide-summary.md` before every reply. It knows the guide but
never cites it: no principle IDs, no quotations, no "the guide says." Its job
is to lead the developer through the door rather than point at the sign. It
asks the question, or sets the small step, that lets the developer discover the
principle for themselves. Retrieved sections add depth on the current question
and stay private in the same way.

## How a conversation runs

Each turn, Zen Coder (Sonnet) gets its system prompt and the summary (cached,
so after the first call they bill at the cache-read rate), the top k guide
sections, and the conversation so far. It replies with JSON:
`{response, query}`. Turn 1 retrieves for the developer's opening message;
every later turn retrieves for the `query` Zen Coder asked for the turn
before. A simulated novice (Haiku) answers. It holds a few hidden facts, such
as platform, experience, and what "done" means, and reveals each one only when
asked. After three Zen turns, an Opus judge scores the transcript.

## Running it

From the dashboard (`pixi run -e llm app`, page **2 · Conversation**), or:

```bash
pixi run -e llm zen-run --run 2026-10-23a --prompts baseline door --ks 0 3 \
    --scenarios alarm plant --repeats 2
pixi run -e llm zen-score 2026-10-23a
pixi run -e llm zen-score 2026-10-23a --by scenario k --verdicts
```

It needs `ANTHROPIC_API_KEY` (in `llm/.env`, or exported). Each conversation
is six API calls (three Zen turns, two novice turns, one judge) and costs
about $0.15 with the default models. Four conversations run at a time
(`--workers`). Re-running a name resumes it: conversations that already have
a judge file are skipped. Scenarios: `alarm`, `club`, `grades`, `plant`. Also
accepted: `--turns`, `--zen-model`, `--novice-model`, `--judge-model`
(`sonnet`, `haiku`, `opus`, or a model id), `--retrieval-model`, and
`--retrieval-mode`.

`config.json` in the run directory records the arguments and the exact text of
each prompt as run, so editing a prompt later doesn't rewrite history.

**Without an API key**, the same harness runs as a Claude Code workflow. Ask
Claude Code:

> Run the workflow at llm/zen/zen.workflow.js with args
> `{"run": "2026-10-22a", "prompts": ["baseline", "principle"], "ks": [0, 3, 5], "scenarios": ["alarm", "plant"]}`

The workflow has its own copy of the prompts and scenarios, as they were for
runs 2026-10-22a to c, because a workflow script can't read files. Runs 2026-10-22a to c
were made this way, and the dashboard shows them alongside API runs.

## How quality is judged

**Automatic metrics.** These are computed in the workflow and are free and objective:

| Metric | Meaning |
| --- | --- |
| `relevant_recall` | Share of the scenario's hand-picked guide sections that Zen Coder's queries retrieved |
| `guide_leaks` | Principle IDs, or mentions of "the guide" or retrieval, shown to the developer. It should be 0 |
| `questions_per_turn` | The prompt asks for at most two |
| `words_per_turn` | Concision |

**Judge scores.** Opus scores each criterion 1–5, writing out its evidence first:

- **understands:** asks the questions that matter, one or two at a time
- **fits:** recommends the simplest workable path for this novice's constraints
- **teaches:** leads the novice to discover why, through a question or a small step, instead of lecturing them
- **grounded:** the advice is consistent with the guide, which the judge also reads; showing an ID or the guide itself is a serious fault
- **tone:** calm, concise, not preachy

The judge also lists which hidden facts the novice revealed (`facts`) and the
first turn with a concrete next step (`first_rec`).

Treat small differences as noise. One conversation per configuration is an
anecdote, so use `repeats` before drawing conclusions. Before trusting the
judge, score a handful of transcripts yourself and check that the judge agrees
with you.
