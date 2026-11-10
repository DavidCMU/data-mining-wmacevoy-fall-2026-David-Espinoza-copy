"""
Compare embedding models and search modes on questions with known answers.

    python compare.py                          # every model, every mode
    python compare.py --models minilm bge-small --modes vector hybrid
    python compare.py --max-chars 800          # does smaller chunking help?
    python compare.py --details                # which questions each setup missed

Mirrors llm/compare.py (minus --pin: there's no principle-ID scheme to pin
on this corpus). questions.jsonl holds the test set: a question, the
section(s) that answer it, and a kind:

    paraphrase  says it in other words ("how do I grow a list" for "Storing
                Lists of Values with Vectors"): the case embeddings exist for
    keyword     uses the book's own term ("lifetime elision", "trait object")

For each question we retrieve chunks, collapse them to distinct sections, and
find the rank of the first section that answers it. Reported per setup:

    hit@1   fraction answered by the top section
    hit@5   fraction answered somewhere in the top five (what a RAG prompt with
            k=5 would contain)
    MRR     mean of 1/rank (0 past rank 10): rewards ranking the answer higher

Every run is appended to data/compare.csv, so you can compare runs with
different --max-chars, or after you add questions, later.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import chunks
import index
import models
import store

HERE = Path(__file__).parent
QUESTIONS = HERE / "questions.jsonl"
RESULTS = HERE / "data" / "compare.csv"
DEPTH = 10              # ranks past this count as a miss


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rank_of(rows: list[dict], expect: list[str]) -> int | None:
    """1-based rank of the first distinct section in `expect`, or None if not in the top DEPTH."""
    refs = list(dict.fromkeys(r["ref"] for r in rows))[:DEPTH]
    return next((i for i, ref in enumerate(refs, 1) if ref in expect), None)


def score(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {"hit@1": sum(r == 1 for r in ranks) / n,
            "hit@5": sum(r is not None and r <= 5 for r in ranks) / n,
            "MRR": sum(1 / r for r in ranks if r) / n}


def evaluate(table, key: str | None, mode: str, questions: list[dict]) -> list[dict]:
    """One row per question: its rank, and how long the search took."""
    if key and mode != "fts":
        models.embed_query(key, "warm up")  # the first call pays one-time setup; don't time it
    out = []
    for q in questions:
        t0 = time.perf_counter()
        vector = None if mode == "fts" else models.embed_query(key, q["q"])
        rows = store.search(table, mode, q["q"], vector, k=3 * DEPTH)
        out.append({**q, "rank": rank_of(rows, q["expect"]),
                    "ms": 1000 * (time.perf_counter() - t0)})
    return out


def summarize(results: list[dict]) -> dict:
    row = score([r["rank"] for r in results])
    for kind in ("paraphrase", "keyword"):
        ranks = [r["rank"] for r in results if r["kind"] == kind]
        if ranks:
            row[f"{kind} hit@5"] = score(ranks)["hit@5"]
    row["ms/query"] = sum(r["ms"] for r in results) / len(results)
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    p.add_argument("--models", nargs="+", default=list(models.MODELS), choices=list(models.MODELS))
    p.add_argument("--modes", nargs="+", default=list(store.MODES), choices=store.MODES)
    p.add_argument("--corpus", help="path to the book (default: see chunks.find_book)")
    p.add_argument("--max-chars", type=int, default=chunks.MAX_CHARS)
    p.add_argument("--details", action="store_true", help="list each setup's misses")
    args = p.parse_args(argv)

    questions = load_questions()
    root = chunks.find_book(args.corpus)
    chunk_list = chunks.chunk_book(root, args.max_chars)
    print(f"{len(questions)} questions, {len(chunk_list)} chunks (max {args.max_chars} chars)\n")

    db = store.connect()
    rows, details = [], []
    for key in args.models:
        timing = index.index_model(db, key, chunk_list, args.max_chars,
                                   index.book_version(root))
        table = store.open_table(db, store.table_name(key))
        print(f"indexed {key}: {timing['embed_s']:.1f}s, {timing['truncated']} truncated")
        for mode in args.modes:
            if mode == "fts" and any(r["mode"] == "fts" for r in rows):
                continue            # keyword search ignores the model; score it once
            results = evaluate(table, key, mode, questions)
            label = "(any)" if mode == "fts" else key
            rows.append({"model": label, "mode": mode, **summarize(results),
                         "embed s": None if mode == "fts" else timing["embed_s"],
                         "truncated": None if mode == "fts" else timing["truncated"]})
            details += [(label, mode, r) for r in results]

    df = pd.DataFrame(rows).sort_values("MRR", ascending=False)
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}", na_rep="-"))

    if args.details:
        for label, mode, r in details:
            if r["rank"] != 1:
                where = f"rank {r['rank']}" if r["rank"] else f"not in top {DEPTH}"
                print(f"{label:14s} {mode:7s} {where:15s} [{r['kind']}] {r['q']}")

    append_results(df, args.max_chars, len(chunk_list), len(questions))
    print(f"\nappended to {RESULTS}")


def append_results(df: pd.DataFrame, max_chars: int, n_chunks: int, n_questions: int) -> None:
    """Add summary rows to data/compare.csv, stamped with the settings that produced them."""
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    df.assign(when=datetime.now().isoformat(timespec="seconds"), max_chars=max_chars,
              chunks=n_chunks, questions=n_questions).to_csv(
        RESULTS, mode="a", header=not RESULTS.exists(), index=False)


if __name__ == "__main__":
    main()
