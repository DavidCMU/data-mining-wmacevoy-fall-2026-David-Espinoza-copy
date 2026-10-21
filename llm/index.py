"""
Chunk the guide, embed every chunk, and write a LanceDB table per model.

    python index.py                          # the default model (e5-small)
    python index.py --model bge-small
    python index.py --model all              # every model in models.py
    python index.py --corpus ~/src/software-engineering --max-chars 800

Re-run it whenever the guide changes; it rebuilds the table from scratch.
That takes seconds here, so there is nothing to gain from updating in place.
"""
import argparse
import time

import chunks
import models
import store


def index_model(db, key: str, chunk_list: list[chunks.Chunk]) -> dict:
    """Embed and store one model's table. Returns timings for compare.py."""
    t0 = time.perf_counter()
    models.load(key)
    t1 = time.perf_counter()
    texts = [c.text for c in chunk_list]
    vectors = models.embed_documents(key, texts)
    t2 = time.perf_counter()
    store.build(db, store.table_name(key), chunk_list, vectors)
    t3 = time.perf_counter()
    return {"load_s": t1 - t0, "embed_s": t2 - t1, "store_s": t3 - t2,
            "truncated": models.n_truncated(key, texts)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    p.add_argument("--model", default=models.DEFAULT, choices=[*models.MODELS, "all"])
    p.add_argument("--corpus", help="path to the guide (default: see chunks.find_guide)")
    p.add_argument("--max-chars", type=int, default=chunks.MAX_CHARS)
    args = p.parse_args(argv)

    root = chunks.find_guide(args.corpus)
    chunk_list = chunks.chunk_guide(root, args.max_chars)
    print(f"{root}: {len(chunk_list)} chunks")

    db = store.connect()
    keys = list(models.MODELS) if args.model == "all" else [args.model]
    for key in keys:
        t = index_model(db, key, chunk_list)
        print(f"{key:14s} -> {store.table_name(key)}: embedded in {t['embed_s']:.1f}s, "
              f"{t['truncated']} of {len(chunk_list)} chunks truncated")
    print(f"database: {store.DB_DIR}")


if __name__ == "__main__":
    main()
