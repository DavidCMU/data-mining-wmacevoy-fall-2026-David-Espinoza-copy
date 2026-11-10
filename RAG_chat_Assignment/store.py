"""
The LanceDB side: write chunks and their vectors to a table, and search it.

Mirrors llm/store.py. LanceDB is an *embedded* database, like SQLite:
`lancedb.connect(path)` opens a folder, and there is no server to start.
Each embedding model gets its own table (chunks_bge_small, chunks_minilm,
...) because vectors from different models live in different spaces and
cannot be compared.

Three ways to search the same table:

    vector   embed the question, return the nearest chunks by cosine distance.
             Finds paraphrases ("how do I grow a list" -> "Storing Lists of
             Values with Vectors"); can miss exact tokens like a specific
             trait or macro name.
    fts      full-text (BM25) keyword search over the chunk text. The opposite
             strengths: exact identifiers and rare words, but no synonyms.
    hybrid   run both and merge the rankings (reciprocal rank fusion).

With about a thousand rows, LanceDB simply compares the question against
every vector (a few milliseconds). An approximate index (IVF_PQ, HNSW) only
starts to pay for itself around 100,000 rows, so we do not build one.

Unlike llm/store.py, there is no `pin_principles`: the software-engineering
guide has explicit principle IDs (SE1/T3) worth pinning by exact match; the
Rust book has no such ID scheme, so every question is answered by vector,
fts, or hybrid search alone.
"""
import os

os.environ.setdefault("LANCEDB_LOG", "error")  # hush "No existing dataset ... will be created"

from pathlib import Path  # noqa: E402

import lancedb  # noqa: E402
import numpy as np  # noqa: E402
from lancedb.index import FTS  # noqa: E402

from chunks import Chunk  # noqa: E402

DB_DIR = Path(__file__).parent / "data" / "lancedb"
MODES = ("vector", "fts", "hybrid")


def table_name(model_key: str) -> str:
    return "chunks_" + model_key.replace("-", "_")


def connect(db_dir: Path = DB_DIR):
    return lancedb.connect(str(db_dir))


def build(db, name: str, chunks: list[Chunk], vectors: np.ndarray):
    """(Re)create a table from chunks and their vectors, with a full-text index on `text`."""
    rows = [{"ref": c.ref, "file": c.file, "heading": c.heading,
             "part": c.part, "text": c.text, "vector": v} for c, v in zip(chunks, vectors)]
    table = db.create_table(name, data=rows, mode="overwrite")
    table.create_index("text", config=FTS(), replace=True)
    return table


def open_table(db, name: str):
    """The table, or a FileNotFoundError that says how to build it."""
    try:
        return db.open_table(name)
    except (FileNotFoundError, ValueError):
        model = name.removeprefix("chunks_").replace("_", "-")
        raise FileNotFoundError(
            f"no table {name!r} yet -- run: pixi run -e rag index --model {model}") from None


def search(table, mode: str, text: str, vector: np.ndarray | None = None,
           k: int = 5) -> list[dict]:
    """Top-k rows, best first. Each row gets a `score` where higher is better."""
    if mode == "vector":
        rows = table.search(vector).distance_type("cosine").limit(k).to_list()
        for r in rows:
            r["score"] = 1 - r["_distance"]          # cosine similarity
    elif mode == "fts":
        rows = table.search(text, query_type="fts").limit(k).to_list()
        for r in rows:
            r["score"] = r["_score"]                 # BM25
    elif mode == "hybrid":
        rows = (table.search(query_type="hybrid").vector(vector).text(text)
                .distance_type("cosine").limit(k).to_list())
        for r in rows:
            r["score"] = r["_relevance_score"]       # reciprocal rank fusion
    else:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    return [{key: v for key, v in r.items() if key != "vector"} for r in rows]
