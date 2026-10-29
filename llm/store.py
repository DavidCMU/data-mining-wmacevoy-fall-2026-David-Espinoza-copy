"""
The LanceDB side: write chunks and their vectors to a table, and search it.

LanceDB is an *embedded* database, like SQLite: `lancedb.connect(path)` opens
a folder, and there is no server to start. Each embedding model gets its own
table (chunks_bge_small, chunks_minilm, ...) because vectors from different
models live in different spaces and cannot be compared.

Three ways to search the same table:

    vector   embed the question, return the nearest chunks by cosine distance.
             Finds paraphrases ("flaky bug I can't trigger" -> "a bug you cannot
             reproduce is a rumor"); can miss exact tokens like SE1/O8.
    fts      full-text (BM25) keyword search over the chunk text. The opposite
             strengths: exact IDs and rare words, but no synonyms.
    hybrid   run both and merge the rankings (reciprocal rank fusion).

With a few hundred rows LanceDB compares the question against every vector,
which takes well under a millisecond. An approximate index (IVF_PQ, HNSW)
only starts to pay for itself around 100,000 rows, so we do not build one.
"""
import os

os.environ.setdefault("LANCEDB_LOG", "error")  # hush "No existing dataset ... will be created"

from pathlib import Path  # noqa: E402

import lancedb  # noqa: E402
import numpy as np  # noqa: E402
from lancedb.index import FTS  # noqa: E402

from chunks import Chunk, principles_in  # noqa: E402

DB_DIR = Path(__file__).parent / "data" / "lancedb"
MODES = ("vector", "fts", "hybrid")


def table_name(model_key: str) -> str:
    return "chunks_" + model_key.replace("-", "_")


def connect(db_dir: Path = DB_DIR):
    return lancedb.connect(str(db_dir))


def build(db, name: str, chunks: list[Chunk], vectors: np.ndarray):
    """(Re)create a table from chunks and their vectors, with a full-text index on `text`."""
    rows = [{"ref": c.ref, "file": c.file, "heading": c.heading, "principle": c.principle,
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
            f"no table {name!r} yet -- run: pixi run -e llm index --model {model}") from None


def search(table, mode: str, text: str, vector: np.ndarray | None = None,
           k: int = 5, pin: bool = True) -> list[dict]:
    """Top-k rows, best first. Each row gets a `score` where higher is better.
    With pin, a principle the question names by ID comes first (see pin_principles)."""
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
    if pin:
        rows = pin_principles(table, text, rows, k)
    return [{key: v for key, v in r.items() if key != "vector"} for r in rows]


def pin_principles(table, text: str, rows: list[dict], k: int) -> list[dict]:
    """If the question names a principle ("what does SE1/O8 say?"), put that
    principle's own section first, found by an exact metadata filter.

    Neither search mode does this reliably: to an embedding, "O8" is a meaningless
    token; to BM25, every chunk that *cites* O8 competes with the one that defines
    it. A `where` clause on a column we stored for the purpose is exact and cheap.
    """
    ids = principles_in(text)
    if not ids:
        return rows
    listed = ", ".join(f"'{i}'" for i in ids)
    pinned = table.search().where(f"principle IN ({listed}) AND part = 0").to_list()
    for r in pinned:
        r["score"] = float("inf")
    seen = {r["ref"] for r in pinned}
    return (pinned + [r for r in rows if not (r["ref"] in seen and r["part"] == 0)])[:k]
