"""
The small local embedding models we compare, and how to call each one.

Every candidate here makes 384-number vectors, so the comparison is about the
model, not the vector size. They differ in size, training, how many tokens
they read before truncating, and whether they expect an instruction in front
of the text.

That last point is easy to get wrong. Retrieval models are often trained
*asymmetrically*: a short question and a long passage are embedded with
different prefixes ("query: " vs "passage: " for E5). Leave the prefix off and
the model still returns vectors, just worse ones. Nothing errors.

The first use of a model downloads it from Hugging Face into
~/.cache/huggingface (between ~90 and ~200 MB each); after that it runs offline.
"""
from dataclasses import dataclass

import numpy as np

BGE_QUERY = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class EmbeddingModel:
    key: str
    name: str               # Hugging Face model id
    params: str             # rough size, for the comparison table
    query_prefix: str = ""
    doc_prefix: str = ""
    note: str = ""


MODELS = {m.key: m for m in [
    EmbeddingModel("minilm", "sentence-transformers/all-MiniLM-L6-v2", "22M",
                   note="the classic baseline; reads only 256 tokens"),
    EmbeddingModel("bge-small", "BAAI/bge-small-en-v1.5", "33M", query_prefix=BGE_QUERY,
                   note="strong all-rounder; instruction on queries only"),
    EmbeddingModel("e5-small", "intfloat/e5-small-v2", "33M",
                   query_prefix="query: ", doc_prefix="passage: ",
                   note="prefixes on both sides"),
    EmbeddingModel("arctic-s", "Snowflake/snowflake-arctic-embed-s", "33M",
                   query_prefix=BGE_QUERY, note="tuned for retrieval"),
    EmbeddingModel("granite-small", "ibm-granite/granite-embedding-small-english-r2", "47M",
                   note="newest here (ModernBERT); reads 8,192 tokens"),
]}
DEFAULT = "e5-small"      # best MRR in compare.py on this guide; see README for the numbers

_loaded = {}


def load(key: str):
    """The SentenceTransformer for a model key, loaded once per process."""
    if key not in _loaded:
        from sentence_transformers import SentenceTransformer  # slow import; only when needed
        from transformers.utils import logging
        logging.disable_progress_bar()      # the "Loading weights" bar, on every load
        _loaded[key] = SentenceTransformer(MODELS[key].name)
    return _loaded[key]


def embed_documents(key: str, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Unit-length float32 vectors, one row per text, so cosine similarity is a dot product."""
    m = MODELS[key]
    return load(key).encode([m.doc_prefix + t for t in texts], batch_size=batch_size,
                            normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def embed_query(key: str, text: str) -> np.ndarray:
    m = MODELS[key]
    return load(key).encode(m.query_prefix + text, normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32)


def n_truncated(key: str, texts: list[str]) -> int:
    """How many texts are longer than the model reads. The rest of each is silently ignored."""
    model = load(key)
    limit = model.max_seq_length
    prefix = MODELS[key].doc_prefix
    return sum(len(model.tokenizer(prefix + t, verbose=False)["input_ids"]) > limit for t in texts)
