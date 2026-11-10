"""
Offline tests: no model downloads, no real LanceDB server (it's embedded,
but we still don't want a test suite that writes to disk every run).
Vectors are made by hand, so these check the chunking, the LanceDB
plumbing and the scoring, not any model's quality. That is compare.py's job.

Mirrors llm/tests/test_retrieval.py.
"""
from pathlib import Path

import numpy as np
import pytest

import chunks
import compare
import store

BOOK = """\
# 8. Collections

Intro to the chapter.

## Storing Lists of Values with Vectors

Background.

### Creating a New Vector

Use `Vec::new()` to make an empty vector. See [other methods][otherinstall].

```rust
# fn main() {
let v: Vec<i32> = Vec::new();
# }
```

### Reading Elements of Vectors

Index with `[]` or use `.get()`.

[otherinstall]: https://doc.rust-lang.org/std/vec/struct.Vec.html
"""


@pytest.fixture
def book(tmp_path) -> Path:
    (tmp_path / "book").mkdir()
    (tmp_path / "book" / "ch08-01-vectors.md").write_text(BOOK)
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "hidden.md").write_text("# skip me\n\ntext\n")
    return tmp_path / "book"


def test_chunks_follow_headings_and_build_anchors(book):
    cs = chunks.chunk_book(book)
    assert [c.ref for c in cs] == [
        "ch08-01-vectors.md",
        "ch08-01-vectors.md#storing-lists-of-values-with-vectors",
        "ch08-01-vectors.md#creating-a-new-vector",
        "ch08-01-vectors.md#reading-elements-of-vectors",
    ]
    new_vector = cs[2]
    assert new_vector.heading == "8. Collections > Storing Lists of Values with Vectors > Creating a New Vector"
    assert "# fn main() {" in new_vector.body      # fenced code is not split on as a heading
    assert "[other methods]" in new_vector.body and "doc.rust-lang.org" not in new_vector.body
    assert "otherinstall]:" not in new_vector.body  # the bare reference-definition line is dropped


def test_long_tables_keep_their_header():
    rows = "\n".join(f"| term{i} | meaning {i} {'x' * 40} |" for i in range(40))
    body = f"| Term | Meaning |\n| --- | --- |\n{rows}"
    pieces = chunks.split_body(body, max_chars=500)
    assert len(pieces) > 1
    assert all(p.startswith("| Term | Meaning |\n| --- | --- |") for p in pieces)
    assert all(len(p) <= 500 for p in pieces)
    assert sum(p.count("| term") for p in pieces) == 40


def test_table_padding_is_removed():
    assert chunks.compact_tables("| a     | b  |\n| ----- | -- |") == "| a | b |\n| --- | -- |"


def test_html_comments_are_removed():
    assert chunks.clean_links("See the [note]<!-- ignore --> above.") == "See the [note] above."


def test_inline_and_reference_links_both_cleaned():
    assert chunks.clean_links("[text](https://example.com)") == "[text]"
    assert chunks.clean_links("[text][ref]\n\n[ref]: https://example.com") == "[text]\n\n"


@pytest.fixture
def table(book, tmp_path):
    cs = chunks.chunk_book(book)
    vectors = np.eye(len(cs), 8, dtype=np.float32)     # chunk i points along axis i
    return store.build(store.connect(tmp_path / "db"), "chunks_test", cs, vectors)


def test_vector_search_finds_the_nearest_chunk(table):
    q = np.zeros(8, dtype=np.float32)
    q[3] = 1
    rows = store.search(table, "vector", "anything", q, k=2)
    assert rows[0]["ref"] == "ch08-01-vectors.md#reading-elements-of-vectors"
    assert rows[0]["score"] == pytest.approx(1.0)
    assert "vector" not in rows[0]


def test_keyword_search(table):
    rows = store.search(table, "fts", "reading elements vectors", k=2)
    assert rows[0]["ref"] == "ch08-01-vectors.md#reading-elements-of-vectors"


def test_missing_table_says_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="index --model e5-small"):
        store.open_table(store.connect(tmp_path), "chunks_e5_small")


def test_rank_counts_sections_not_chunks():
    rows = [{"ref": "a"}, {"ref": "a"}, {"ref": "b"}, {"ref": "c"}]
    assert compare.rank_of(rows, ["c"]) == 3
    assert compare.rank_of(rows, ["x"]) is None
    assert compare.score([1, 3, None, 6]) == {"hit@1": 0.25, "hit@5": 0.5,
                                              "MRR": pytest.approx((1 + 1 / 3 + 1 / 6) / 4)}


def test_every_expected_answer_exists_in_the_book():
    """A typo in questions.jsonl would quietly count as a miss for every model."""
    refs = {c.ref for c in chunks.chunk_book(chunks.find_book())}
    missing = [e for q in compare.load_questions() for e in q["expect"] if e not in refs]
    assert not missing
