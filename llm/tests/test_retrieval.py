"""
Offline tests: no model downloads. Vectors are made by hand, so these check
the chunking, the LanceDB plumbing and the scoring, not any model's quality.
That is compare.py's job.
"""
import os
from pathlib import Path

import numpy as np
import pytest

import chunks
import compare
import store

GUIDE = """\
# 7. Operating

Intro to the chapter.

## If you have never deployed anything

Background.

### <a id="o8"></a>O8. Secure by default

Deny by default. See [SE1/T3](06-testing.md#t3).

```bash
# not a heading
```

### <a id="o9"></a>O9. In an incident, restore service first

Roll back, then learn.
"""


@pytest.fixture
def guide(tmp_path) -> Path:
    (tmp_path / "guide").mkdir()
    (tmp_path / "guide" / "07-operating.md").write_text(GUIDE)
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "hidden.md").write_text("# skip me\n\ntext\n")
    return tmp_path


def test_chunks_follow_headings_and_cite_anchors(guide):
    cs = chunks.chunk_guide(guide)
    assert [c.ref for c in cs] == [
        "guide/07-operating.md",
        "guide/07-operating.md#if-you-have-never-deployed-anything",
        "guide/07-operating.md#o8",
        "guide/07-operating.md#o9",
    ]
    o8 = cs[2]
    assert o8.principle == "SE1/O8"
    # A principle hangs off its chapter, not the `##` section before it.
    assert o8.heading == "7. Operating > SE1/O8. Secure by default"
    assert "# not a heading" in o8.body          # fenced code is not split on
    assert "[SE1/T3]" in o8.body and "06-testing.md" not in o8.body
    assert "<a id" not in o8.text


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


def test_principles_in_questions():
    assert chunks.principles_in("How do U3 and SE1/A2 relate? And U3?") == ["SE1/U3", "SE1/A2"]
    assert chunks.principles_in("p95 latency on a k8s cluster") == []


@pytest.fixture
def table(guide, tmp_path):
    cs = chunks.chunk_guide(guide)
    vectors = np.eye(len(cs), 8, dtype=np.float32)     # chunk i points along axis i
    return store.build(store.connect(tmp_path / "db"), "chunks_test", cs, vectors)


def test_vector_search_finds_the_nearest_chunk(table):
    q = np.zeros(8, dtype=np.float32)
    q[3] = 1
    rows = store.search(table, "vector", "anything", q, k=2)
    assert rows[0]["ref"] == "guide/07-operating.md#o9"
    assert rows[0]["score"] == pytest.approx(1.0)
    assert "vector" not in rows[0]


def test_keyword_search(table):
    rows = store.search(table, "fts", "incident rollback", k=2)
    assert rows[0]["ref"] == "guide/07-operating.md#o9"


def test_naming_a_principle_pins_it_first(table):
    q = np.zeros(8, dtype=np.float32)
    q[3] = 1                                   # nearest is O9 ...
    rows = store.search(table, "vector", "what does O8 say", q, k=2)
    assert rows[0]["ref"] == "guide/07-operating.md#o8"   # ... but O8 was named
    unpinned = store.search(table, "vector", "what does O8 say", q, k=2, pin=False)
    assert unpinned[0]["ref"] == "guide/07-operating.md#o9"


def test_missing_table_says_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="index --model e5-small"):
        store.open_table(store.connect(tmp_path), "chunks_e5_small")


def test_rank_counts_sections_not_chunks():
    rows = [{"ref": "a"}, {"ref": "a"}, {"ref": "b"}, {"ref": "c"}]
    assert compare.rank_of(rows, ["c"]) == 3
    assert compare.rank_of(rows, ["x"]) is None
    assert compare.score([1, 3, None, 6]) == {"hit@1": 0.25, "hit@5": 0.5,
                                              "MRR": pytest.approx((1 + 1 / 3 + 1 / 6) / 4)}


def test_every_expected_answer_exists_in_the_guide():
    """A typo in questions.jsonl would quietly count as a miss for every model."""
    root = next((Path(p).expanduser() for p in (os.environ.get("SE_GUIDE"),
                 "~/projects/software-engineering", chunks.DATA_DIR / "software-engineering")
                 if p and Path(p).expanduser().is_dir()), None)
    if root is None:
        pytest.skip("guide not checked out; run `pixi run -e llm index` once")
    refs = {c.ref for c in chunks.chunk_guide(root)}
    missing = [e for q in compare.load_questions() for e in q["expect"] if e not in refs]
    assert not missing
