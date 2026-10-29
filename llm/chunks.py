"""
Split the software-engineering guide into chunks a retriever can return.

    python chunks.py                 # summary of what would be indexed
    python chunks.py --show 5        # print the first five chunks

The guide is Markdown organized as numbered principles, one heading each
(`### <a id="t3"></a>T3. A test that cannot fail proves nothing`). A heading
is the natural unit: it is what a person would cite. So we split on headings,
and only fall back to splitting by paragraph (then by line) when a section is
longer than --max-chars. Tables keep their header row in every piece, so a
slice of the glossary still says what its columns mean.

Each chunk records where it came from (`ref`, e.g. guide/06-testing.md#t3) so
an answer can cite it, and so compare.py can check whether retrieval found
the right section.
"""
import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

GUIDE_REPO = "https://github.com/wmacevoy/software-engineering.git"
GUIDE_DIR = Path(__file__).parent / "vendor" / "software-engineering"   # see vendor.py
DATA_DIR = Path(__file__).parent / "data"
MAX_CHARS = 1500        # ~350 tokens: under every candidate model's limit but MiniLM's
PRINCIPLE_VERSION = "SE1"   # the guide cites principles as SE1/T3 (see its llms.txt)

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
ANCHOR = re.compile(r'<a id="([^"]+)"></a>\s*')
PRINCIPLE_ID = re.compile(r"^[a-z]\d+$")
CITED_ID = re.compile(rf"\b(?:{PRINCIPLE_VERSION}/)?([A-Z]\d{{1,2}})\b")
LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


@dataclass
class Chunk:
    file: str           # path inside the guide, e.g. guide/06-testing.md
    ref: str            # file#anchor, the citation for this section
    heading: str        # "6. Testing > SE1/T3. A test that cannot fail proves nothing"
    principle: str      # "SE1/T3", or "" for sections that are not principles
    part: int           # 0, 1, ... when a long section was split
    body: str

    @property
    def text(self) -> str:
        """What gets embedded and keyword-indexed: the heading path, then the body."""
        return f"{self.heading}\n\n{self.body}"


def principles_in(text: str) -> list[str]:
    """Principle IDs a question names, normalized: "what does t3 say about SE1/O8" ->
    ["SE1/T3", "SE1/O8"]. Bare IDs must be capitalized ("T3"), so "a k8s cluster"
    or "p95 latency" don't count."""
    return [f"{PRINCIPLE_VERSION}/{m}" for m in dict.fromkeys(CITED_ID.findall(text))]


def find_guide(path: str | None = None) -> Path:
    """The guide: --corpus, else $SE_GUIDE (a working checkout, for editing the guide and
    the lab together), else the vendored copy in vendor/software-engineering."""
    for candidate in (path, os.environ.get("SE_GUIDE"), GUIDE_DIR):
        if candidate and (p := Path(candidate).expanduser()).is_dir():
            return p
    raise SystemExit(f"no guide at {GUIDE_DIR}; run `pixi run -e llm vendor` to fetch it")


def guide_files(root: Path) -> list[Path]:
    """Every Markdown file in the guide, skipping hidden directories."""
    return sorted(p for p in root.rglob("*.md")
                  if not any(part.startswith(".") for part in p.relative_to(root).parts))


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lowercase, punctuation dropped, spaces to hyphens."""
    s = re.sub(r"[^\w\- ]", "", heading.lower())
    return s.replace(" ", "-")


def sections(markdown: str) -> list[tuple[list[str], str, str]]:
    """Split on headings. Returns (heading path, anchor, body) per section.

    A `#` inside a ``` fence is code, not a heading.
    """
    out = []
    path: list[str] = []        # heading text by level, e.g. ["6. Testing", "T3. ..."]
    anchor, lines, fenced = "", [], False

    def flush():
        body = "\n".join(lines).strip()
        if body:
            out.append((list(path), anchor, body))

    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        m = None if fenced else HEADING.match(line)
        if not m:
            lines.append(line)
            continue
        flush()
        level, title = len(m.group(1)), m.group(2).strip()
        a = ANCHOR.search(title)
        title = ANCHOR.sub("", title).strip()
        path = path[:level - 1] + [""] * (level - 1 - len(path)) + [title]
        anchor = a.group(1) if a else slug(title)
        lines = []
    flush()
    return out


def compact_tables(body: str) -> str:
    """Drop the padding that aligns table columns in an editor. It is invisible
    once rendered, but a model would pay tokens for every space."""
    rows = []
    for line in body.splitlines():
        if line.startswith("|"):
            line = re.sub(r"-{3,}", "---", re.sub(r" {2,}", " ", line))
        rows.append(line)
    return "\n".join(rows)


def split_body(body: str, max_chars: int) -> list[str]:
    """Pack paragraphs into pieces of at most max_chars. A paragraph that is
    itself too long is split by line; a table repeats its header in each piece."""
    blocks = []
    for para in re.split(r"\n\s*\n", body):
        if len(para) <= max_chars:
            blocks.append(para)
            continue
        rows = para.splitlines()
        header = rows[:2] if rows[0].startswith("|") and len(rows) > 2 else []
        piece = list(header)
        for row in rows[len(header):]:
            if piece != header and len("\n".join(piece + [row])) > max_chars:
                blocks.append("\n".join(piece))
                piece = list(header)
            piece.append(row)
        blocks.append("\n".join(piece))

    pieces, current = [], ""
    for block in blocks:
        if current and len(current) + 2 + len(block) > max_chars:
            pieces.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current:
        pieces.append(current)
    return pieces


def chunk_file(root: Path, path: Path, max_chars: int = MAX_CHARS) -> list[Chunk]:
    rel = path.relative_to(root).as_posix()
    chunks = []
    for heading_path, anchor, body in sections(path.read_text(encoding="utf-8")):
        principle = ""
        if PRINCIPLE_ID.match(anchor):
            # A principle belongs to its chapter, even when an unrelated `##` section
            # (like 7. Operating's "If you have never deployed anything") precedes it.
            principle = f"{PRINCIPLE_VERSION}/{anchor.upper()}"
            heading_path = [heading_path[0], f"{PRINCIPLE_VERSION}/{heading_path[-1]}"]
        # Text before the first `##` belongs to the page itself, so cite the file.
        ref = rel if len(heading_path) <= 1 else f"{rel}#{anchor}"
        heading = " > ".join(h for h in heading_path if h)
        # Keep a link's text, drop its target: "[SE1/O8](07-operating.md#o8)" -> "[SE1/O8]".
        # Targets are noise to a model, and to keyword search they look like extra
        # mentions of o8 -- enough to outrank the section that defines it.
        body = compact_tables(LINK.sub(r"[\1]", ANCHOR.sub("", body)))
        for i, piece in enumerate(split_body(body, max_chars)):
            chunks.append(Chunk(rel, ref, heading, principle, i, piece))
    return chunks


def chunk_guide(root: Path, max_chars: int = MAX_CHARS) -> list[Chunk]:
    return [c for p in guide_files(root) for c in chunk_file(root, p, max_chars)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    p.add_argument("--corpus", help="path to the guide (default: see find_guide)")
    p.add_argument("--max-chars", type=int, default=MAX_CHARS)
    p.add_argument("--show", type=int, default=0, help="print this many chunks")
    args = p.parse_args(argv)

    root = find_guide(args.corpus)
    chunks = chunk_guide(root, args.max_chars)
    sizes = sorted(len(c.text) for c in chunks)
    print(f"{root}: {len(guide_files(root))} files -> {len(chunks)} chunks, "
          f"{sum(c.part == 0 for c in chunks)} sections")
    print(f"chunk length (chars): min {sizes[0]}, median {sizes[len(sizes) // 2]}, "
          f"max {sizes[-1]}")
    for c in chunks[:args.show]:
        print(f"\n--- {c.ref} (part {c.part}) ---\n{c.text}")


if __name__ == "__main__":
    main()
