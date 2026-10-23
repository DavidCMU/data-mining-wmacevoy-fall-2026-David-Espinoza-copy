"""
Zen Coder's retrieval step: print the top k guide sections for a query.

    python zen/retrieve.py -k 3 "remind me before class without distracting me"
    echo "remind me before class" | python zen/retrieve.py -k 3

zen.workflow.js has Zen Coder run this once per turn, passing the query on
stdin so quotes and apostrophes need no escaping. Search settings are the
project defaults from ask.py (e5-small, vector, principles pinned by ID).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ask import retrieve  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    p.add_argument("query", nargs="?", help="the query (default: read stdin)")
    p.add_argument("-k", type=int, default=5, help="how many sections to print")
    args = p.parse_args(argv)

    query = args.query or sys.stdin.read().strip()
    for r in retrieve(query, k=args.k):
        body = r["text"].split("\n\n", 1)[-1]
        print(f"[{r['ref']}] {r['heading']}\n\n{body}\n")


if __name__ == "__main__":
    main()
