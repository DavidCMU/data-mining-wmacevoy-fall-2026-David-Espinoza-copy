"""
Vendor the software-engineering guide: copy one version of it into this repository.

    python vendor.py                 # update vendor/software-engineering to the guide's main
    python vendor.py --ref v1.1      # pin a tag (or any branch)
    python vendor.py --check         # does the vendored copy match upstream? (exit 1 if not)
    python vendor.py --source ~/projects/software-engineering   # from a local checkout

The copy is committed, so a fresh clone of the course indexes the same text
every time, with no network and no second checkout. Updating is deliberate:
run this, read the file list it prints, re-index, and commit. The commit it
came from is recorded in vendor/software-engineering.json.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from chunks import GUIDE_DIR, GUIDE_REPO

RECORD = GUIDE_DIR.with_suffix(".json")


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def digests(root: Path) -> dict[str, str]:
    """sha256 of every file under root, by relative path."""
    if not root.is_dir():
        return {}
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def recorded() -> dict:
    return json.loads(RECORD.read_text()) if RECORD.exists() else {}


def check(source: str, ref: str) -> int:
    """Compare the vendored commit with the tip of ref upstream; 1 if they differ."""
    have = recorded().get("commit")
    lines = git("ls-remote", source, ref).splitlines()
    if not lines:
        sys.exit(f"{source} has no ref {ref!r}")
    # A tag may list itself and its peeled commit (^{}); the peeled one is the commit.
    tip = next((ln for ln in lines if ln.endswith("^{}")), lines[0]).split()[0]
    if have == tip:
        print(f"up to date: {ref} is {tip[:7]}")
        return 0
    print(f"differs: vendored {have[:7] if have else 'nothing'}, {ref} is {tip[:7]}. "
          f"Run `pixi run -e llm vendor --ref {ref}`.")
    return 1


def vendor(source: str, ref: str) -> None:
    before = digests(GUIDE_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "guide"
        git("clone", "--quiet", "--depth", "1", "--branch", ref, source, str(clone))
        commit = git("rev-parse", "HEAD", cwd=clone)
        date = git("log", "-1", "--format=%cI", cwd=clone)
        shutil.rmtree(GUIDE_DIR, ignore_errors=True)
        shutil.copytree(clone, GUIDE_DIR, ignore=shutil.ignore_patterns(".git"))
    record = {"repo": source, "ref": ref, "commit": commit, "date": date}
    RECORD.write_text(json.dumps(record, indent=2) + "\n")

    after = digests(GUIDE_DIR)
    changes = ([f"+ {f}" for f in after if f not in before]
               + [f"- {f}" for f in before if f not in after]
               + [f"M {f}" for f in after if f in before and before[f] != after[f]])
    print(f"vendored {ref} at {commit[:7]} ({date[:10]}) into vendor/{GUIDE_DIR.name}")
    if not changes:
        print("no files changed")
        return
    print("\n".join(sorted(changes, key=lambda c: c[2:])))
    print("\nNext: `pixi run -e llm index` (or `compare`) to re-embed, and check that "
          "zen/guide-summary.md still matches (`pixi run -e llm test`).")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    p.add_argument("--ref", default="main", help="branch or tag to vendor (default: main)")
    p.add_argument("--source", default=GUIDE_REPO, help="repository URL or local checkout")
    p.add_argument("--check", action="store_true", help="only report whether the copy matches --ref upstream")
    args = p.parse_args(argv)
    source = str(Path(args.source).expanduser()) if Path(args.source).expanduser().is_dir() \
        else args.source
    if args.check:
        sys.exit(check(source, args.ref))
    vendor(source, args.ref)


if __name__ == "__main__":
    main()
