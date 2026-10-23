"""
Run a batch of Zen Coder test conversations and judge each one.

    python -m zen.run --run 2026-10-23a                      # the default grid
    python -m zen.run --run k-sweep --prompts door --ks 0 1 3 5 --scenarios alarm plant
    python -m zen.run --run 2026-10-23a                      # again: resumes, skipping finished ones

Each conversation is one (prompt, k, scenario, repeat). A run directory,
data/zen/<run>/, holds:

    config.json        the arguments, so a run can be reproduced
    status.json        running | done | stopped | failed, counts, and the bill so far
    progress.jsonl     one line per Zen turn and per finished conversation, appended live
    <id>.json          the conversation: turns, automatic metrics, tokens and cost
    <id>.judge.json    the judge's scores

The dashboard (app.py) starts runs and reads these files while they grow; it
never waits on this process. Touching data/zen/<run>/STOP asks the run to stop
after the current turns. Conversations run a few at a time (--workers), so a
12-conversation run takes a few minutes rather than half an hour.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from zen import coder

HERE = Path(__file__).resolve().parent
LLM = HERE.parent
RUNS_DIR = LLM / "data" / "zen"
STALE_SECONDS = 600     # an API call with retries can take minutes; longer silence is "lost"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    p.add_argument("--run", required=True, help="run name: the directory under data/zen/")
    p.add_argument("--prompts", nargs="+", default=["baseline", "door"])
    p.add_argument("--ks", nargs="+", type=int, default=[0, 3])
    p.add_argument("--scenarios", nargs="+", default=["alarm", "plant"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--turns", type=int, default=3, help="Zen turns per conversation")
    p.add_argument("--zen-model", default=coder.ROLES["zen"], help="sonnet, haiku, opus, or an id")
    p.add_argument("--novice-model", default=coder.ROLES["novice"])
    p.add_argument("--judge-model", default=coder.ROLES["judge"])
    p.add_argument("--retrieval-model", default=None, help="embedding model (default: e5-small)")
    p.add_argument("--retrieval-mode", default="vector", choices=["vector", "fts", "hybrid"])
    p.add_argument("--workers", type=int, default=4, help="conversations at a time")
    return p.parse_args(argv)


def write_json(path: Path, obj) -> None:
    """Write via a temp file so a reader never sees half a file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def conversations(cfg: dict) -> list[dict]:
    return [{"id": f"{p}-k{k}-{s}-r{r}", "prompt": p, "k": k, "scenario": s, "repeat": r}
            for p in cfg["prompts"] for k in cfg["ks"] for s in cfg["scenarios"]
            for r in range(1, cfg["repeats"] + 1)]


class Run:
    """The files of one run, written safely from several worker threads."""

    def __init__(self, run_dir: Path, cfg: dict, total: int):
        self.dir, self.cfg = run_dir, cfg
        self.lock = threading.Lock()
        self.status = {"state": "running", "started": time.time(), "pid": os.getpid(),
                       "total": total, "done": 0, "failed": 0, "skipped": 0, "cost": 0.0,
                       "errors": []}

    def log(self, **row) -> None:
        with self.lock, (self.dir / "progress.jsonl").open("a") as f:
            f.write(json.dumps({"t": time.time(), **row}) + "\n")

    def update(self, **changes) -> None:
        with self.lock:
            for k, v in changes.items():
                if k in ("done", "failed", "skipped", "cost"):
                    self.status[k] += v
                elif k == "error":
                    self.status["errors"].append(v)
                else:
                    self.status[k] = v
            write_json(self.dir / "status.json", self.status)

    def stopping(self) -> bool:
        return (self.dir / "STOP").exists()


def one(run: Run, c: dict, prompts: dict, scenarios: dict) -> None:
    """Converse, judge, and write one conversation's two files."""
    if (run.dir / f"{c['id']}.judge.json").exists():
        run.update(skipped=1)
        return
    if run.stopping():
        return
    cfg, claude = run.cfg, None
    scenario = scenarios[c["scenario"]]
    retrieval = {"model": cfg["retrieval_model"], "mode": cfg["retrieval_mode"]}
    try:
        claude = coder.Claude()
        turns = coder.converse(
            claude, prompts[c["prompt"]], c["k"], scenario, cfg["turns"],
            models={"zen": cfg["zen_model"], "novice": cfg["novice_model"]},
            retrieval=retrieval,
            on_turn=lambda n, t: run.log(kind="turn", id=c["id"], n=n,
                                         words=len(t["response"].split())),
            should_stop=run.stopping)
        if run.stopping():
            run.log(kind="stopped", id=c["id"], turns=len(turns))
            run.update(cost=claude.cost)
            return                  # a cut-short conversation isn't judged against full ones
        rec = {**c, "run": run.dir.name, "turns": turns,
               "metrics": coder.metrics(scenario, turns)}
        scores = coder.judge(claude, scenario, turns, cfg["judge_model"])
        rec["usage"], rec["cost"] = claude.usage, claude.cost
        write_json(run.dir / f"{c['id']}.json", rec)
        write_json(run.dir / f"{c['id']}.judge.json", scores)
        overall = sum(scores[k] for k in coder.CRITERIA) / len(coder.CRITERIA)
        run.log(kind="conversation", id=c["id"], overall=overall, cost=claude.cost)
        run.update(done=1, cost=claude.cost)
    except Exception as e:          # one bad conversation must not sink the rest of the run
        run.log(kind="error", id=c["id"], error=f"{type(e).__name__}: {e}")
        run.update(failed=1, cost=claude.cost if claude else 0.0, error=f"{c['id']}: {type(e).__name__}: {e}")
        traceback.print_exc()


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = {k: v for k, v in vars(args).items() if k != "run"}
    prompts, scenarios = coder.load_prompts(), coder.load_scenarios()
    for name in cfg["prompts"]:
        if name not in prompts:
            sys.exit(f"no prompt {name!r} in zen/prompts.json")
    for name in cfg["scenarios"]:
        if name not in scenarios:
            sys.exit(f"no scenario {name!r} in zen/scenarios.json")
    cfg["prompt_text"] = {p: prompts[p] for p in cfg["prompts"]}    # what was actually run

    run_dir = RUNS_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "STOP").unlink(missing_ok=True)
    write_json(run_dir / "config.json", cfg)
    convs = conversations(cfg)
    run = Run(run_dir, cfg, len(convs))
    run.update()
    print(f"{len(convs)} conversations -> {run_dir}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in [pool.submit(one, run, c, prompts, scenarios) for c in convs]:
            f.result()

    s = run.status
    state = ("stopped" if run.stopping() else
             "failed" if s["failed"] and not (s["done"] or s["skipped"]) else "done")
    run.update(state=state, finished=time.time())
    print(f"{state}: {s['done']} done, {s['skipped']} already done, {s['failed']} failed, "
          f"${s['cost']:.2f}", flush=True)


# ---------- for the dashboard ----------

def list_runs(root: Path = RUNS_DIR) -> list[Path]:
    """Run directories, newest first. Runs made by zen.workflow.js have records but no
    config.json, and are listed too."""
    if not root.exists():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and any(p.glob("*.json"))),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def state(run_dir: Path, now: float | None = None) -> str:
    status = read_json(run_dir / "status.json")
    if status is None:
        return "done" if any(run_dir.glob("*.judge.json")) else "unknown"
    s = status.get("state", "unknown")
    if s == "running":
        progress = run_dir / "progress.jsonl"
        last = progress.stat().st_mtime if progress.exists() else status.get("started", 0)
        if (now or time.time()) - last > STALE_SECONDS:
            return "lost"
    return s


def read_progress(run_dir: Path) -> list[dict]:
    rows = []
    try:
        text = (run_dir / "progress.jsonl").read_text()
    except OSError:
        return rows
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass                    # a half-written last line
    return rows


def start(name: str, cfg: dict, root: Path = RUNS_DIR) -> Path:
    """Launch a run as a detached process, so a Streamlit rerun or closed tab doesn't stop it."""
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "-m", "zen.run", "--run", name]
    for k, v in cfg.items():
        if v is None:
            continue
        args.append(f"--{k.replace('_', '-')}")
        args += [str(x) for x in v] if isinstance(v, list) else [str(v)]
    detach = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32"
              else {"start_new_session": True})
    log = (run_dir / "stdout.log").open("w")
    subprocess.Popen(args, cwd=LLM, stdout=log, stderr=subprocess.STDOUT, **detach)
    return run_dir


def stop(run_dir: Path) -> None:
    """Ask a run to stop; conversations finish the turn they are on."""
    (run_dir / "STOP").touch()


if __name__ == "__main__":
    main()
