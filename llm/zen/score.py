"""
Summarize a Zen Coder harness run: judge scores and automatic metrics by prompt and k.

    python zen/score.py <run>              # one row per (prompt, k)
    python zen/score.py <run> --by scenario
    python zen/score.py <run> --verdicts   # every judge verdict, worst first

Reads data/zen/<run>/<id>.json (the conversation, with its automatic metrics)
and <id>.judge.json (the judge's scores), as written by zen/run.py or zen.workflow.js.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

RUNS = Path(__file__).resolve().parent.parent / "data" / "zen"
CRITERIA = ["understands", "fits", "teaches", "grounded", "tone"]


def load(run: str) -> pd.DataFrame:
    rows = []
    for path in sorted((RUNS / run).glob("*.json")):
        if path.name.endswith(".judge.json"):
            continue
        rec = json.loads(path.read_text())
        if "id" not in rec:
            continue                # config.json, status.json
        judge_path = path.with_suffix(".judge.json")
        judge = json.loads(judge_path.read_text()) if judge_path.exists() else {}
        rows.append({"id": rec["id"], "prompt": rec["prompt"], "k": rec["k"],
                     "scenario": rec["scenario"], **rec["metrics"],
                     **{c: judge.get(c) for c in CRITERIA},
                     "facts": len(judge.get("facts_discovered", [])),
                     "first_rec": judge.get("first_recommendation_turn"),
                     "cost": rec.get("cost"),
                     "verdict": judge.get("verdict", "")})
    if not rows:
        raise SystemExit(f"no records in {RUNS / run}")
    df = pd.DataFrame(rows)
    df["overall"] = df[CRITERIA].mean(axis=1)
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    p.add_argument("run")
    p.add_argument("--by", nargs="+", default=["prompt", "k"],
                   help="columns to group by (prompt, k, scenario)")
    p.add_argument("--verdicts", action="store_true", help="list each judge verdict, worst first")
    args = p.parse_args(argv)

    df = load(args.run)
    cols = ["overall", *CRITERIA, "facts", "first_rec", "relevant_recall", "guide_leaks",
            "citations", "invalid_citations", "ungrounded_citations",  # runs before 2026-10-22c
            "questions_per_turn", "words_per_turn", "cost"]
    cols = [c for c in cols if c in df and df[c].notna().any()]
    table = df.groupby(args.by)[cols].mean().round(2)
    table.insert(0, "n", df.groupby(args.by).size())
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.sort_values("overall", ascending=False))
    if args.verdicts:
        for r in df.sort_values("overall").itertuples():
            print(f"\n{r.overall:.1f}  {r.id}\n     {r.verdict}")


if __name__ == "__main__":
    main()
