#!/usr/bin/env python3
"""Compact per-episode outcomes of every reported run, and their expansion into the layout the analysis scripts read.

  python scripts/outcomes.py unpack     # data/outcomes/*.jsonl.gz -> results/<run_id>.episodes.jsonl, plus the
                                        # audit and task files the analyses read -> results/placebo/
  python scripts/outcomes.py export     # (authors) full episode logs in results/ -> data/outcomes/*.jsonl.gz

One line per episode: run, seed, task_id, task_type, success, reward, steps, memory_tokens, error and actions.
`actions` holds every environment action of a WebShop session (the analyses count searches and purchases), and only
the first non-"think" action of an ALFWorld episode (the generic-rule analysis compares first actions). Prompts,
observations and the rest of each trajectory are not included. Unpacking never overwrites an existing file.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "data/outcomes"
STUDIES = {  # file stem -> run-id prefixes
    "five_readers": ("RERUN1010_", "P2A_", "P2B_", "P2C_"),
    "webshop": ("WS1010_", "WS1010R2_"),
    "crossover_and_generic": ("W1A1013_", "W1B1013_"),
    "pool_replication": ("PRP1012_",),
}
EXCLUDE = ("COLLECT",)
FIELDS = ("seed", "task_id", "task_type", "success", "reward", "steps", "memory_tokens")
# files the analysis scripts read from results/placebo/ (all released under data/summaries/)
SUPPORT = ["rerun_2026-10-10/frozen_tasks_rerun_2026-10-10.json", "w1a_2026-10-13/audits/*.json",
           "w1b_2026-10-13/audits/*_audit.json", "poolrep_2026-10-12/audits/*.json"]


def env_actions(traj):
    return [str(s.get("action", "")) for s in traj or []]


def first_action(traj):
    for s in traj or []:
        a = str(s.get("action", ""))
        if not a.startswith("think"):
            return [a]
    return []


def export():
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, prefixes in STUDIES.items():
        runs = sorted(p.name[:-len(".episodes.jsonl")] for p in RES.glob("*.episodes.jsonl")
                      if p.name.startswith(prefixes) and not any(x in p.name for x in EXCLUDE))
        n = 0
        with gzip.open(OUT / f"{stem}.jsonl.gz", "wt", encoding="utf-8") as f:
            for run in runs:
                webshop = run.startswith("WS")
                for line in (RES / f"{run}.episodes.jsonl").read_text().splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    row = {"run": run, **{k: r.get(k) for k in FIELDS}, "error": "error" if r.get("error") else "",
                           "actions": env_actions(r.get("trajectory")) if webshop else first_action(r.get("trajectory"))}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
        print(f"{stem}: {len(runs)} runs, {n} episodes")


def unpack():
    RES.mkdir(exist_ok=True)
    (RES / "placebo").mkdir(exist_ok=True)
    by_run = {}
    for path in sorted(OUT.glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                by_run.setdefault(r.pop("run"), []).append(r)
    written = skipped = 0
    for run, rows in by_run.items():
        dst = RES / f"{run}.episodes.jsonl"
        if dst.exists():
            skipped += 1
            continue
        with dst.open("w", encoding="utf-8") as f:
            for r in rows:
                r["trajectory"] = [{"action": a} for a in r.pop("actions")]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        written += 1
    for pattern in SUPPORT:
        for src in sorted((ROOT / "data/summaries").glob(pattern)):
            dst = RES / "placebo" / src.name
            if not dst.exists():
                shutil.copy(src, dst)
    print(f"wrote {written} episode files to results/ ({skipped} already present); copied audit and task files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("unpack", "export"))
    {"unpack": unpack, "export": export}[ap.parse_args().command]()
