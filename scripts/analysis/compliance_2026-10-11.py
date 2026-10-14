#!/usr/bin/env python3
"""WebShop action counts (exploratory): searches per session under no memory, the memory and its matched
unrelated-text control, for each reader. The distilled WebShop pool contains "search again" rules; the count
shows how far those rules reach the agent's actions. Reads the WebShop episode logs; no model calls.
Output: results/placebo/compliance_2026-10-11.json"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"


def load(run):
    p = RES / f"{run}.episodes.jsonl"
    if not p.exists():
        return None
    rows = {}
    for line in p.read_text().splitlines():
        if line.strip():
            x = json.loads(line)
            rows[(int(x["seed"]), int(x["task_id"]))] = x
    return rows


def score(rows):
    return sum(float(x.get("reward", 0.0)) for x in rows.values()) / len(rows)


def searches(rows):
    return sum(sum(s["action"].lower().startswith("search[") for s in x["trajectory"]) for x in rows.values()) / len(rows)


def main():
    out = {"webshop": {}}
    for ex in ("GPT", "GPT41MINI", "QWEN"):
        R = {c: load(f"WS1010_{c}_{ex}") for c in ("NONE", "TRUE", "IRRELEVANT")}
        if any(v is None for v in R.values()):
            continue
        out["webshop"][ex] = {"content_value_R_score": round(score(R["TRUE"]) - score(R["IRRELEVANT"]), 4),
                              "searches": {c: round(searches(v), 2) for c, v in R.items()},
                              "search_compliance": round(searches(R["TRUE"]) - searches(R["IRRELEVANT"]), 2),
                              "episodes": len(R["TRUE"])}
    (RES / "placebo/compliance_2026-10-11.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
