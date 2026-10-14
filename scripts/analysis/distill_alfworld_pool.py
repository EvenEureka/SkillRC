#!/usr/bin/env python3
"""Build an ALFWorld insight pool from one source's
valid_seen no-memory trajectories, with the same pipeline for every source.

Successful episodes, in game order, are serialised as the initial observation followed by
"> action" / observation lines (thinks kept as "> think: ..." / "OK."), grouped by task type, and the first
six per type go to the original ExpeL-style insight prompt of skillrc/build_skills.py (INSIGHT_PROMPT,
_bullets), distilled by gpt-4o-mini at temperature 0 with 512 completion tokens.
Outputs: data/pools/src1011_<source>_insights.jsonl and results/placebo/src1011_<source>_pool_audit.json."""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

from skillrc.build_skills import INSIGHT_PROMPT, _bullets
from skillrc.llm import LLMClient
from skillrc.tokens import count_tokens

ROOT = Path(__file__).resolve().parents[2]
TYPES = ("clean", "cool", "examine", "heat", "put", "puttwo")


def serialise(row) -> str:
    lines = [row["task_text"].strip()]
    for step in row["trajectory"]:
        lines.append(f"> {step['action']}\n{step['obs'].strip()}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["gpt", "qwen"])
    ap.add_argument("--max-per-type", type=int, default=6)
    a = ap.parse_args()
    run = f"SRC1011_COLLECT_{a.source.upper()}"
    rows = [json.loads(x) for x in (ROOT / f"results/{run}.episodes.jsonl").read_text().splitlines() if x.strip()]
    rows.sort(key=lambda r: int(r["task_id"]))
    wins = collections.defaultdict(list)
    for r in rows:
        if r["success"] and not r.get("error"):
            wins[r["task_type"]].append(r)
    llm = LLMClient(model="gpt-4o-mini", provider="openai", temperature=0.0, max_completion_tokens=512,
                    max_retries=5, api_key=os.environ.get("OPENAI_API_KEY"))
    items, audit = [], {"source_run": run, "episodes": len(rows), "successes": sum(len(v) for v in wins.values()),
                        "distiller": "gpt-4o-mini", "temperature": 0.0, "max_per_type": a.max_per_type, "types": {}}
    for t in TYPES:
        chosen = wins[t][:a.max_per_type]
        entry = {"successes": len(wins[t]), "games_used": [int(r["task_id"]) for r in chosen]}
        if chosen:
            resp = llm.complete([{"role": "user", "content": INSIGHT_PROMPT.format(
                task_type=t, trajs="\n\n---\n\n".join(serialise(r) for r in chosen))}])
            bullets = _bullets(resp.text)
            items += [{"text": f"[{t}] {b}", "task_type": t} for b in bullets]
            entry.update({"n_insights": len(bullets), "response": resp.text})
        else:
            entry["n_insights"] = 0
        audit["types"][t] = entry
        print(t, {k: v for k, v in entry.items() if k != "response"}, flush=True)
    ctx = "\n\n".join(it["text"] for it in items)
    audit.update({"n_items": len(items), "footprint_gpt4omini": count_tokens(ctx, "gpt-4o-mini")})
    out = ROOT / f"data/pools/src1011_{a.source}_insights.jsonl"
    out.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    (ROOT / f"results/placebo/src1011_{a.source}_pool_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"{len(items)} insights, {audit['footprint_gpt4omini']} tokens -> {out}")


if __name__ == "__main__":
    main()
