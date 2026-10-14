#!/usr/bin/env python3
"""Distill the WebShop experience memory (2026-10-10 study) exactly as the ALFWorld compact insights
were distilled (skillrc/build_skills.py): one call per task type with at most six successful
trajectories, the ExpeL-style insight prompt with only the environment words changed, gpt-4o-mini at
temperature 0, 512 completion tokens. In WebShop the task type is the product category.

Source: the train-session collection runs WS1010_COLLECT_<CATEGORY> (gpt-4o-mini, no memory, sessions
>= 1500, disjoint from the test sessions 0-499). Per category the first six successful episodes
(reward 1.0) in session order are used; a category without a success contributes no insights.
Trajectories are serialized as instruction + action/observation lines; observations are cut to
300 characters so that six trajectories fit one prompt.

Outputs: data/pools/webshop1010_insights.jsonl ("[<category>] <insight>", task_type=<category>)
         results/placebo/webshop1010_pool_audit.json (sessions used, raw responses, token footprint)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from skillrc.build_skills import _bullets
from skillrc.llm import LLMClient
from skillrc.tokens import count_tokens

ROOT = Path(__file__).resolve().parents[2]
CATS = ("beauty", "electronics", "fashion", "garden", "grocery")
PROMPT = (
    "Below are successful agent trajectories for WebShop '{task_type}' shopping tasks.\n"
    "Extract 3 to 6 concise, GENERAL insights (rules/heuristics) that help solve this "
    "task type. Each insight: one imperative line, no specific product names, product IDs or "
    "prices. Output ONLY a bullet list, each line starting with '- '.\n\n"
    "Trajectories:\n{trajs}"
)
OBS_CHARS = 300


def instruction(text: str) -> str:
    marker = "Instruction: [SEP]"
    return text.split(marker, 1)[1].split("[SEP]", 1)[0].strip() if marker in text else text[:400]


def serialize(row) -> str:
    lines = [f"Instruction: {instruction(row['task_text'])}"]
    for step in row["trajectory"]:
        obs = " ".join(step["obs"].split())
        lines.append(f"Action: {step['action']}\nObservation: {obs[:OBS_CHARS]}{' ...' if len(obs) > OBS_CHARS else ''}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/pools/webshop1010_insights.jsonl")
    ap.add_argument("--audit", type=Path, default=ROOT / "results/placebo/webshop1010_pool_audit.json")
    ap.add_argument("--max-per-type", type=int, default=6)
    a = ap.parse_args()
    llm = LLMClient(model="gpt-4o-mini", provider="openai", temperature=0.0, max_completion_tokens=512,
                    max_retries=5, api_key=os.environ.get("OPENAI_API_KEY"))
    items, audit = [], {"model": "gpt-4o-mini", "temperature": 0.0, "max_per_type": a.max_per_type,
                        "obs_chars": OBS_CHARS, "prompt": PROMPT, "categories": {}}
    for cat in CATS:
        path = ROOT / f"results/WS1010_COLLECT_{cat.upper()}.episodes.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        rows.sort(key=lambda r: int(r["task_id"]))
        assert all(int(r["task_id"]) >= 1500 for r in rows), "collection must use train sessions only"
        wins = [r for r in rows if not r.get("error") and r["success"]]
        chosen = wins[:a.max_per_type]
        entry = {"episodes": len(rows), "successes": len(wins), "mean_reward": round(sum(r["reward"] for r in rows) / max(1, len(rows)), 4),
                 "sessions_used": [int(r["task_id"]) for r in chosen]}
        if chosen:
            joined = "\n\n---\n\n".join(serialize(r) for r in chosen)
            resp = llm.complete([{"role": "user", "content": PROMPT.format(task_type=cat, trajs=joined)}])
            bullets = _bullets(resp.text)
            entry.update({"response": resp.text, "n_insights": len(bullets), "prompt_tokens": resp.prompt_tokens})
            items += [{"text": f"[{cat}] {b}", "task_type": cat} for b in bullets]
        else:
            entry["n_insights"] = 0
        audit["categories"][cat] = entry
        print(cat, {k: v for k, v in entry.items() if k != "response"}, flush=True)
    ctx = "\n\n".join(it["text"] for it in items)
    audit.update({"n_items": len(items), "footprint_gpt4omini": count_tokens(ctx, "gpt-4o-mini"),
                  "body_words": sum(len(it["text"].split()) - 1 for it in items)})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    a.audit.parent.mkdir(parents=True, exist_ok=True)
    a.audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"{len(items)} insights, {audit['footprint_gpt4omini']} tokens (gpt-4o-mini) -> {a.out}")


if __name__ == "__main__":
    main()
