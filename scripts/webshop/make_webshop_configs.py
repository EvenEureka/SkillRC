#!/usr/bin/env python3
"""Configs for the 2026-10-10 WebShop study.

  collect  GPT-4o-mini, no memory, on train sessions (>= 1500): the first 120 sessions of each
           product category in id order (all 96 grocery sessions), max 15 steps.
           Run ids WS1010_COLLECT_<CATEGORY>.
  eval     test sessions 0-499 x {none, true, placebo, irrelevant} x executors
           {GPT (gpt-4o-mini), GPT41MINI (gpt-4.1-mini-2025-04-14), QWEN (Qwen3-8B, thinking off)}.
           Run ids WS1010_<COND>_<EXECUTOR>; a second gpt-4o-mini run WS1010R2_<COND>_GPT (noise floor).
           --budget is the true pool's footprint (all items fit).

Session -> category comes from the seeded server (data/summaries/webshop_2026-10-10/instructions_seed0_1k.json,
seed 0, 1,000 products; built by the determinism check)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "configs/webshop_2026_10_10"
INSTR = ROOT / "data/summaries/webshop_2026-10-10/instructions_seed0_1k.json"
CATS = ("beauty", "electronics", "fashion", "garden", "grocery")
PER_CAT = 120
TEST = list(range(500))
TRUE_POOL = "data/pools/webshop1010_insights.jsonl"
POOLS = {"true": TRUE_POOL, "placebo": "data/pools/webshop1010_insights_order_placebo.jsonl",
         "irrelevant": "data/pools/webshop1010_insights_irrelevant_coherent.jsonl"}
EXECUTORS = {
    "GPT": {"provider": "openai", "model": "gpt-4o-mini"},
    "GPT41MINI": {"provider": "openai", "model": "gpt-4.1-mini-2025-04-14"},
    "QWEN": {"provider": "openai", "model": "qwen3-8b", "api_key": "EMPTY",
             "base_url": "http://127.0.0.1:8000/v1", "chat_template_kwargs": {"enable_thinking": False}},
}
BASE = {
    "out_dir": "results", "env": "webshop", "webshop_base_url": "http://127.0.0.1:3000",
    "webshop_n_tasks": 500, "temperature": 0.0, "max_completion_tokens": 256, "max_steps": 15,
    "request_timeout": 120.0, "max_retries": 5, "n_fewshot": 2, "system_prompt_key": "default",
    "prompt_style": "thought_action", "inject_admissible": True,
    "fewshot_file": "configs/webshop_react_fewshot.jsonl", "save_trajectories": True, "seeds": [0],
}


def dump(name, cfg):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, width=100))


def collect():
    instr = json.loads(INSTR.read_text())
    assert instr["seed"] == 0 and instr["num_products"] == 1000
    by_cat = {c: [] for c in CATS}
    for sid in sorted(int(k) for k in instr["instructions"]):
        cat = instr["instructions"][str(sid)]["category"]
        if sid >= 1500 and len(by_cat[cat]) < PER_CAT:
            by_cat[cat].append(sid)
    manifest = {}
    for cat, ids in by_cat.items():
        run = f"WS1010_COLLECT_{cat.upper()}"
        dump(run, {"run_id": run, "milestone": "webshop-2026-10-10-collect", **BASE,
                   **EXECUTORS["GPT"], "task_ids": ids, "memory": "none"})
        manifest[cat] = ids
    (OUT / "collect_sessions.json").write_text(json.dumps(
        {"instructions_sha256": instr["sha256"], "per_category": PER_CAT, "sessions": manifest}, indent=1) + "\n")
    print({c: len(v) for c, v in manifest.items()})


def evaluate(budget):
    jobs = [("WS1010", ex, exc) for ex, exc in EXECUTORS.items()] + [("WS1010R2", "GPT", EXECUTORS["GPT"])]
    for prefix, ex, exc in jobs:
        for cond in ("none", "true", "placebo", "irrelevant"):
            run = f"{prefix}_{cond.upper()}_{ex}"
            cfg = {"run_id": run, "milestone": "webshop-2026-10-10-eval", **BASE, **exc, "task_ids": TEST}
            if cond == "none":
                cfg["memory"] = "none"
            else:
                cfg.update({"memory": "expel", "retriever": "bm25", "memory_k": 20,
                            "token_budget": budget, "demo_pool": POOLS[cond]})
                if cond != "true":
                    cfg["rank_reference_pool"] = TRUE_POOL
            dump(run, cfg)
    print(f"wrote {len(jobs) * 4} eval configs, budget {budget}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["collect", "eval"])
    ap.add_argument("--budget", type=int)
    a = ap.parse_args()
    collect() if a.stage == "collect" else evaluate(a.budget)
