#!/usr/bin/env python3
"""Evaluation configs for the pool-replication study (Qwen3-8B reader).

Every setting is copied from configs/source_2026_10_11/SRC1011_{NONE,POOLQ}_QWEN.yaml (134 valid_unseen games,
seeds [0, 1, 5], ReAct, 2 demos, admissible actions, T = 0, 256 tokens, 30 steps, memory expel, BM25, k = 20);
only run_id, milestone, demo_pool, token_budget and rank_reference_pool change.
- <POOL>_TRUE / <POOL>_IRR: the pool and its gardening control, injected whole (token_budget = the pool's o200k
  footprint, which the control matches); the control ranks by the true pool (rank_reference_pool).
- GREF661_TRUE: the SRC1011_POOLQ_QWEN config unchanged (the Qwen-sourced pool at B = 661), a same-window anchor.
- Q32R1_TRUEDUP: Q32R1_TRUE again, run on the other server (noise floor).
Writes configs/poolrep_2026_10_12/PRP1012_<arm>_QWEN.yaml and prints the job split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skillrc.tokens import count_tokens  # noqa: E402

SRC = ROOT / "configs/source_2026_10_11"
OUT = ROOT / "configs/poolrep_2026_10_12"
POOLS = {"GREF": "data/pools/src1011_qwen_insights.jsonl",
         **{f"Q32R{r}": f"data/pools/poolrep_q32r{r}_insights.jsonl" for r in range(4)}}
CONTROL = {"GREF": "data/pools/poolrep_gref_irrelevant.jsonl",
           **{f"Q32R{r}": f"data/pools/poolrep_q32r{r}_irrelevant.jsonl" for r in range(4)}}
JOBS = {"A": ["NONE", "GREF661_TRUE", "GREF_TRUE", "GREF_IRR", "Q32R0_TRUE", "Q32R0_IRR", "Q32R1_TRUEDUP"],
        "B": ["Q32R1_TRUE", "Q32R1_IRR", "Q32R2_TRUE", "Q32R2_IRR", "Q32R3_TRUE", "Q32R3_IRR"]}


def footprint(path: str) -> int:
    items = [json.loads(x) for x in (ROOT / path).read_text().splitlines() if x.strip()]
    return count_tokens("\n\n".join(it["text"] for it in items), "gpt-4o-mini")


def main():
    OUT.mkdir(exist_ok=True)
    none = yaml.safe_load((SRC / "SRC1011_NONE_QWEN.yaml").read_text())
    poolq = yaml.safe_load((SRC / "SRC1011_POOLQ_QWEN.yaml").read_text())
    assert poolq["token_budget"] == 661 and poolq["demo_pool"] == POOLS["GREF"]
    cfgs = {"NONE": dict(none), "GREF661_TRUE": dict(poolq)}
    budgets = {}
    for name, pool in POOLS.items():
        b = footprint(pool)
        assert footprint(CONTROL[name]) == b, f"{name}: control footprint differs from the pool's"
        budgets[name] = b
        cfgs[f"{name}_TRUE"] = dict(poolq, demo_pool=pool, token_budget=b)
        cfgs[f"{name}_IRR"] = dict(poolq, demo_pool=CONTROL[name], token_budget=b, rank_reference_pool=pool)
    cfgs["Q32R1_TRUEDUP"] = dict(cfgs["Q32R1_TRUE"])
    for arm in [a for job in JOBS.values() for a in job]:
        cfg = dict(cfgs[arm], run_id=f"PRP1012_{arm}_QWEN", milestone="poolrep-2026-10-12")
        assert cfg["model"] == "qwen3-8b" and cfg["api_key"] == "EMPTY"
        assert cfg["base_url"].startswith(("http://localhost", "http://127.0.0.1")), cfg["base_url"]
        assert cfg["task_ids"] == list(range(134)) and cfg["seeds"] == [0, 1, 5]
        assert cfg.get("retriever", "bm25") == "bm25"
        path = OUT / f"PRP1012_{arm}_QWEN.yaml"
        if path.exists() and yaml.safe_load(path.read_text()) != cfg:
            sys.exit(f"refusing to overwrite a different {path}")
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=110))
    print("budgets (o200k footprint = token_budget):", budgets)
    for job, arms in JOBS.items():
        print(f"job {job}:", ":".join(f"configs/poolrep_2026_10_12/PRP1012_{a}_QWEN.yaml" for a in arms))


if __name__ == "__main__":
    main()
