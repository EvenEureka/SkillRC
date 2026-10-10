#!/usr/bin/env python3
"""Write the eight configs of the 2026-10-10 rerun of the placebo gate (PREREG_rerun_2026-10-10.md):
{none, true, placebo, irrelevant} x {gpt, qwen}, all identical to configs/papero_placebo_*.yaml except
for run_id, the memory pool and (for `none`) memory=none. Task ids are copied from the frozen gate."""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = {"gpt": ROOT / "configs/papero_placebo_gpt.yaml", "qwen": ROOT / "configs/papero_placebo_qwen.yaml"}
POOLS = {"true": "data/pools/expel_insights.jsonl", "placebo": "data/pools/expel_insights_order_placebo.jsonl",
         "irrelevant": "data/pools/expel_insights_irrelevant_coherent.jsonl"}
OUT = ROOT / "configs/rerun_2026_10_10"
OUT.mkdir(exist_ok=True)
for ex, path in BASE.items():
    base = yaml.safe_load(path.read_text())
    assert base["seeds"] == [0, 1, 5] and len(base["task_ids"]) == 60 and base["token_budget"] == 645
    for cond in ("none", "true", "placebo", "irrelevant"):
        cfg = copy.deepcopy(base)
        cfg["run_id"] = f"RERUN1010_{cond.upper()}_{ex.upper()}"
        cfg["milestone"] = "rerun-2026-10-10-irrelevant-control"
        cfg["save_trajectories"] = True
        if cond == "none":
            cfg["memory"] = "none"
            for k in ("retriever", "memory_k", "token_budget", "demo_pool", "rank_reference_pool"):
                cfg.pop(k, None)
        else:
            cfg["memory"] = "expel"
            cfg["demo_pool"] = POOLS[cond]
            cfg["rank_reference_pool"] = "data/pools/expel_insights.jsonl"
            if cond == "true":
                cfg.pop("rank_reference_pool")
        (OUT / f"{cond}_{ex}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
        print("wrote", OUT / f"{cond}_{ex}.yaml")
