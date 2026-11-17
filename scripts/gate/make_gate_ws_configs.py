#!/usr/bin/env python3
"""Configs and the frozen schedule for the WebShop release-check runs (engineering/GATE_DESIGN_v3.md §5, v4 §5).

Reader Qwen3-8B (pinned snapshot), page label ON, 15 steps, seeded server, whole-pool injection, BM25 order by PUB.
  Panel T  test sessions 0-499:      NONE r1/r2, TRUE r1/r2
  Panel D  sessions 520-1019:        NONE r1/r2, TRUE r1/r2, ABL r1/r2, COMP1-3 (one repetition, split into halves)
Repeated arms: r1 in block 1, r2 in block 2, on different GPU chains. Each COMP_i's tasks are split into stratified
halves H_a / H_b (within each category, tasks shuffled with seed 20261116 then assigned alternately); COMP1 and COMP3
run H_a in block 1 and H_b in block 2, COMP2 the reverse; each half runs on the chain not used by that COMP's other
half. The order within each chain is drawn with seed 20261116 + 10 * block + chain.

Writes configs/gate_ws_2026_11/<RUN>.yaml, schedule.json and halves.json. Never overwrites.
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "configs/gate_ws_2026_11"
BASE = ROOT / "configs/r2_2026_10_31/R2031_P_L_PUB_TRUE_Q8.yaml"
PUB = "data/pools/webshop1010_insights.jsonl"
PAY = "data/pools/gate_ws_2026_11/PUB_{}.jsonl"
SNAPSHOT = "b968826d9c46dd6066d109eabc6255188de91218"   # Qwen/Qwen3-8B, as used in r2
SEED = 20261116
T_TASKS = list(range(0, 500))
D_TASKS = list(range(520, 1020))
INSTR = ROOT / "data/summaries/webshop_2026-10-10/instructions_seed0_1k.json"


def memory_block(kind: str) -> dict:
    if kind == "NONE":
        return {"memory": "none"}
    m = {"memory": "expel", "retriever": "bm25", "memory_k": 20, "token_budget": 524}
    if kind == "TRUE":
        return {**m, "demo_pool": PUB}
    return {**m, "demo_pool": PAY.format(kind), "rank_reference_pool": PUB}


def make(run_id: str, kind: str, tasks, rep: int, block: int, base: dict) -> dict:
    c = copy.deepcopy(base)
    for k in ("memory", "retriever", "memory_k", "token_budget", "demo_pool", "rank_reference_pool"):
        c.pop(k, None)
    c.update({"run_id": run_id, "milestone": "gate-ws-2026-11", "out_dir": "results/gate_ws_2026_11",
              "task_ids": list(tasks), "repetition": rep, "block": block, "model_revision": SNAPSHOT,
              "max_infra_reruns": 2, "resume": True, "webshop_page_label": True})
    c.update(memory_block(kind))
    return c


def halves():
    instr = json.loads(INSTR.read_text())["instructions"]
    by_cat = {}
    for t in D_TASKS:
        by_cat.setdefault(instr[str(t)]["category"], []).append(t)
    rng = random.Random(SEED)
    ha, hb = [], []
    for cat in sorted(by_cat):
        ts = sorted(by_cat[cat])
        rng.shuffle(ts)
        for j, t in enumerate(ts):
            (ha if j % 2 == 0 else hb).append(t)
    return sorted(ha), sorted(hb), {c: len(v) for c, v in sorted(by_cat.items())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.glob("GWS_*.yaml")) or (OUT / "schedule.json").exists():
        sys.exit(f"{OUT} already has configs; refusing to overwrite")
    base = yaml.safe_load(BASE.read_text())
    ha, hb, strata = halves()
    runs = {}

    def add(name, kind, tasks, rep, block):
        rid = f"GWS_{name}_Q8"
        runs[rid] = make(rid, kind, tasks, rep, block, base)
        return rid

    for panel, tasks in (("T", T_TASKS), ("D", D_TASKS)):
        kinds = ["NONE", "TRUE"] + (["ABL"] if panel == "D" else [])
        for k in kinds:
            for rep in (1, 2):
                add(f"{panel}_{k}_R{rep}", k, tasks, rep, rep)
    half_plan = {"COMP1": ("A", "B"), "COMP2": ("B", "A"), "COMP3": ("A", "B")}   # (block-1 half, block-2 half)
    for comp, (h1, h2) in half_plan.items():
        for block, h in ((1, h1), (2, h2)):
            add(f"D_{comp}_H{h}", comp, ha if h == "A" else hb, 1, block)
    chains = {
        1: {"A": ["T_NONE_R1", "D_NONE_R1", "D_ABL_R1", "D_COMP3_HA"],
            "B": ["T_TRUE_R1", "D_TRUE_R1", "D_COMP1_HA", "D_COMP2_HB"]},
        2: {"A": ["T_TRUE_R2", "D_TRUE_R2", "D_COMP1_HB", "D_COMP2_HA"],
            "B": ["T_NONE_R2", "D_NONE_R2", "D_ABL_R2", "D_COMP3_HB"]},
    }
    schedule = {"seed": SEED, "blocks": {}}
    for block, ch in chains.items():
        schedule["blocks"][str(block)] = {}
        for ci, (chain, names) in enumerate(sorted(ch.items())):
            order = list(names)
            random.Random(SEED + 10 * block + ci).shuffle(order)
            rids = [f"GWS_{n}_Q8" for n in order]
            assert all(r in runs and runs[r]["block"] == block for r in rids), rids
            schedule["blocks"][str(block)][chain] = rids
    scheduled = [r for b in schedule["blocks"].values() for c in b.values() for r in c]
    assert sorted(scheduled) == sorted(runs), (set(runs) ^ set(scheduled))
    for rid, c in runs.items():
        (OUT / f"{rid}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    (OUT / "schedule.json").write_text(json.dumps(schedule, indent=1) + "\n")
    (OUT / "halves.json").write_text(json.dumps({"seed": SEED, "strata": strata, "H_A": ha, "H_B": hb,
                                                 "plan": half_plan}, indent=1) + "\n")
    for b, ch in schedule["blocks"].items():
        for chain, rids in ch.items():
            (OUT / f"chain_B{b}_{chain}.txt").write_text("\n".join(f"configs/gate_ws_2026_11/{r}.yaml" for r in rids) + "\n")
    print(f"{len(runs)} run configs; halves {len(ha)}/{len(hb)}; strata {strata}")
    print(json.dumps(schedule, indent=1))


if __name__ == "__main__":
    main()
