#!/usr/bin/env python3
"""Qwen3-32B extension of the WebShop release check (engineering/GATE_DESIGN_v5_32b.md, PREREG addendum A).

  pilot  100 sessions from 1020-1499, stratified by category (proportional, seed 20261116): NONE, TRUE, one repetition
  T      test sessions 0-499: NONE r1/r2, TRUE r1/r2 (launched only if the pilot passes)
  D      sessions 520-1019: NONE r1/r2, TRUE r1/r2, ABL r1/r2, COMP1-3 halves (launched only if C1-32B's primary
         component is BLOCK)
Same harness and payloads as the 8B runs; reader Qwen3-32B (snapshot 9216db57), tensor parallel 2, so one chain:
every phase is a separate job, runs in a frozen order (seed 20261116 + phase index), r1 phases before r2 phases.
Writes configs/gate_ws32_2026_11/. Never overwrites.
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "configs/gate_ws_2026_11"
OUT = ROOT / "configs/gate_ws32_2026_11"
INSTR = ROOT / "data/summaries/webshop_2026-10-10/instructions_seed0_1k.json"
SNAPSHOT = "9216db5781bf21249d130ec9da846c4624c16137"
SEED = 20261116
PILOT_POOL = range(1020, 1500)
PILOT_N = 100


def pilot_sessions():
    instr = json.loads(INSTR.read_text())["instructions"]
    by_cat = {}
    for t in PILOT_POOL:
        by_cat.setdefault(instr[str(t)]["category"], []).append(t)
    total = sum(len(v) for v in by_cat.values())
    quota = {c: max(1, round(PILOT_N * len(v) / total)) for c, v in by_cat.items()}
    while sum(quota.values()) > PILOT_N:
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < PILOT_N:
        quota[max(quota, key=quota.get)] += 1
    rng = random.Random(SEED)
    picked = []
    for c in sorted(by_cat):
        picked += rng.sample(sorted(by_cat[c]), quota[c])
    return sorted(picked), quota


def to32(c8: dict, run_id: str, **upd) -> dict:
    c = copy.deepcopy(c8)
    c.update({"run_id": run_id, "model": "qwen3-32b", "model_revision": SNAPSHOT,
              "out_dir": "results/gate_ws32_2026_11", "milestone": "gate-ws32-2026-11", **upd})
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.glob("GWS32_*.yaml")):
        sys.exit(f"{OUT} already has configs; refusing to overwrite")
    runs, phases = {}, {}
    pilot, quota = pilot_sessions()
    for kind in ("NONE", "TRUE"):
        c8 = yaml.safe_load((SRC / f"GWS_T_{kind}_R1_Q8.yaml").read_text())
        rid = f"GWS32_P_{kind}_Q32"
        runs[rid] = to32(c8, rid, task_ids=pilot, repetition=1, block=1)
    for f in sorted(SRC.glob("GWS_*_Q8.yaml")):
        c8 = yaml.safe_load(f.read_text())
        rid = c8["run_id"].replace("GWS_", "GWS32_").replace("_Q8", "_Q32")
        runs[rid] = to32(c8, rid)
    order = lambda names, k: random.Random(SEED + k).sample(names, len(names))  # noqa: E731
    phases["pilot"] = order(["GWS32_P_NONE_Q32", "GWS32_P_TRUE_Q32"], 0)
    phases["T1"] = order(["GWS32_T_NONE_R1_Q32", "GWS32_T_TRUE_R1_Q32"], 1)
    phases["T2"] = order(["GWS32_T_NONE_R2_Q32", "GWS32_T_TRUE_R2_Q32"], 2)
    phases["D1"] = order(["GWS32_D_NONE_R1_Q32", "GWS32_D_TRUE_R1_Q32", "GWS32_D_ABL_R1_Q32",
                          "GWS32_D_COMP1_HA_Q32", "GWS32_D_COMP2_HB_Q32", "GWS32_D_COMP3_HA_Q32"], 3)
    phases["D2"] = order(["GWS32_D_NONE_R2_Q32", "GWS32_D_TRUE_R2_Q32", "GWS32_D_ABL_R2_Q32",
                          "GWS32_D_COMP1_HB_Q32", "GWS32_D_COMP2_HA_Q32", "GWS32_D_COMP3_HB_Q32"], 4)
    scheduled = [r for p in phases.values() for r in p]
    assert sorted(scheduled) == sorted(runs), set(runs) ^ set(scheduled)
    for p, rids in phases.items():
        blocks = {runs[r]["block"] for r in rids}
        assert len(blocks) == 1, (p, blocks)
    for rid, c in runs.items():
        (OUT / f"{rid}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    for p, rids in phases.items():
        (OUT / f"chain_{p}.txt").write_text("\n".join(f"configs/gate_ws32_2026_11/{r}.yaml" for r in rids) + "\n")
    (OUT / "schedule.json").write_text(json.dumps({"seed": SEED, "phases": phases, "pilot_sessions": pilot,
                                                    "pilot_quota": quota}, indent=1) + "\n")
    # audit carry-over: apart from reader identity and output location, every 32B config equals its 8B counterpart
    # (same harness, payload files, retrieval and budget), so the 8B injection audit (same Qwen3 tokenizer) applies.
    reader_keys = {"run_id", "model", "model_revision", "out_dir", "milestone"}
    for rid, c in runs.items():
        if rid.startswith("GWS32_P_"):
            c8 = yaml.safe_load((SRC / f"GWS_T_{rid.split('_')[2]}_R1_Q8.yaml").read_text())
            c8 = {**c8, "task_ids": c["task_ids"]}
        else:
            c8 = yaml.safe_load((SRC / f"{rid.replace('GWS32_', 'GWS_').replace('_Q32', '_Q8')}.yaml").read_text())
        diff = sorted(k for k in set(c) | set(c8) if k not in reader_keys and c.get(k) != c8.get(k))
        assert not diff, (rid, diff)
    (OUT / "AUDIT_OK").write_text("32B configs equal their 8B counterparts except reader identity and output; "
                                  "8B injection audit: results/gate_ws_2026_11/injection_audit.json\n")
    print(len(runs), "configs; pilot quota", quota)
    print(json.dumps(phases, indent=1))


if __name__ == "__main__":
    main()
