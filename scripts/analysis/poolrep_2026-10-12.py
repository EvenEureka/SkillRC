#!/usr/bin/env python3
"""Pool-replication analysis, written before any episode.

Reader Qwen3-8B, the 134 valid_unseen games, seeds [0, 1, 5]. Runs PRP1012_<arm>_QWEN.
Per pool p in GREF, Q32R0..3: R_p = TRUE - IRR (content value), T_p = TRUE - NONE, I_p = IRR - NONE.
Replicates Q32R1-3: R-bar, T-bar, I-bar (per-task mean over replicates, then bootstrapped), the range and SD of R,
the three pairwise R_i - R_j. Noise floor DUP = Q32R1_TRUE - Q32R1_TRUEDUP and the share of differing episodes.
Secondary: R_Q32R0 - R_GREF (distiller, input fixed), T_GREF661 (the Qwen-sourced pool at a 661-token budget).
Equal-pair task means; stratified task bootstrap by task type, 20,000 draws, seed 20260713 (phase-2 helpers).
Errored episodes count as failures; runs with > 2% errors are flagged and their R recomputed without errored pairs.
Works on partial outputs (missing or incomplete runs are reported as such).
Output: results/placebo/poolrep_2026-10-12_results.json
Run:    python scripts/analysis/poolrep_2026-10-12.py"""
from __future__ import annotations

import importlib.util
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("p2", ROOT / "scripts/analysis/analyze_phase2_2026-10-10.py")
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)
TASKS = list(range(134))
SEEDS = p2.SEEDS
POOLS = ["GREF", "Q32R0", "Q32R1", "Q32R2", "Q32R3"]
REPS = ["Q32R1", "Q32R2", "Q32R3"]
ARMS = ["NONE", "GREF661_TRUE", "Q32R1_TRUEDUP"] + [f"{p}_{c}" for p in POOLS for c in ("TRUE", "IRR")]
OUT = ROOT / "results/placebo/poolrep_2026-10-12_results.json"


def rid(arm):
    return f"PRP1012_{arm}_QWEN"


def boot(values, types):
    return p2.boot(values, types)


def task_types():
    types = {}
    for run in ("SRC1011_NONE_QWEN", rid("NONE")):
        for (_, t), r in (p2.load(run) or {}).items():
            types.setdefault(t, r["task_type"])
    assert sorted(types) == TASKS, "task types unknown for some games"
    return types


def load_all():
    rows, status = {}, {}
    for arm in ARMS:
        r = p2.load(rid(arm))
        if r is None:
            status[arm] = "missing"
            continue
        missing = [(s, t) for s in SEEDS for t in TASKS if (s, t) not in r]
        if missing:
            status[arm] = f"incomplete ({len(r)}/{len(SEEDS) * len(TASKS)} episodes)"
            continue
        n_err = sum(1 for s in SEEDS for t in TASKS if r[(s, t)].get("error"))
        status[arm] = {"episodes": len(SEEDS) * len(TASKS), "errors": n_err, "flag_errors_gt_2pct": n_err > 0.02 * 402,
                       "success": round(float(np.mean([r[(s, t)]["success"] for s in SEEDS for t in TASKS])), 4),
                       "memory_tokens": sorted({int(r[(s, t)].get("memory_tokens", 0)) for s in SEEDS for t in TASKS})}
        rows[arm] = r
    return rows, status


def tmeans(r):
    return {t: float(np.mean([r[(s, t)]["success"] for s in SEEDS])) for t in TASKS}


def diff(a, b):
    return {t: a[t] - b[t] for t in TASKS}


def diff_excluding_errors(ra, rb):
    """Per-task mean of (a - b) over the seeds where neither episode errored; tasks with none left are dropped."""
    out = {}
    for t in TASKS:
        ok = [s for s in SEEDS if not ra[(s, t)].get("error") and not rb[(s, t)].get("error")]
        if ok:
            out[t] = float(np.mean([ra[(s, t)]["success"] - rb[(s, t)]["success"] for s in ok]))
    return out


def by_type(values, types):
    g = defaultdict(list)
    for t, v in values.items():
        g[types[t]].append(v)
    return {k: round(float(np.mean(v)), 4) for k, v in sorted(g.items())}


def main():
    types = task_types()
    rows, status = load_all()
    tm = {arm: tmeans(r) for arm, r in rows.items()}
    out = {"runs": {rid(a): status[a] for a in ARMS},
           "pools": {}, "replicates": {}, "secondary": {}, "audits": {}}
    # pool sizes and control audits (written by the distiller / builder)
    for p in POOLS:
        a = ROOT / ("results/placebo/src1011_qwen_pool_audit.json" if p == "GREF" else f"results/placebo/poolrep_{p.lower()}_audit.json")
        c = ROOT / f"results/placebo/poolrep_{p.lower()}_irrelevant_audit.json"
        info = {}
        if a.exists():
            d = json.loads(a.read_text())
            info.update({"n_items": d.get("n_items"), "footprint_o200k": d.get("footprint_o200k", d.get("footprint_gpt4omini")),
                         "insights_per_type": {t: v.get("n_insights") for t, v in d.get("types", {}).items()}})
        if c.exists():
            d = json.loads(c.read_text())
            info["control"] = {"nonce_replacements": d["nonce_replacements"], "retained_body_fraction": d["retained_body_fraction"],
                               "context_tokens": d["context_tokens"], "test_game_vocab_overlap": d["test_game_vocab_overlap"]}
        out["audits"][p] = info
    # memory-length audit: TRUE and IRR of a pool must inject the same number of o200k tokens in every episode
    for p in POOLS:
        a, b = f"{p}_TRUE", f"{p}_IRR"
        if a in rows and b in rows:
            mism = sum(1 for s in SEEDS for t in TASKS
                       if int(rows[a][(s, t)].get("memory_tokens", -1)) != int(rows[b][(s, t)].get("memory_tokens", -2)))
            out["audits"].setdefault(p, {})["memory_tokens_mismatched_episodes"] = mism
    # per-pool estimands
    for p in POOLS:
        a, b = f"{p}_TRUE", f"{p}_IRR"
        blk = {}
        if a in tm and b in tm:
            blk["R"] = boot(diff(tm[a], tm[b]), types)
            blk["R_by_type"] = by_type(diff(tm[a], tm[b]), types)
            if status[a]["flag_errors_gt_2pct"] or status[b]["flag_errors_gt_2pct"]:
                blk["R_excluding_errored_pairs"] = boot(diff_excluding_errors(rows[a], rows[b]), types)
        if a in tm and "NONE" in tm:
            blk["T"] = boot(diff(tm[a], tm["NONE"]), types)
        if b in tm and "NONE" in tm:
            blk["I"] = boot(diff(tm[b], tm["NONE"]), types)
        for arm in (a, b):
            if arm in tm:
                blk[f"{arm.split('_')[-1].lower()}_mean"] = round(float(np.mean(list(tm[arm].values()))), 4)
        out["pools"][p] = blk
    if "NONE" in tm:
        out["none_mean"] = round(float(np.mean(list(tm["NONE"].values()))), 4)
    # replicates
    have = [p for p in REPS if f"{p}_TRUE" in tm and f"{p}_IRR" in tm]
    rep = {"available": have}
    if len(have) == len(REPS):
        Rp = {p: diff(tm[f"{p}_TRUE"], tm[f"{p}_IRR"]) for p in REPS}
        rep["R_bar"] = boot({t: float(np.mean([Rp[p][t] for p in REPS])) for t in TASKS}, types)
        Rm = [out["pools"][p]["R"]["mean"] for p in REPS]
        rep["R_values"] = dict(zip(REPS, Rm))
        rep["R_range"] = round(max(Rm) - min(Rm), 4)
        rep["R_sd"] = round(float(np.std(Rm, ddof=1)), 4)
        rep["pairwise"] = {f"{i}-{j}": boot({t: Rp[i][t] - Rp[j][t] for t in TASKS}, types) for i, j in itertools.combinations(REPS, 2)}
        if "NONE" in tm:
            rep["T_bar"] = boot({t: float(np.mean([tm[f"{p}_TRUE"][t] for p in REPS])) - tm["NONE"][t] for t in TASKS}, types)
            rep["I_bar"] = boot({t: float(np.mean([tm[f"{p}_IRR"][t] for p in REPS])) - tm["NONE"][t] for t in TASKS}, types)
    out["replicates"] = rep
    # secondary
    sec = {}
    if "Q32R1_TRUE" in rows and "Q32R1_TRUEDUP" in rows:
        a, b = rows["Q32R1_TRUE"], rows["Q32R1_TRUEDUP"]
        sec["DUP"] = boot(diff(tm["Q32R1_TRUE"], tm["Q32R1_TRUEDUP"]), types)
        sec["DUP_share_episodes_differing"] = round(float(np.mean([a[(s, t)]["success"] != b[(s, t)]["success"] for s in SEEDS for t in TASKS])), 4)
    if all(f"{p}_{c}" in tm for p in ("Q32R0", "GREF") for c in ("TRUE", "IRR")):
        sec["distiller_R_Q32R0_minus_R_GREF"] = boot({t: (tm["Q32R0_TRUE"][t] - tm["Q32R0_IRR"][t]) - (tm["GREF_TRUE"][t] - tm["GREF_IRR"][t]) for t in TASKS}, types)
    if "GREF661_TRUE" in tm and "NONE" in tm:
        sec["T_GREF661"] = boot(diff(tm["GREF661_TRUE"], tm["NONE"]), types)
    if "GREF661_TRUE" in tm and "GREF_TRUE" in tm:
        sec["GREF_whole_minus_661"] = boot(diff(tm["GREF_TRUE"], tm["GREF661_TRUE"]), types)
    out["secondary"] = sec
    # frozen predictions
    pred = {}
    if "R_bar" in rep:
        pred["P1 R_bar > 0, CI excludes 0"] = rep["R_bar"]["ci95"][0] > 0
        pred["P2 range of R over Q32R1-3 < 0.05"] = rep["R_range"] < 0.05
        pred["P3 no pairwise R_i - R_j CI excludes 0"] = all(v["ci95"][0] <= 0 <= v["ci95"][1] for v in rep["pairwise"].values())
    if "I_bar" in rep:
        pred["P4 |I_bar| <= 0.04"] = abs(rep["I_bar"]["mean"]) <= 0.04
    if "distiller_R_Q32R0_minus_R_GREF" in sec:
        ci = sec["distiller_R_Q32R0_minus_R_GREF"]["ci95"]
        pred["secondary: distiller alone changes R (CI excludes 0)"] = not (ci[0] <= 0 <= ci[1])
    if "DUP" in sec:
        pred["secondary: DUP CI includes 0"] = sec["DUP"]["ci95"][0] <= 0 <= sec["DUP"]["ci95"][1]
        pred["secondary: |DUP| >= 0.03 (a 0.05 pool range is within run noise)"] = abs(sec["DUP"]["mean"]) >= 0.03
    out["predictions"] = pred
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("runs", "pools", "replicates", "secondary", "predictions") if k in out}, indent=1)[:6000])


if __name__ == "__main__":
    main()
