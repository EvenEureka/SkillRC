#!/usr/bin/env python3
"""W1-B (generic-rule study) analysis, written before any evaluation episode.

Readers Qwen3-8B (Q8) and Qwen3-14B (Q14); focal pools FOCAL (expel_insights, 645) and GREF (src1011_qwen_insights, 706).
Runs: W1B1013_FOCAL_{TRUE,IRR,GENERIC,IRR2}_<rd>, W1B1013_GREF_{GENERIC,IRR2}_<rd>, and from W1-A (same jobs, same window)
W1A1013_{NONE,GREF_TRUE,GREF_IRR}_<rd>. Equal-pair task means over seeds [0, 1, 5].
Per (focal f, reader rd): R = TRUE - IRR, E = TRUE - GENERIC, G = GENERIC - IRR, dI = IRR2 - IRR, I = IRR - NONE,
I2 = IRR2 - NONE. Pooled E-bar, G-bar, R-bar, dI-bar = per-task means over the four (f, rd) cells. Stratified task
bootstrap by type, 20,000 draws, seed 20261013 (one task draw shared by all cells), percentile 95% intervals.
Descriptive: per-type values; share of episodes whose first action differs between GENERIC and TRUE; payload
vocabulary overlap (from the control audits). Errored episodes count as failures; runs with > 2% errors are flagged.
Output: results/placebo/w1b_2026-10-13_results.json
Run:    python scripts/analysis/w1b_2026-10-13.py"""
from __future__ import annotations

import importlib.util
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
READERS = ("Q8", "Q14")
FOCALS = ("FOCAL", "GREF")
COND = ("NONE", "TRUE", "IRR", "GENERIC", "IRR2")
N_BOOT, SEED = 20_000, 20261013
OUT = ROOT / "results/placebo/w1b_2026-10-13_results.json"
CONTRASTS = {"R": ("TRUE", "IRR"), "E": ("TRUE", "GENERIC"), "G": ("GENERIC", "IRR"), "dI": ("IRR2", "IRR"),
             "I": ("IRR", "NONE"), "I2": ("IRR2", "NONE"), "T": ("TRUE", "NONE"), "T_generic": ("GENERIC", "NONE")}


def rid(f, c, rd):
    if c == "NONE":
        return f"W1A1013_NONE_{rd}"
    if f == "GREF" and c in ("TRUE", "IRR"):
        return f"W1A1013_GREF_{c}_{rd}"
    return f"W1B1013_{f}_{c}_{rd}"


def task_types():
    types = {}
    for run in ["SRC1011_NONE_QWEN"] + [rid(None, "NONE", rd) for rd in READERS]:
        for (_, t), r in (p2.load(run) or {}).items():
            types.setdefault(t, r["task_type"])
    assert sorted(types) == TASKS
    return types


class Boot:
    def __init__(self, types):
        rng = np.random.default_rng(SEED)
        strata = defaultdict(list)
        for t in TASKS:
            strata[types[t]].append(t)
        self.counts = np.zeros((N_BOOT, len(TASKS)))
        for _, ts in sorted(strata.items()):
            ts = np.asarray(ts)
            draws = rng.integers(0, len(ts), size=(N_BOOT, len(ts)))
            for j in range(len(ts)):
                np.add.at(self.counts, (np.arange(N_BOOT), ts[draws[:, j]]), 1)

    def summ(self, arr):
        draws = self.counts @ arr / len(TASKS)
        p = 2 * min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0)))
        return {"mean": round(float(arr.mean()), 4), "ci95": [round(float(np.quantile(draws, .025)), 4),
                                                              round(float(np.quantile(draws, .975)), 4)], "p_boot": round(min(1.0, p), 4)}


def by_type(arr, types):
    g = defaultdict(list)
    for t in TASKS:
        g[types[t]].append(arr[t])
    return {k: round(float(np.mean(v)), 4) for k, v in sorted(g.items())}


def first_action(r):
    traj = r.get("trajectory") or []
    for st in traj:
        a = str(st.get("action", ""))
        if not a.startswith("think"):
            return a.strip()
    return None


def main():
    types = task_types()
    B = Boot(types)
    rows, status, tm = {}, {}, {}
    for rd in READERS:
        for f in FOCALS:
            for c in COND:
                run = rid(f, c, rd)
                if run in status:
                    continue
                r = p2.load(run)
                if r is None:
                    status[run] = "missing"
                    continue
                if any((s, t) not in r for s in SEEDS for t in TASKS):
                    status[run] = f"incomplete ({len(r)}/402 episodes)"
                    continue
                n_err = sum(1 for s in SEEDS for t in TASKS if r[(s, t)].get("error"))
                status[run] = {"episodes": 402, "errors": n_err, "flag_errors_gt_2pct": n_err > 0.02 * 402,
                               "success": round(float(np.mean([r[(s, t)]["success"] for s in SEEDS for t in TASKS])), 4),
                               "memory_tokens": sorted({int(r[(s, t)].get("memory_tokens", 0)) for s in SEEDS for t in TASKS})}
                rows[run] = r
                tm[run] = np.array([np.mean([r[(s, t)]["success"] for s in SEEDS]) for t in TASKS])
    out = {"runs": status, "cells": {}, "pooled": {},
           "descriptive": {}, "decisions": {}}
    diffs = defaultdict(dict)
    for rd in READERS:
        for f in FOCALS:
            cell = {"means": {c: round(float(tm[rid(f, c, rd)].mean()), 4) for c in COND if rid(f, c, rd) in tm}}
            for name, (a, b) in CONTRASTS.items():
                ra, rb = rid(f, a, rd), rid(f, b, rd)
                if ra in tm and rb in tm:
                    d = tm[ra] - tm[rb]
                    diffs[name][(f, rd)] = d
                    cell[name] = B.summ(d)
                    cell[f"{name}_by_type"] = by_type(d, types)
            # memory-length audit: every payload arm injects the true pool's length in every episode
            rt = rid(f, "TRUE", rd)
            for c in ("IRR", "GENERIC", "IRR2"):
                rc = rid(f, c, rd)
                if rt in rows and rc in rows:
                    cell.setdefault("memory_tokens_mismatch", {})[c] = sum(
                        1 for s in SEEDS for t in TASKS
                        if int(rows[rt][(s, t)].get("memory_tokens", -1)) != int(rows[rc][(s, t)].get("memory_tokens", -2)))
            # behaviour: first action differs, GENERIC vs TRUE
            rg = rid(f, "GENERIC", rd)
            if rt in rows and rg in rows:
                pairs = [(first_action(rows[rt][(s, t)]), first_action(rows[rg][(s, t)])) for s in SEEDS for t in TASKS]
                cell["first_action_differs_generic_vs_true"] = round(float(np.mean([x != y for x, y in pairs])), 4)
            out["cells"][f"{f}_{rd}"] = cell
    for name in CONTRASTS:
        got = diffs[name]
        if len(got) == 4:
            out["pooled"][f"{name}_bar"] = B.summ(np.mean([got[k] for k in sorted(got)], axis=0))
        for rd in READERS:
            ks = [(f, rd) for f in FOCALS if (f, rd) in got]
            if len(ks) == 2:
                out["pooled"][f"{name}_bar_{rd}"] = B.summ(np.mean([got[k] for k in ks], axis=0))
    for f in ("focal", "gref"):
        a = ROOT / f"results/placebo/w1b_{f}_generic_audit.json"
        if a.exists():
            d = json.loads(a.read_text())
            out["descriptive"][f"{f}_generic_audit"] = {k: d[k] for k in ("nonce_replacements", "retained_body_fraction",
                                                                         "context_tokens", "test_game_vocab_overlap")}
        a = ROOT / f"results/placebo/w1b_{f}_astronomy_audit.json"
        if a.exists():
            d = json.loads(a.read_text())
            out["descriptive"][f"{f}_astronomy_audit"] = {k: d[k] for k in ("nonce_replacements", "retained_body_fraction",
                                                                           "context_tokens", "test_game_vocab_overlap")}
    P, dec = out["pooled"], out["decisions"]
    if "E_bar" in P and "G_bar" in P:
        elo, ehi = P["E_bar"]["ci95"]
        glo, ghi = P["G_bar"]["ci95"]
        dec["B1"] = "holds" if elo > 0 else "fails"
        dec["B2"] = "holds" if glo > 0 else "fails"
        dec["reading"] = ("experience adds value beyond generic advice" if elo > 0 else
                          "generic advice beats distilled experience" if ehi < 0 else
                          "generic advice reproduces the content value (R measures task-relevant advice, not experience)" if glo > 0 else
                          f"not resolved at this power: experience adds at most {ehi:+.3f} beyond generic advice")
    if "dI_bar" in P:
        lo, hi = P["dI_bar"]["ci95"]
        dec["B3"] = "holds" if abs(P["dI_bar"]["mean"]) < 0.03 and lo <= 0 <= hi else "fails"
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    for k in ("R_bar", "E_bar", "G_bar", "dI_bar", "I_bar", "I2_bar"):
        if k in P:
            print(f"{k:8s} {P[k]['mean']:+.3f} {P[k]['ci95']}")
    print("decisions:", json.dumps(dec))
    print("->", OUT)


if __name__ == "__main__":
    main()
