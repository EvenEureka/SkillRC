#!/usr/bin/env python3
"""W1-A (crossover study) analysis, written before any evaluation episode.

Runs W1A1013_<arm>_<Q8|Q14|Q32>; 134 valid_unseen games, seeds [0, 1, 5]; equal-pair task means over seeds.
Per pool p and reader: R = TRUE - IRR, I = IRR - NONE, T = TRUE - NONE (stratified task bootstrap by type).
Pooled estimands use the planned hierarchical bootstrap (20,000 draws, seed 20261013): each draw resamples the
134 tasks within type strata (one draw shared by all readers and arms) and, per source, its three replicate subsamples
with replacement (shared by both distillers and all readers). The task-only interval (pools fixed) is reported too.
  E1 R-bar per reader and per cell     E2 own-source interaction Delta_own (sources Q8,Q32 x readers Q8,Q32)
  E3 source main effect and contrasts  E4 distiller main effect D32 - D8 (paired on subsample), overall and per reader
  E5 heterogeneity: Q = sum_c d' S^-1 d over the 18 (source, distiller, reader) cells, chi2(2 per cell); moment tau^2
  E6 I-bar per reader                  E7 reference crossover R_GREF - R_GPTREF, T_GREF - T_GPTREF per reader
  noise floor DUP = Q8_D32_R1_TRUE - Q8_D32_R1_TRUEDUP per reader; Holm over {E2, R_Q8 - R_Q32, E3 Q8-GPT, E4}
Errored episodes count as failures; runs with > 2% errored episodes are flagged and their pool's R is recomputed
without errored pairs. Works on partial outputs (estimands that need a missing run are reported as not available).
Output: results/placebo/w1a_2026-10-13_results.json
Run:    python scripts/analysis/w1a_2026-10-13.py"""
from __future__ import annotations

import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("p2", ROOT / "scripts/analysis/analyze_phase2_2026-10-10.py")
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)
TASKS = list(range(134))
SEEDS = p2.SEEDS
READERS = ("Q8", "Q14", "Q32")
SOURCES, DISTS, REPS = ("GPT", "Q8", "Q32"), ("D32", "D8"), (1, 2, 3)
BAL = [f"{s}_{d}_R{r}" for s in SOURCES for d in DISTS for r in REPS]
REFS = ("GPTREF", "GREF")
POOLS = BAL + list(REFS)
ARMS = ["NONE", "Q8_D32_R1_TRUEDUP"] + [f"{p}_{c}" for p in POOLS for c in ("TRUE", "IRR")]
N_BOOT, SEED = 20_000, 20261013
OUT = ROOT / "results/placebo/w1a_2026-10-13_results.json"


def rid(arm, rd):
    return f"W1A1013_{arm}_{rd}"


def task_types():
    types = {}
    for run in ["SRC1011_NONE_QWEN"] + [rid("NONE", rd) for rd in READERS]:
        for (_, t), r in (p2.load(run) or {}).items():
            types.setdefault(t, r["task_type"])
    assert sorted(types) == TASKS, "task types unknown for some games"
    return types


def load_all():
    rows, status = {}, {}
    for rd in READERS:
        for arm in ARMS:
            key = (rd, arm)
            r = p2.load(rid(arm, rd))
            if r is None:
                status[key] = "missing"
                continue
            missing = [(s, t) for s in SEEDS for t in TASKS if (s, t) not in r]
            if missing:
                status[key] = f"incomplete ({len(r)}/{len(SEEDS) * len(TASKS)} episodes)"
                continue
            n_err = sum(1 for s in SEEDS for t in TASKS if r[(s, t)].get("error"))
            status[key] = {"episodes": len(SEEDS) * len(TASKS), "errors": n_err, "flag_errors_gt_2pct": n_err > 0.02 * 402,
                           "success": round(float(np.mean([r[(s, t)]["success"] for s in SEEDS for t in TASKS])), 4),
                           "memory_tokens": sorted({int(r[(s, t)].get("memory_tokens", 0)) for s in SEEDS for t in TASKS})}
            rows[key] = r
    return rows, status


def tmean(r):
    return np.array([np.mean([r[(s, t)]["success"] for s in SEEDS]) for t in TASKS], dtype=float)


def tmean_excl(ra, rb):
    """per-task mean of (a - b) over seeds where neither errored; tasks with none left -> nan"""
    out = []
    for t in TASKS:
        ok = [s for s in SEEDS if not ra[(s, t)].get("error") and not rb[(s, t)].get("error")]
        out.append(np.mean([ra[(s, t)]["success"] - rb[(s, t)]["success"] for s in ok]) if ok else np.nan)
    return np.array(out)


class Boot:
    """Shared resampling plan: task-count matrix (stratified) and per-source replicate choices."""

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
        self.reps = {s: rng.integers(0, 3, size=(N_BOOT, 3)) for s in SOURCES}  # slot -> replicate index 0..2
        self.types = types

    def task_means(self, arr):
        """arr: (134,) per-task values -> (N_BOOT,) bootstrap means (nan-safe: nan tasks dropped per draw)"""
        if np.isnan(arr).any():
            ok = ~np.isnan(arr)
            return (self.counts[:, ok] @ arr[ok]) / self.counts[:, ok].sum(1)
        return self.counts @ arr / len(TASKS)


def summ(point, draws):
    draws = np.asarray(draws)
    p = 2 * min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0)))
    return {"mean": round(float(point), 4), "ci95": [round(float(np.quantile(draws, .025)), 4), round(float(np.quantile(draws, .975)), 4)],
            "p_boot": round(min(1.0, p), 4)}


def by_type(arr, types):
    g = defaultdict(list)
    for t in TASKS:
        g[types[t]].append(arr[t])
    return {k: round(float(np.mean(v)), 4) for k, v in sorted(g.items())}


def chi2_sf_even(x, df):
    k = df // 2
    term, s = 1.0, 1.0
    for i in range(1, k):
        term *= (x / 2) / i
        s += term
    return float(math.exp(-x / 2) * s)


def main():
    types = task_types()
    rows, status = load_all()
    tm = {key: tmean(r) for key, r in rows.items()}
    out = {"runs": {rid(arm, rd): status[(rd, arm)] for rd in READERS for arm in ARMS},
           "none_mean": {rd: round(float(tm[(rd, "NONE")].mean()), 4) for rd in READERS if (rd, "NONE") in tm},
           "pools": {}, "pooled": {}, "heterogeneity": {}, "reference": {}, "noise_floor": {}, "audits": {}, "decisions": {}}
    B = Boot(types)
    # per-pool, per-reader arrays
    Rarr, Iarr, Tarr = {}, {}, {}
    for rd in READERS:
        for p in POOLS:
            a, b = (rd, f"{p}_TRUE"), (rd, f"{p}_IRR")
            blk = {}
            if a in tm and b in tm:
                Rarr[(rd, p)] = tm[a] - tm[b]
                flagged = status[a]["flag_errors_gt_2pct"] or status[b]["flag_errors_gt_2pct"]
                blk["R"] = summ(Rarr[(rd, p)].mean(), B.task_means(Rarr[(rd, p)]))
                blk["R_by_type"] = by_type(Rarr[(rd, p)], types)
                if flagged:
                    ex = tmean_excl(rows[a], rows[b])
                    blk["R_excluding_errored_pairs"] = summ(np.nanmean(ex), B.task_means(ex))
                out["audits"].setdefault(p, {})[f"memory_tokens_mismatch_{rd}"] = sum(
                    1 for s in SEEDS for t in TASKS
                    if int(rows[a][(s, t)].get("memory_tokens", -1)) != int(rows[b][(s, t)].get("memory_tokens", -2)))
            if (rd, "NONE") in tm:
                if b in tm:
                    Iarr[(rd, p)] = tm[b] - tm[(rd, "NONE")]
                    blk["I"] = summ(Iarr[(rd, p)].mean(), B.task_means(Iarr[(rd, p)]))
                if a in tm:
                    Tarr[(rd, p)] = tm[a] - tm[(rd, "NONE")]
                    blk["T"] = summ(Tarr[(rd, p)].mean(), B.task_means(Tarr[(rd, p)]))
            for arm in (a, b):
                if arm in tm:
                    blk[f"{arm[1].split('_')[-1].lower()}_mean"] = round(float(tm[arm].mean()), 4)
            out["pools"].setdefault(p, {})[rd] = blk
    # pool audits written by the distiller / builder
    for p in POOLS:
        if p in REFS:
            a = ROOT / ("results/placebo/src1011_gpt_pool_audit.json" if p == "GPTREF" else "results/placebo/src1011_qwen_pool_audit.json")
            c = ROOT / ("results/placebo/w1a_gptref_irrelevant_audit.json" if p == "GPTREF" else "results/placebo/poolrep_gref_irrelevant_audit.json")
        else:
            s, d, r = p.split("_")
            stem = f"w1a_{s.lower()}_{d.lower()}_{r.lower()}"
            a, c = ROOT / f"results/placebo/{stem}_audit.json", ROOT / f"results/placebo/{stem}_irrelevant_audit.json"
        info = out["audits"].setdefault(p, {})
        if a.exists():
            dd = json.loads(a.read_text())
            info.update({"n_items": dd.get("n_items"), "footprint_o200k": dd.get("footprint_o200k", dd.get("footprint_gpt4omini")),
                         "insights_per_type": {t: v.get("n_insights") for t, v in dd.get("types", {}).items()}})
        if c.exists():
            dd = json.loads(c.read_text())
            info["control"] = {"nonce_replacements": dd["nonce_replacements"], "retained_body_fraction": dd["retained_body_fraction"],
                               "context_tokens": dd["context_tokens"], "test_game_vocab_overlap": dd["test_game_vocab_overlap"]}

    # ---------------- pooled estimands over the balanced pools (hierarchical bootstrap) ----------------
    def have(arrs, rds):
        return all((rd, p) in arrs for rd in rds for p in BAL)

    def cell_draws(arrs, rd, s, d, hier=True):
        """(N_BOOT,) mean over the (resampled) replicates of a cell; and its point estimate"""
        per = np.stack([B.task_means(arrs[(rd, f"{s}_{d}_R{r}")]) for r in REPS], axis=1)  # (N, 3)
        point = float(np.mean([arrs[(rd, f"{s}_{d}_R{r}")].mean() for r in REPS]))
        if not hier:
            return per.mean(1), point
        idx = B.reps[s]
        return np.take_along_axis(per, idx, axis=1).mean(1), point

    def estimate(arrs, weights):
        """weights: {(rd, s, d): w}; returns hierarchical and task-only summaries"""
        res = {}
        for hier in (True, False):
            draws, point = 0.0, 0.0
            for (rd, s, d), w in weights.items():
                dr, pt = cell_draws(arrs, rd, s, d, hier)
                draws = draws + w * dr
                point += w * pt
            res["hier" if hier else "task_only"] = summ(point, draws)
        out_ = dict(res["hier"])
        out_["task_only_ci95"] = res["task_only"]["ci95"]
        out_["task_only_p_boot"] = res["task_only"]["p_boot"]
        return out_

    avail = [rd for rd in READERS if have(Rarr, [rd])]
    out["pooled"]["readers_available"] = avail
    for rd in avail:  # E1
        out["pooled"][f"R_bar_{rd}"] = estimate(Rarr, {(rd, s, d): 1 / 6 for s in SOURCES for d in DISTS})
        for s in SOURCES:
            out["pooled"][f"R_bar_{rd}_src_{s}"] = estimate(Rarr, {(rd, s, d): 1 / 2 for d in DISTS})
            for d in DISTS:
                out["pooled"][f"R_bar_{rd}_{s}_{d}"] = estimate(Rarr, {(rd, s, d): 1.0})
        for d in DISTS:
            out["pooled"][f"R_bar_{rd}_dist_{d}"] = estimate(Rarr, {(rd, s, d): 1 / 3 for s in SOURCES})
        out["pooled"][f"D32_minus_D8_{rd}"] = estimate(Rarr, {**{(rd, s, "D32"): 1 / 3 for s in SOURCES},
                                                              **{(rd, s, "D8"): -1 / 3 for s in SOURCES}})
        if have(Iarr, [rd]):  # E6
            out["pooled"][f"I_bar_{rd}"] = estimate(Iarr, {(rd, s, d): 1 / 6 for s in SOURCES for d in DISTS})
        if have(Tarr, [rd]):
            out["pooled"][f"T_bar_{rd}"] = estimate(Tarr, {(rd, s, d): 1 / 6 for s in SOURCES for d in DISTS})
    tests = {}
    if "Q8" in avail and "Q32" in avail:  # E2 and the reader contrast
        w = {}
        for d in DISTS:
            for rd, sign in (("Q8", 1), ("Q32", -1)):
                w[(rd, "Q8", d)] = w.get((rd, "Q8", d), 0) + sign * 0.5
                w[(rd, "Q32", d)] = w.get((rd, "Q32", d), 0) - sign * 0.5
        out["pooled"]["Delta_own"] = tests["Delta_own"] = estimate(Rarr, w)
        out["pooled"]["R_Q8_minus_R_Q32"] = tests["R_Q8_minus_R_Q32"] = estimate(
            Rarr, {**{("Q8", s, d): 1 / 6 for s in SOURCES for d in DISTS}, **{("Q32", s, d): -1 / 6 for s in SOURCES for d in DISTS}})
    for x, y in (("Q8", "Q14"), ("Q14", "Q32")):
        if x in avail and y in avail:
            out["pooled"][f"R_{x}_minus_R_{y}"] = estimate(
                Rarr, {**{(x, s, d): 1 / 6 for s in SOURCES for d in DISTS}, **{(y, s, d): -1 / 6 for s in SOURCES for d in DISTS}})
    if avail:  # E3 and E4 over the readers available (the planned version needs all three)
        n = len(avail)
        for s in SOURCES:
            out["pooled"][f"R_src_{s}"] = estimate(Rarr, {(rd, s, d): 1 / (2 * n) for rd in avail for d in DISTS})
        for a_, b_ in (("Q8", "GPT"), ("Q32", "Q8"), ("Q32", "GPT")):
            e = estimate(Rarr, {**{(rd, a_, d): 1 / (2 * n) for rd in avail for d in DISTS},
                                **{(rd, b_, d): -1 / (2 * n) for rd in avail for d in DISTS}})
            out["pooled"][f"src_{a_}_minus_{b_}"] = e
            if (a_, b_) == ("Q8", "GPT"):
                tests["src_Q8_minus_GPT"] = e
        e = estimate(Rarr, {**{(rd, s, "D32"): 1 / (3 * n) for rd in avail for s in SOURCES},
                            **{(rd, s, "D8"): -1 / (3 * n) for rd in avail for s in SOURCES}})
        out["pooled"]["D32_minus_D8"] = tests["D32_minus_D8"] = e
        out["pooled"]["main_effects_over_readers"] = avail
    # Holm over the four planned contrasts (only when all are available)
    if len(tests) == 4:
        order = sorted(tests, key=lambda k: tests[k]["p_boot"])
        adj, run = {}, 0.0
        for i, k in enumerate(order):
            run = max(run, min(1.0, (4 - i) * tests[k]["p_boot"]))
            adj[k] = round(run, 4)
        out["pooled"]["holm_adjusted_p"] = adj

    # ---------------- E5 heterogeneity ----------------
    # small-sample factor of the stratified bootstrap variance, sum_s (n_s/N) * n_s/(n_s - 1) (= 1.047 for these strata);
    # added before any episode after a null calibration (2026-10-13 execution log): without it the Q test rejects ~10% at 0.05
    ns = np.array([sum(1 for t in TASKS if types[t] == ty) for ty in sorted(set(types.values()))])
    vcorr = float(np.sum(ns / len(TASKS) * ns / (ns - 1)))
    het = {"cells": {}, "Q": None, "bootstrap_variance_factor": round(vcorr, 4)}
    Qsum, df, taus, corr_pairs = 0.0, 0, [], []
    for rd in avail:
        for s in SOURCES:
            for d in DISTS:
                Rs = [Rarr[(rd, f"{s}_{d}_R{r}")] for r in REPS]
                pts = np.array([x.mean() for x in Rs])
                dr = np.stack([B.task_means(x) for x in Rs], axis=1)  # task-only draws (N, 3)
                dvec = np.array([pts[0] - pts[1], pts[0] - pts[2]])
                ddraw = np.stack([dr[:, 0] - dr[:, 1], dr[:, 0] - dr[:, 2]], axis=1)
                S = np.cov(ddraw.T) * vcorr
                Qc = float(dvec @ np.linalg.lstsq(S, dvec, rcond=None)[0])
                v = {(i, j): float(np.var(dr[:, i] - dr[:, j])) * vcorr for i in range(3) for j in range(i + 1, 3)}
                tau2 = float(np.mean([((pts[i] - pts[j]) ** 2 - v[(i, j)]) / 2 for (i, j) in v]))
                het["cells"][f"{s}_{d}_{rd}"] = {"R": [round(float(x), 4) for x in pts], "Q": round(Qc, 3), "tau2": round(tau2, 6)}
                Qsum += Qc
                df += 2
                taus.append(tau2)
        for s in SOURCES:
            for r in REPS:
                corr_pairs.append((Rarr[(rd, f"{s}_D32_R{r}")].mean(), Rarr[(rd, f"{s}_D8_R{r}")].mean()))
    if df:
        tau2 = float(np.mean(taus))
        het.update({"Q": round(Qsum, 3), "df": df, "p": round(chi2_sf_even(Qsum, df), 5), "tau2_pooled": round(tau2, 6),
                    "tau_pooled": round(math.sqrt(max(0.0, tau2)), 4)})
        if len(corr_pairs) >= 3:
            x, y = np.array(corr_pairs).T
            het["corr_R_D32_vs_D8_same_subsample"] = {"r": round(float(np.corrcoef(x, y)[0, 1]), 3), "n": len(corr_pairs)}
    out["heterogeneity"] = het

    # ---------------- E7 reference crossover, noise floor ----------------
    for rd in READERS:
        ref = {}
        if (rd, "GREF") in Rarr and (rd, "GPTREF") in Rarr:
            dd = Rarr[(rd, "GREF")] - Rarr[(rd, "GPTREF")]
            ref["R_GREF_minus_R_GPTREF"] = summ(dd.mean(), B.task_means(dd))
        if (rd, "GREF") in Tarr and (rd, "GPTREF") in Tarr:
            dd = Tarr[(rd, "GREF")] - Tarr[(rd, "GPTREF")]
            ref["T_GREF_minus_T_GPTREF"] = summ(dd.mean(), B.task_means(dd))
        out["reference"][rd] = ref
        a, b = (rd, "Q8_D32_R1_TRUE"), (rd, "Q8_D32_R1_TRUEDUP")
        if a in tm and b in tm:
            dd = tm[a] - tm[b]
            out["noise_floor"][rd] = {"DUP": summ(dd.mean(), B.task_means(dd)),
                                      "share_episodes_differing": round(float(np.mean(
                                          [rows[a][(s, t)]["success"] != rows[b][(s, t)]["success"] for s in SEEDS for t in TASKS])), 4)}

    # ---------------- decisions (rules fixed before the runs) ----------------
    P, dec = out["pooled"], out["decisions"]
    if "Delta_own" in P:
        lo, hi = P["Delta_own"]["ci95"]
        dec["A1"] = ("holds: no own-source advantage (interval includes 0, upper bound < +0.05)" if lo <= 0 <= hi and hi < 0.05 else
                     "fails: own-source advantage in content value" if lo > 0 else
                     "fails: other-source advantage" if hi < 0 else
                     "unresolved at this power (interval includes 0, upper bound >= +0.05)")
    if "R_Q8_minus_R_Q32" in P:
        dec["A2"] = "holds" if P["R_Q8_minus_R_Q32"]["ci95"][0] > 0 else "fails"
    if "src_Q8_minus_GPT" in P:
        lo, hi = P["src_Q8_minus_GPT"]["ci95"]
        dec["A3"] = ("holds: Q8 - GPT interval includes 0 (gap attributed to coverage and pool variation)" if lo <= 0 <= hi else
                     "fails: Q8 source > GPT source at balanced counts" if lo > 0 else "fails: GPT source > Q8 source")
        dec["A3_over_readers"] = P.get("main_effects_over_readers")
    if "D32_minus_D8" in P:
        lo, hi = P["D32_minus_D8"]["ci95"]
        dec["A4"] = "distiller effect (interval excludes 0)" if lo > 0 or hi < 0 else "no detectable distiller effect"
    if het.get("Q") is not None:
        dec["A5"] = ("holds" if het["p"] < 0.05 and het["tau_pooled"] >= 0.02 else
                     "fails: pools homogeneous at this resolution" if het["tau_pooled"] < 0.01 and het["p"] >= 0.05 else "partial")
    a6 = {rd: abs(P[f"I_bar_{rd}"]["mean"]) <= 0.04 for rd in READERS if f"I_bar_{rd}" in P}
    if a6:
        dec["A6"] = {rd: ("holds" if ok else "fails") for rd, ok in a6.items()}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    # console summary
    print("runs:", sum(1 for v in status.values() if isinstance(v, dict)), "complete of", len(status))
    for k in ("R_bar_Q8", "R_bar_Q14", "R_bar_Q32", "I_bar_Q8", "I_bar_Q14", "I_bar_Q32", "Delta_own", "R_Q8_minus_R_Q32",
              "src_Q8_minus_GPT", "src_Q32_minus_Q8", "D32_minus_D8"):
        if k in P:
            print(f"{k:18s} {P[k]['mean']:+.3f} hier {P[k]['ci95']} task-only {P[k]['task_only_ci95']}")
    if het.get("Q") is not None:
        print(f"heterogeneity Q={het['Q']} df={het['df']} p={het['p']} tau={het['tau_pooled']}")
    print("decisions:", json.dumps(dec))
    print("->", OUT)


if __name__ == "__main__":
    main()
