#!/usr/bin/env python3
"""WebShop study analysis.

Per executor, on the WebShop score (primary) and success (secondary), over test sessions 0-499:
T = true-none, P = placebo-none, I = irrelevant-none, S = true-placebo, R = true-irrelevant,
C = irrelevant-placebo; session bootstrap stratified by product category, 20,000 draws, seed 20260713.
Errors count as score 0. Gaps use gpt-4o-mini as reference. Noise floor from WS1010R2_*_GPT.
Missing runs are reported as missing. Output: results/placebo/webshop_2026-10-10_results.json."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
TEST = list(range(500))
CONDS = ("none", "true", "placebo", "irrelevant")
EXECUTORS = ("GPT", "GPT41MINI", "QWEN")
N_BOOT, BOOT_SEED = 20_000, 20260713
CONTRASTS = {"T": ("true", "none"), "P": ("placebo", "none"), "I": ("irrelevant", "none"),
             "S": ("true", "placebo"), "R": ("true", "irrelevant"), "C": ("irrelevant", "placebo")}


def load(run):
    p = RES / f"{run}.episodes.jsonl"
    if not p.exists():
        return None
    rows = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[int(r["task_id"])] = r
    return rows


def boot(values, strata):
    by = defaultdict(list)
    for t in values:
        by[strata[t]].append(values[t])
    rng = np.random.default_rng(BOOT_SEED)
    groups = [np.asarray(v) for _, v in sorted(by.items())]
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        draws[i] = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]).mean()
    return {"mean": round(float(np.mean(list(values.values()))), 4),
            "ci95": [round(float(np.quantile(draws, .025)), 4), round(float(np.quantile(draws, .975)), 4)]}


def outcome(r, kind):
    if r.get("error"):
        return 0.0
    return float(r["reward"]) if kind == "score" else float(bool(r["success"]))


def block(prefix, ex, strata):
    per, info = {}, {}
    for c in CONDS:
        rows = load(f"{prefix}_{c.upper()}_{ex}")
        if rows is None:
            return {"missing": f"{prefix}_{c.upper()}_{ex}"}
        miss = [t for t in TEST if t not in rows]
        if miss:
            return {"incomplete": f"{c}: {len(miss)} sessions missing"}
        per[c] = {k: {t: outcome(rows[t], k) for t in TEST} for k in ("score", "success")}
        info[c] = {"errors": sum(bool(rows[t].get("error")) for t in TEST),
                   "steps": round(float(np.mean([rows[t]["steps"] for t in TEST])), 2),
                   "purchase_rate": round(float(np.mean([any(s["action"].lower().startswith("click[buy now")
                                                             for s in rows[t].get("trajectory", [])) for t in TEST])), 4)}
        if c != "none":
            info[c]["footprint"] = sorted({int(rows[t]["memory_tokens"]) for t in TEST})
    out = {"means": {k: {c: round(float(np.mean(list(per[c][k].values()))), 4) for c in CONDS} for k in ("score", "success")},
           "info": info, "_per": per}
    out["estimands"] = {k: {e: boot({t: per[a][k][t] - per[b][k][t] for t in TEST}, strata) for e, (a, b) in CONTRASTS.items()}
                        for k in ("score", "success")}
    return out


def main():
    strata = {}
    instr = json.loads((ROOT / "data/summaries/webshop_2026-10-10/instructions_seed0_1k.json").read_text())
    for t in TEST:
        strata[t] = instr["instructions"][str(t)]["category"]
    blocks = {ex: block("WS1010", ex, strata) for ex in EXECUTORS}
    blocks["GPT_run2"] = block("WS1010R2", "GPT", strata)
    ok = {k: v for k, v in blocks.items() if "_per" in v}
    gaps = {}
    for ex in ("GPT41MINI", "QWEN"):
        if ex in ok and "GPT" in ok:
            A, B = ok["GPT"]["_per"], ok[ex]["_per"]
            gaps[f"{ex}-GPT"] = {e: boot({t: (B[a]["score"][t] - B[b]["score"][t]) - (A[a]["score"][t] - A[b]["score"][t]) for t in TEST}, strata)
                                 for e, (a, b) in CONTRASTS.items()}
    noise = {}
    if "GPT" in ok and "GPT_run2" in ok:
        for c in CONDS:
            a, b = ok["GPT"]["_per"][c]["score"], ok["GPT_run2"]["_per"][c]["score"]
            dis = sum(abs(a[t] - b[t]) > 1e-9 for t in TEST)
            noise[c] = {"sessions": len(TEST), "differ": dis, "rate": round(dis / len(TEST), 4)}
        avg = {c: {t: (ok["GPT"]["_per"][c]["score"][t] + ok["GPT_run2"]["_per"][c]["score"][t]) / 2 for t in TEST} for c in CONDS}
        noise["estimands_on_run_average"] = {e: boot({t: avg[a][t] - avg[b][t] for t in TEST}, strata) for e, (a, b) in CONTRASTS.items()}
    preds = {}
    est = {ex: ok[ex]["estimands"]["score"] for ex in EXECUTORS if ex in ok}
    if "GPT" in est:
        preds["W1 R_GPT > 0"] = est["GPT"]["R"]["ci95"][0] > 0
    if len(est) == 3:
        rng_I = max(e["I"]["mean"] for e in est.values()) - min(e["I"]["mean"] for e in est.values())
        rng_R = max(e["R"]["mean"] for e in est.values()) - min(e["R"]["mean"] for e in est.values())
        preds["W2 range(I) > range(R)"] = {"holds": rng_I > rng_R, "range_I": round(rng_I, 4), "range_R": round(rng_R, 4)}
        preds["W3 C_E <= 0 (point)"] = {ex: e["C"]["mean"] <= 0 for ex, e in est.items()}
        preds["W4 T_GPT41MINI < T_GPT"] = est["GPT41MINI"]["T"]["mean"] < est["GPT"]["T"]["mean"]
    out = {"blocks": {k: {kk: vv for kk, vv in v.items() if kk != "_per"} for k, v in blocks.items()},
           "gaps_score": gaps, "noise_floor_gpt4omini": noise, "predictions": preds,
           "strata_counts": {c: sum(1 for t in TEST if strata[t] == c) for c in sorted(set(strata.values()))}}
    (RES / "placebo/webshop_2026-10-10_results.json").write_text(json.dumps(out, indent=2) + "\n")
    for name, b in out["blocks"].items():
        if "means" not in b:
            print(name, b)
            continue
        m, e = b["means"]["score"], b["estimands"]["score"]
        f = lambda k: f"{e[k]['mean']:+.3f} [{e[k]['ci95'][0]:+.3f}, {e[k]['ci95'][1]:+.3f}]"  # noqa: E731
        print(f"{name}: score none {m['none']:.3f} scr {m['placebo']:.3f} irr {m['irrelevant']:.3f} true {m['true']:.3f} | "
              + " ".join(f"{k} {f(k)}" for k in CONTRASTS) + f" | SR {b['means']['success']}")
    for g, v in gaps.items():
        print("gap", g, " ".join(f"{k} {v[k]['mean']:+.3f} [{v[k]['ci95'][0]:+.3f}, {v[k]['ci95'][1]:+.3f}]" for k in ("T", "I", "R", "S")))
    print("noise", json.dumps(noise)[:400])
    print("predictions", json.dumps(preds))


if __name__ == "__main__":
    main()
