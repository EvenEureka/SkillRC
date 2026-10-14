#!/usr/bin/env python3
"""Phase-2 (five-reader study) analysis. Same estimator as the rerun analysis: equal-pair task
means over seeds [0, 1, 5]; stratified task bootstrap by observed task type, 20,000 draws, seed 20260713.

  A  74 out-of-sample games, gpt-4o-mini and Qwen3-8B            (P2A_*)
  134 pooled: 60-game rerun (RERUN1010_*) + A, same day             (run 1)
  B  second gpt-4o-mini run of all 134 games                       (P2B_*): noise floor, run-2 estimands
  C  gpt-4.1-mini, Llama-3.1-8B, Qwen3-14B on all 134 games        (P2C_*)
Missing runs are skipped (reported as missing), so the script can be run while part C is still running.
Output: results/placebo/phase2_2026-10-10_results.json and Markdown tables on stdout."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
SEEDS = (0, 1, 5)
CONDS = ("none", "true", "placebo", "irrelevant")
N_BOOT, BOOT_SEED = 20_000, 20260713
CONTRASTS = {"T": ("true", "none"), "P": ("placebo", "none"), "I": ("irrelevant", "none"),
             "S": ("true", "placebo"), "R": ("true", "irrelevant"), "C": ("irrelevant", "placebo")}


def load(run_id):
    path = RES / f"{run_id}.episodes.jsonl"
    if not path.exists():
        return None
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if int(r["seed"]) in SEEDS:
                rows[(int(r["seed"]), int(r["task_id"]))] = r
    return rows


def merged(run_ids):
    out = {}
    for rid in run_ids:
        rows = load(rid)
        if rows is None:  # seed-split runs (Llama-3.1-8B was split into one job per demonstration pair)
            parts = [load(f"{rid}_S{sd}") for sd in SEEDS]
            if any(x is None for x in parts):
                return None
            rows = {k: v for x in parts for k, v in x.items()}
        out.update(rows)
    return out


def task_means(rows, tasks):
    missing = [(s, t) for s in SEEDS for t in tasks if (s, t) not in rows]
    if missing:
        return None, len(missing)
    errors = sum(1 for s in SEEDS for t in tasks if rows[(s, t)].get("error"))
    return {t: float(np.mean([rows[(s, t)]["success"] for s in SEEDS])) for t in tasks}, errors


def boot(values, types):
    by = defaultdict(list)
    for t in values:
        by[types[t]].append(t)
    rng = np.random.default_rng(BOOT_SEED)
    groups = [np.asarray([values[t] for t in sorted(v)]) for _, v in sorted(by.items())]
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        draws[i] = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]).mean()
    m = float(np.mean(list(values.values())))
    return {"mean": round(m, 4), "ci95": [round(float(np.quantile(draws, .025)), 4), round(float(np.quantile(draws, .975)), 4)]}


def executor_block(sources, tasks, types):
    """sources: cond -> list of run ids. Returns means, estimands, per-task means (for gaps)."""
    means, per, info = {}, {}, {}
    for c in CONDS:
        rows = merged(sources[c])
        if rows is None:
            return None
        tm, err = task_means(rows, tasks)
        if tm is None:
            return {"incomplete": f"{c}: {err} missing episodes"}
        per[c] = tm
        info[c] = {"errors": err}
        means[c] = round(float(np.mean(list(tm.values()))), 4)
        if c != "none":
            info[c]["footprint"] = sorted({int(rows[(s, t)]["memory_tokens"]) for s in SEEDS for t in tasks})
    est = {}
    for k, (a, b) in CONTRASTS.items():
        est[k] = boot({t: per[a][t] - per[b][t] for t in tasks}, types)
    rule = "4 (inconclusive)"
    return {"means": means, "estimands": est, "info": info, "_per": per}


def gaps(blk_a, blk_b, tasks, types):
    out = {}
    for k, (a, b) in CONTRASTS.items():
        out[k] = boot({t: (blk_b["_per"][a][t] - blk_b["_per"][b][t]) - (blk_a["_per"][a][t] - blk_a["_per"][b][t]) for t in tasks}, types)
    return out


def main():
    types = {}
    for rid in ("RERUN1010_NONE_GPT", "P2A_NONE_GPT", "P2B_NONE_GPT"):
        rows = load(rid) or {}
        for (_, t), r in rows.items():
            types.setdefault(t, r["task_type"])
    gate = json.loads((RES / "placebo/frozen_tasks_rerun_2026-10-10.json").read_text())["task_ids"]
    comp = [t for t in range(134) if t not in set(gate)]
    all134 = list(range(134))
    out = {"missing_types": [t for t in all134 if t not in types]}
    tasks_ok = lambda ts: all(t in types for t in ts)  # noqa: E731
    src = lambda prefix, ex: {c: [f"{p}_{c.upper()}_{ex}" for p in (prefix if isinstance(prefix, list) else [prefix])] for c in CONDS}  # noqa: E731
    blocks = {}
    if tasks_ok(comp):
        for ex in ("GPT", "QWEN"):
            blocks[f"A74/{ex}"] = executor_block(src("P2A", ex), comp, types)
    if tasks_ok(all134):
        for ex in ("GPT", "QWEN"):
            blocks[f"run1_134/{ex}"] = executor_block(src(["RERUN1010", "P2A"], ex), all134, types)
        blocks["run2_134/GPT"] = executor_block(src("P2B", "GPT"), all134, types)
        for ex in ("GPT41MINI", "LLAMA8B", "QWEN14B"):
            blocks[f"C134/{ex}"] = executor_block(src("P2C", ex), all134, types)
    gap_out = {}
    for name, (a, b, ts) in {"A74 Qwen-GPT": ("A74/GPT", "A74/QWEN", comp), "134 Qwen-GPT": ("run1_134/GPT", "run1_134/QWEN", all134),
                             "134 GPT4.1mini-GPT": ("run1_134/GPT", "C134/GPT41MINI", all134), "134 Llama8B-GPT": ("run1_134/GPT", "C134/LLAMA8B", all134),
                             "134 Qwen14B-GPT": ("run1_134/GPT", "C134/QWEN14B", all134)}.items():
        A, B = blocks.get(a), blocks.get(b)
        if A and B and "_per" in A and "_per" in B:
            gap_out[name] = gaps(A, B, ts, types)
    # noise floor: run 1 vs run 2 for gpt-4o-mini, episode-level disagreement per condition
    noise = {}
    for c in CONDS:
        r1 = merged([f"RERUN1010_{c.upper()}_GPT", f"P2A_{c.upper()}_GPT"])
        r2 = load(f"P2B_{c.upper()}_GPT")
        if r1 and r2:
            keys = [k for k in r2 if k in r1]
            dis = sum(bool(r1[k]["success"]) != bool(r2[k]["success"]) for k in keys)
            noise[c] = {"episodes": len(keys), "disagree": dis, "rate": round(dis / max(1, len(keys)), 4)}
    if "run1_134/GPT" in blocks and "run2_134/GPT" in blocks and blocks["run2_134/GPT"] and "_per" in (blocks["run2_134/GPT"] or {}):
        avg = {c: {t: (blocks["run1_134/GPT"]["_per"][c][t] + blocks["run2_134/GPT"]["_per"][c][t]) / 2 for t in all134} for c in CONDS}
        noise["estimands_on_run_average"] = {k: boot({t: avg[a][t] - avg[b][t] for t in all134}, types) for k, (a, b) in CONTRASTS.items()}
    for name, blk in blocks.items():
        if blk and "estimands" in blk:
            e = blk["estimands"]
            if abs(e["C"]["mean"]) <= 0.05 and e["S"]["ci95"][0] > 0:
                blk["decision"] = "content increment present; controls agree (rule 1 if R also > 0)" if e["R"]["ci95"][0] > 0 else "content increment over placebo; R interval includes 0"
            elif e["S"]["ci95"][0] <= 0:
                blk["decision"] = "no content increment over placebo at 95%"
            else:
                blk["decision"] = "content increment; controls differ"
    # Addendum B: ladder decomposition, leave-one-executor-out prediction, content value by difficulty
    five = {"GPT-4o-mini": "run1_134/GPT", "Qwen3-8B": "run1_134/QWEN", "GPT-4.1-mini": "C134/GPT41MINI",
            "Llama-3.1-8B": "C134/LLAMA8B", "Qwen3-14B": "C134/QWEN14B"}
    have = {n: blocks[b] for n, b in five.items() if blocks.get(b) and "estimands" in blocks[b]}
    decomp = {"executors": {}}
    for n, b in have.items():
        e = b["estimands"]
        decomp["executors"][n] = {"none": b["means"]["none"], "T": e["T"]["mean"], "cost_I": e["I"]["mean"],
                                  "vocab_cue_P_minus_I": round(-e["C"]["mean"], 4), "procedure_S": e["S"]["mean"],
                                  "content_R": e["R"]["mean"], "I_ci": e["I"]["ci95"], "R_ci": e["R"]["ci95"]}
    if len(have) == 5:
        ex = decomp["executors"]
        rI = max(v["cost_I"] for v in ex.values()) - min(v["cost_I"] for v in ex.values())
        rR = max(v["content_R"] for v in ex.values()) - min(v["content_R"] for v in ex.values())
        decomp["D1_range_I_gt_range_R"] = {"holds": rI > rR, "range_I": round(rI, 4), "range_R": round(rR, 4)}
        loo = {}
        for n, v in ex.items():
            others = [w for m, w in ex.items() if m != n]
            dec = v["cost_I"] + float(np.mean([w["content_R"] for w in others]))
            naive = float(np.mean([w["T"] for w in others]))
            loo[n] = {"observed_T": v["T"], "decomposition_pred": round(dec, 4), "naive_pred": round(naive, 4)}
        mae_dec = float(np.mean([abs(x["observed_T"] - x["decomposition_pred"]) for x in loo.values()]))
        mae_nai = float(np.mean([abs(x["observed_T"] - x["naive_pred"]) for x in loo.values()]))
        decomp["D2_loo"] = {"per_executor": loo, "mae_decomposition": round(mae_dec, 4), "mae_naive": round(mae_nai, 4),
                            "holds": mae_dec < mae_nai}
    d3 = {}
    for n, bname in five.items():
        srcs = src(["RERUN1010", "P2A"], bname.split("/")[1]) if bname.startswith("run1") else src("P2C", bname.split("/")[1])
        rows = {c: merged(srcs[c]) for c in CONDS}
        if any(r is None for r in rows.values()) or any((s, t) not in rows[c] for c in CONDS for s in SEEDS for t in all134):
            continue
        cls = defaultdict(list)
        for t in all134:
            for s in SEEDS:
                other = [rows["none"][(s2, t)]["success"] for s2 in SEEDS if s2 != s]
                k = "hard" if sum(other) == 0 else ("easy" if sum(other) == len(other) else "mixed")
                cls[k].append(float(rows["true"][(s, t)]["success"]) - float(rows["irrelevant"][(s, t)]["success"]))
        d3[n] = {k: {"R": round(float(np.mean(v)), 4), "n_obs": len(v)} for k, v in sorted(cls.items())}
    decomp["D3_content_by_crossfit_difficulty"] = d3
    out["addendum_B"] = decomp
    out["blocks"] = {k: ({kk: vv for kk, vv in v.items() if kk != "_per"} if v else None) for k, v in blocks.items()}
    out["gaps"] = gap_out
    out["noise_floor_gpt4omini"] = noise
    (RES / "placebo/phase2_2026-10-10_results.json").write_text(json.dumps(out, indent=2) + "\n")
    for name, blk in out["blocks"].items():
        if not blk:
            print(f"{name}: missing")
            continue
        if "incomplete" in blk:
            print(f"{name}: {blk['incomplete']}")
            continue
        m, e = blk["means"], blk["estimands"]
        f = lambda k: f"{e[k]['mean']:+.3f} [{e[k]['ci95'][0]:+.3f}, {e[k]['ci95'][1]:+.3f}]"  # noqa: E731
        print(f"{name}: none {m['none']:.3f} scr {m['placebo']:.3f} irr {m['irrelevant']:.3f} true {m['true']:.3f} | "
              + " ".join(f"{k} {f(k)}" for k in CONTRASTS) + f" | {blk.get('decision')}")
    for name, g in gap_out.items():
        print(f"gap {name}: " + " ".join(f"{k} {g[k]['mean']:+.3f} [{g[k]['ci95'][0]:+.3f}, {g[k]['ci95'][1]:+.3f}]" for k in ("T", "P", "S")))
    print("noise floor:", json.dumps(noise)[:600])
    print("addendum B:", json.dumps({k: v for k, v in out["addendum_B"].items()}, indent=1)[:3000])


if __name__ == "__main__":
    main()
