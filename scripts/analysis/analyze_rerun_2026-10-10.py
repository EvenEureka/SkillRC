#!/usr/bin/env python3
"""Analysis of the 2026-10-10 rerun of the placebo gate with the coherent task-irrelevant control
(PREREG_rerun_2026-10-10.md). Same estimator as analyze_semantic_placebo.py: equal-pair task means over
seeds [0, 1, 5], stratified task bootstrap (10 tasks per type), 20,000 draws, seed 20260713.
Output: results/placebo/rerun_2026-10-10_results.json and a Markdown table on stdout."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SEEDS = (0, 1, 5)
CONDS = ("none", "true", "placebo", "irrelevant")
N_BOOT, BOOT_SEED = 20_000, 20260713
OLD = {"gpt": {"none": 0.544, "true": 0.681, "placebo": 0.633}, "qwen": {"none": 0.728, "true": 0.683, "placebo": 0.620}}


def load(run_id):
    rows = {}
    path = RESULTS / f"{run_id}.episodes.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r["seed"]) not in SEEDS:
            continue
        if r.get("error"):
            raise ValueError(f"episode error in {run_id}: {r['error']}")
        rows[(int(r["seed"]), int(r["task_id"]))] = r
    return rows


def boot(values, task_types):
    by_type = defaultdict(list)
    for t in values:
        by_type[task_types[t]].append(t)
    rng = np.random.default_rng(BOOT_SEED)
    groups = [np.asarray([values[t] for t in sorted(ids)]) for _, ids in sorted(by_type.items())]
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        draws[i] = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]).mean()
    return float(np.mean(list(values.values()))), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main():
    # the frozen 60-task sample is pinned in the configs; task types come from the episode records
    # (the 2026-07 frozen_tasks.json was not part of the public release)
    import yaml
    cfg = yaml.safe_load((ROOT / "configs/rerun_2026_10_10/none_gpt.yaml").read_text())
    task_ids = [int(t) for t in cfg["task_ids"]]
    task_types = {}
    for ex in ("gpt", "qwen"):
        for r in load(f"RERUN1010_NONE_{ex.upper()}").values():
            task_types.setdefault(int(r["task_id"]), r["task_type"])
    assert set(task_ids) <= set(task_types), "task types missing for some frozen tasks"
    by_type = defaultdict(list)
    for t in task_ids:
        by_type[task_types[t]].append(t)
    # Addendum A: the game behind each id differs from 2026-07; strata are the observed types (unequal sizes allowed)
    assert len(task_ids) == 60 and len(by_type) == 6, {k: len(v) for k, v in by_type.items()}
    (RESULTS / "placebo" / "frozen_tasks_rerun_2026-10-10.json").write_text(json.dumps({"task_ids": task_ids, "by_type": dict(by_type)}, indent=2) + "\n")
    out = {"estimands": {}, "means": {}, "drift_vs_2026_07": {}, "footprint": {}}
    means = {}
    for ex in ("gpt", "qwen"):
        for cond in CONDS:
            rows = load(f"RERUN1010_{cond.upper()}_{ex.upper()}")
            missing = {(s, t) for s in SEEDS for t in task_ids} - set(rows)
            if missing:
                raise ValueError(f"{ex}/{cond}: {len(missing)} missing episodes")
            if cond != "none":
                toks = {int(rows[(s, t)]["memory_tokens"]) for s in SEEDS for t in task_ids}
                out["footprint"][f"{ex}/{cond}"] = sorted(toks)
            means[(ex, cond)] = {t: float(np.mean([rows[(s, t)]["success"] for s in SEEDS])) for t in task_ids}
            out["means"][f"{ex}/{cond}"] = round(float(np.mean(list(means[(ex, cond)].values()))), 4)
            if cond in OLD[ex]:
                out["drift_vs_2026_07"][f"{ex}/{cond}"] = round(out["means"][f"{ex}/{cond}"] - OLD[ex][cond], 4)
    contrasts = {"T": ("true", "none"), "P": ("placebo", "none"), "I": ("irrelevant", "none"),
                 "S": ("true", "placebo"), "R": ("true", "irrelevant"), "C": ("irrelevant", "placebo")}
    per_ex = {}
    for name, (a, b) in contrasts.items():
        for ex in ("gpt", "qwen"):
            vals = {t: means[(ex, a)][t] - means[(ex, b)][t] for t in task_ids}
            per_ex[(name, ex)] = vals
            m, lo, hi = boot(vals, task_types)
            out["estimands"][f"{name}_{ex}"] = {"contrast": f"{a} - {b}", "mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        gap = {t: per_ex[(name, "qwen")][t] - per_ex[(name, "gpt")][t] for t in task_ids}
        m, lo, hi = boot(gap, task_types)
        out["estimands"][f"Delta_{name}"] = {"contrast": f"({name}_qwen) - ({name}_gpt)", "mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    # frozen decision rules
    e = out["estimands"]
    rule = "4 (inconclusive)"
    if all(abs(e[f"C_{ex}"]["mean"]) <= 0.05 for ex in ("gpt", "qwen")) and e["R_gpt"]["ci95"][0] > 0:
        rule = "1 (content increment = task relevance)"
    elif all(abs(e[f"R_{ex}"]["mean"]) <= 0.05 for ex in ("gpt", "qwen")):
        rule = "2 (any coherent instruction block)"
    elif (e["C_gpt"]["ci95"][0] > 0 or e["C_gpt"]["ci95"][1] < 0 or e["C_qwen"]["ci95"][0] > 0 or e["C_qwen"]["ci95"][1] < 0) and e["R_gpt"]["ci95"][0] > 0:
        rule = "3 (coherence and relevance both contribute)"
    out["decision_rule"] = rule
    (RESULTS / "placebo" / "rerun_2026-10-10_results.json").write_text(json.dumps(out, indent=2) + "\n")
    print("| executor | none | placebo | irrelevant | true |\n|---|---|---|---|---|")
    for ex in ("gpt", "qwen"):
        print(f"| {ex} | " + " | ".join(f"{out['means'][f'{ex}/{c}']:.3f}" for c in ("none", "placebo", "irrelevant", "true")) + " |")
    print("\n| estimand | gpt | qwen | gap (qwen - gpt) |\n|---|---|---|---|")
    for name in contrasts:
        f = lambda k: f"{e[k]['mean']:+.3f} [{e[k]['ci95'][0]:+.3f}, {e[k]['ci95'][1]:+.3f}]"  # noqa: E731
        print(f"| {name} ({contrasts[name][0]} - {contrasts[name][1]}) | {f(f'{name}_gpt')} | {f(f'{name}_qwen')} | {f(f'Delta_{name}')} |")
    print("\ndecision rule:", rule)
    print("drift vs 2026-07 gate:", out["drift_vs_2026_07"])


if __name__ == "__main__":
    main()
