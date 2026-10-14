#!/usr/bin/env python3
"""Exploratory intervals for the five-reader study (chosen after the data were seen; not part of its protocol).

1. Span difference: range over the five readers of the content value R minus range of the reading cost I.
2. Difficulty split for GPT-4o-mini (run 1): content value on observations whose game it failed in both other
   no-memory episodes ("hard") minus on those it solved in both ("easy"), as in D3 of the phase-2 analysis.

Both use a stratified task bootstrap by task type (20,000 draws, seed 20260713); in (1) every reader shares the
same task draws. Reads the same episode logs as analyze_phase2_2026-10-10.py.
Output: results/placebo/exploratory_phase2_2026-10-10.json
Run:    python scripts/analysis/exploratory_phase2_2026-10-10.py"""
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
SEEDS, CONDS = p2.SEEDS, ("none", "true", "irrelevant")
TASKS = list(range(134))
N_BOOT, SEED = 20_000, 20260713
READERS = {"GPT-4o-mini": (["RERUN1010", "P2A"], "GPT"), "Qwen3-8B": (["RERUN1010", "P2A"], "QWEN"),
           "GPT-4.1-mini": (["P2C"], "GPT41MINI"), "Llama-3.1-8B": (["P2C"], "LLAMA8B"), "Qwen3-14B": (["P2C"], "QWEN14B")}
OUT = ROOT / "results/placebo/exploratory_phase2_2026-10-10.json"


def rows_of(prefixes, ex):
    return {c: p2.merged([f"{p}_{c.upper()}_{ex}" for p in prefixes]) for c in CONDS}


def draws_index(types):
    """(N_BOOT, 134) resampled task indices, stratified by type."""
    by = defaultdict(list)
    for t in TASKS:
        by[types[t]].append(t)
    rng = np.random.default_rng(SEED)
    groups = [np.asarray(sorted(v)) for _, v in sorted(by.items())]
    return np.stack([np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]) for _ in range(N_BOOT)])


def summ(point, draws):
    return {"mean": round(float(point), 4), "ci95": [round(float(np.quantile(draws, .025)), 4),
                                                     round(float(np.quantile(draws, .975)), 4)],
            "share_draws_le_0": round(float(np.mean(draws <= 0)), 4)}


def main():
    rows = {n: rows_of(*v) for n, v in READERS.items()}
    types = {t: r["task_type"] for (_, t), r in rows["GPT-4o-mini"]["none"].items()}
    idx = draws_index(types)
    per = {n: {c: np.array([np.mean([rows[n][c][(s, t)]["success"] for s in SEEDS]) for t in TASKS]) for c in CONDS}
           for n in READERS}
    I = np.stack([per[n]["irrelevant"] - per[n]["none"] for n in READERS])   # (5, 134)
    R = np.stack([per[n]["true"] - per[n]["irrelevant"] for n in READERS])
    Ib, Rb = I[:, idx].mean(axis=2), R[:, idx].mean(axis=2)                   # (5, N_BOOT)
    span = lambda m, ax: m.max(axis=ax) - m.min(axis=ax)                      # noqa: E731
    out = {"span_R_minus_span_I": summ(span(R.mean(1), 0) - span(I.mean(1), 0), span(Rb, 0) - span(Ib, 0))}
    g = rows["GPT-4o-mini"]
    obs = defaultdict(list)  # task -> list of (class, R_obs)
    for t in TASKS:
        for s in SEEDS:
            other = [g["none"][(s2, t)]["success"] for s2 in SEEDS if s2 != s]
            k = "hard" if sum(other) == 0 else ("easy" if sum(other) == len(other) else "mixed")
            obs[t].append((k, float(g["true"][(s, t)]["success"]) - float(g["irrelevant"][(s, t)]["success"])))

    def hard_minus_easy(task_list):
        h = [v for t in task_list for k, v in obs[t] if k == "hard"]
        e = [v for t in task_list for k, v in obs[t] if k == "easy"]
        return np.mean(h) - np.mean(e)

    out["gpt4omini_hard_minus_easy"] = summ(hard_minus_easy(TASKS), np.array([hard_minus_easy(d) for d in idx]))
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
