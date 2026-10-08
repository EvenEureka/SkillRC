#!/usr/bin/env python3
"""Export cross-fitted baseline-competence analysis for Paper O.

For each held-out seed, task difficulty is measured using only the other two
no-memory seeds. Confidence intervals use a task-cluster bootstrap so repeated
seed observations from the same task are resampled together.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT.parent / "ARR" / "PaperO_assets" / "table4_crossfit_headroom.csv"

SPECS = [
    ("gpt-4o-mini", "R008_nomem_s3", "R012_expel_bm25", "expel", "bm25"),
    ("gpt-4o-mini", "R008_nomem_s3", "R012_expel_dense", "expel", "dense"),
    ("gpt-4o-mini", "R008_nomem_s3", "R012_skillos_bm25", "skillos", "bm25"),
    ("gpt-4o-mini", "R008_nomem_s3", "R012_skillos_dense", "skillos", "dense"),
    ("qwen3-8b", "Q_nomem_s3", "Q_expel_bm25_s3", "expel", "bm25"),
    ("qwen3-8b", "Q_nomem_s3", "Q_expel_s3", "expel", "dense"),
    ("qwen3-8b", "Q_nomem_s3", "Q_skillos_bm25_s3", "skillos", "bm25"),
    ("qwen3-8b", "Q_nomem_s3", "Q_skillos_s3", "skillos", "dense"),
]


def load(run_id: str) -> dict[tuple[int, object], dict]:
    path = RESULTS / f"{run_id}.episodes.jsonl"
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {(int(row["seed"]), row["task_id"]): row for row in rows}


def cluster_bootstrap_ci(rows: list[dict], rng: np.random.Generator) -> tuple[float, float]:
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row["delta_success"])
    task_ids = np.array(sorted(by_task), dtype=object)
    draws = np.empty(10_000, dtype=float)
    for idx in range(len(draws)):
        sampled = rng.choice(task_ids, size=len(task_ids), replace=True)
        values = [value for task_id in sampled for value in by_task[task_id]]
        draws[idx] = np.mean(values)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def main():
    rng = np.random.default_rng(20260712)
    exported = []
    for executor, baseline_id, variant_id, representation, retriever in SPECS:
        baseline = load(baseline_id)
        variant = load(variant_id)
        seeds = sorted({seed for seed, _ in baseline})
        bins = defaultdict(list)
        for (seed, task_id), base_row in baseline.items():
            other_seed_successes = sum(
                int(baseline[(other_seed, task_id)]["success"])
                for other_seed in seeds
                if other_seed != seed
            )
            candidate = variant[(seed, task_id)]
            bins[other_seed_successes].append(
                {
                    "task_id": task_id,
                    "delta_success": int(candidate["success"]) - int(base_row["success"]),
                    "baseline_success": int(base_row["success"]),
                    "delta_prompt_tokens": float(candidate["prompt_tokens"]) - float(base_row["prompt_tokens"]),
                }
            )
        for bin_value in [0, 1, 2]:
            rows = bins[bin_value]
            mean_delta = float(np.mean([row["delta_success"] for row in rows]))
            lo, hi = cluster_bootstrap_ci(rows, rng)
            exported.append(
                {
                    "executor": executor,
                    "baseline_run_id": baseline_id,
                    "variant_run_id": variant_id,
                    "representation": representation,
                    "retriever": retriever,
                    "other_seed_successes": bin_value,
                    "difficulty_label": ["hard (0/2)", "mixed (1/2)", "easy (2/2)"][bin_value],
                    "paired_n": len(rows),
                    "unique_tasks": len({row["task_id"] for row in rows}),
                    "heldout_baseline_success_rate": f"{np.mean([row['baseline_success'] for row in rows]):.4f}",
                    "delta_success_mean": f"{mean_delta:+.4f}",
                    "delta_success_cluster_boot_ci95_lo": f"{lo:+.4f}",
                    "delta_success_cluster_boot_ci95_hi": f"{hi:+.4f}",
                    "delta_prompt_tokens_mean": f"{np.mean([row['delta_prompt_tokens'] for row in rows]):+.1f}",
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    print(OUT)


if __name__ == "__main__":
    main()
