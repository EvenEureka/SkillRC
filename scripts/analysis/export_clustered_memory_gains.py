#!/usr/bin/env python3
"""Export task-cluster bootstrap intervals for all Paper O memory gains."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT.parent / "ARR" / "PaperO_assets" / "table8_clustered_memory_gains.csv"

RUNS = {
    "gpt-4o-mini": {
        "baseline": "R008_nomem_s3",
        "expel_bm25": "R012_expel_bm25",
        "expel_dense": "R012_expel_dense",
        "raw_bm25": "R012_raw_bm25",
        "raw_dense": "R012_raw_dense",
        "skillos_bm25": "R012_skillos_bm25",
        "skillos_dense": "R012_skillos_dense",
    },
    "qwen3-8b": {
        "baseline": "Q_nomem_s3",
        "expel_bm25": "Q_expel_bm25_s3",
        "expel_dense": "Q_expel_s3",
        "skillos_bm25": "Q_skillos_bm25_s3",
        "skillos_dense": "Q_skillos_s3",
    },
}


def load(run_id: str) -> dict[tuple[int, str], int]:
    with (RESULTS / f"{run_id}.episodes.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {(int(row["seed"]), str(row["task_id"])): int(row["success"]) for row in rows}


def main() -> None:
    rng = np.random.default_rng(20260712)
    exported = []
    for executor, ids in RUNS.items():
        baseline = load(ids["baseline"])
        task_ids = sorted({task_id for _, task_id in baseline})
        for condition, run_id in ids.items():
            if condition == "baseline":
                continue
            variant = load(run_id)
            task_values = {
                task_id: np.mean(
                    [variant[(seed, task_id)] - baseline[(seed, task_id)] for seed in [0, 1, 2]]
                )
                for task_id in task_ids
            }
            tasks = np.array(task_ids, dtype=object)
            draws = np.empty(10_000, dtype=float)
            for index in range(len(draws)):
                sampled = rng.choice(tasks, size=len(tasks), replace=True)
                draws[index] = np.mean([task_values[task_id] for task_id in sampled])
            lo, hi = np.quantile(draws, [0.025, 0.975])
            exported.append(
                {
                    "executor": executor,
                    "condition": condition,
                    "run_id": run_id,
                    "estimate": f"{np.mean(list(task_values.values())):+.4f}",
                    "task_cluster_boot_ci95_lo": f"{lo:+.4f}",
                    "task_cluster_boot_ci95_hi": f"{hi:+.4f}",
                    "unique_tasks": len(task_ids),
                    "nominal_seed_observations": len(baseline),
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
