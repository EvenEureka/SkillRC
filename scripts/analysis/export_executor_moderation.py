#!/usr/bin/env python3
"""Export executor-by-memory moderation effects with task-cluster bootstrap CIs."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT.parent / "ARR" / "PaperO_assets" / "table5_executor_moderation.csv"

CONDITIONS = [
    ("expel_bm25", "R012_expel_bm25", "Q_expel_bm25_s3"),
    ("expel_dense", "R012_expel_dense", "Q_expel_s3"),
    ("skillos_bm25", "R012_skillos_bm25", "Q_skillos_bm25_s3"),
    ("skillos_dense", "R012_skillos_dense", "Q_skillos_s3"),
]


def load(run_id: str) -> dict[tuple[int, object], int]:
    path = RESULTS / f"{run_id}.episodes.jsonl"
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {(int(row["seed"]), row["task_id"]): int(row["success"]) for row in rows}


def cluster_ci(values: dict[tuple[int, object], float], rng: np.random.Generator) -> tuple[float, float]:
    by_task = defaultdict(list)
    for (_, task_id), value in values.items():
        by_task[task_id].append(value)
    task_ids = np.array(sorted(by_task), dtype=object)
    draws = np.empty(10_000, dtype=float)
    for idx in range(len(draws)):
        sampled = rng.choice(task_ids, size=len(task_ids), replace=True)
        draws[idx] = np.mean([value for task_id in sampled for value in by_task[task_id]])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def make_row(effect_type: str, label: str, values: dict, rng: np.random.Generator) -> dict:
    lo, hi = cluster_ci(values, rng)
    return {
        "effect_type": effect_type,
        "label": label,
        "estimate": f"{np.mean(list(values.values())):+.4f}",
        "task_cluster_boot_ci95_lo": f"{lo:+.4f}",
        "task_cluster_boot_ci95_hi": f"{hi:+.4f}",
        "paired_n": len(values),
        "unique_tasks": len({task_id for _, task_id in values}),
        "seed0": f"{np.mean([value for (seed, _), value in values.items() if seed == 0]):+.4f}",
        "seed1": f"{np.mean([value for (seed, _), value in values.items() if seed == 1]):+.4f}",
        "seed2": f"{np.mean([value for (seed, _), value in values.items() if seed == 2]):+.4f}",
    }


def main():
    rng = np.random.default_rng(20260712)
    gpt_baseline = load("R008_nomem_s3")
    qwen_baseline = load("Q_nomem_s3")
    loaded = {}
    rows = []
    for label, gpt_id, qwen_id in CONDITIONS:
        gpt = load(gpt_id)
        qwen = load(qwen_id)
        loaded[label] = (gpt, qwen)
        values = {
            key: (qwen[key] - qwen_baseline[key]) - (gpt[key] - gpt_baseline[key])
            for key in gpt_baseline
        }
        rows.append(make_row("executor_x_memory", label, values, rng))

    for representation in ["expel", "skillos"]:
        gpt_bm25, qwen_bm25 = loaded[f"{representation}_bm25"]
        gpt_dense, qwen_dense = loaded[f"{representation}_dense"]
        values = {
            key: (qwen_bm25[key] - qwen_dense[key]) - (gpt_bm25[key] - gpt_dense[key])
            for key in gpt_baseline
        }
        rows.append(
            make_row(
                "executor_x_retriever_within_representation",
                representation,
                values,
                rng,
            )
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT)


if __name__ == "__main__":
    main()
