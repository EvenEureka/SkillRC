#!/usr/bin/env python3
"""Audit and reweight Paper O's few-shot seed axis.

Seeds choose two of three per-task-type demonstrations. Seeds 1 and 2 map to the
same demonstration pair, so the nominal three-seed average overweights that
prompt configuration. This script reports equal-weight estimates over the two
unique configurations that were actually observed.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT.parent / "ARR" / "PaperO_assets" / "table6_seed_axis_sensitivity.csv"

RUNS = {
    "gpt-4o-mini": {
        "baseline": "R008_nomem_s3",
        "expel_bm25": "R012_expel_bm25",
        "expel_dense": "R012_expel_dense",
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


def demo_pair(seed: int) -> str:
    order = list(range(3))
    random.Random(seed).shuffle(order)
    return "".join(str(value) for value in sorted(order[:2]))


def load(run_id: str) -> dict[tuple[int, str], int]:
    with (RESULTS / f"{run_id}.episodes.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {(int(row["seed"]), str(row["task_id"])): int(row["success"]) for row in rows}


def task_bootstrap_ci(values: dict[str, float], rng: np.random.Generator) -> tuple[float, float]:
    task_ids = np.array(sorted(values), dtype=object)
    draws = np.empty(10_000, dtype=float)
    for index in range(len(draws)):
        sampled = rng.choice(task_ids, size=len(task_ids), replace=True)
        draws[index] = np.mean([values[task_id] for task_id in sampled])
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def fmt(value: float) -> str:
    return f"{value:+.4f}"


def main() -> None:
    rng = np.random.default_rng(20260712)
    seed_pairs = {seed: demo_pair(seed) for seed in [0, 1, 2]}
    grouped_seeds = defaultdict(list)
    for seed, pair in seed_pairs.items():
        grouped_seeds[pair].append(seed)
    observed_pairs = sorted(grouped_seeds)
    all_pairs = {"01", "02", "12"}
    missing_pairs = sorted(all_pairs - set(observed_pairs))
    if observed_pairs != ["02", "12"] or missing_pairs != ["01"]:
        raise RuntimeError(f"Unexpected seed mapping: {seed_pairs}")

    loaded = {
        (executor, condition): load(run_id)
        for executor, conditions in RUNS.items()
        for condition, run_id in conditions.items()
    }
    task_ids = sorted({task_id for _, task_id in loaded[("gpt-4o-mini", "baseline")]})
    rows = []
    per_task_gains = {}

    for executor, conditions in RUNS.items():
        baseline = loaded[(executor, "baseline")]
        baseline_disagreement = np.mean(
            [baseline[(1, task_id)] != baseline[(2, task_id)] for task_id in task_ids]
        )
        for condition in [key for key in conditions if key != "baseline"]:
            variant = loaded[(executor, condition)]
            pair_effects = {}
            task_effects = {}
            for pair, seeds in grouped_seeds.items():
                values = [
                    variant[(seed, task_id)] - baseline[(seed, task_id)]
                    for seed in seeds
                    for task_id in task_ids
                ]
                pair_effects[pair] = float(np.mean(values))
            for task_id in task_ids:
                prompt_means = []
                for seeds in grouped_seeds.values():
                    prompt_means.append(
                        np.mean(
                            [variant[(seed, task_id)] - baseline[(seed, task_id)] for seed in seeds]
                        )
                    )
                task_effects[task_id] = float(np.mean(prompt_means))
            per_task_gains[(executor, condition)] = task_effects
            lo, hi = task_bootstrap_ci(task_effects, rng)
            variant_disagreement = np.mean(
                [variant[(1, task_id)] != variant[(2, task_id)] for task_id in task_ids]
            )
            rows.append(
                {
                    "effect_type": "memory_gain_unique_prompt_equal_weight",
                    "executor": executor,
                    "condition": condition,
                    "estimate": fmt(np.mean(list(task_effects.values()))),
                    "task_boot_ci95_lo": fmt(lo),
                    "task_boot_ci95_hi": fmt(hi),
                    "prompt_pair_02_effect": fmt(pair_effects["02"]),
                    "prompt_pair_12_effect": fmt(pair_effects["12"]),
                    "duplicate_pair_seed12_disagreement_baseline": f"{baseline_disagreement:.4f}",
                    "duplicate_pair_seed12_disagreement_variant": f"{variant_disagreement:.4f}",
                    "observed_prompt_pairs": ";".join(observed_pairs),
                    "missing_prompt_pairs": ";".join(missing_pairs),
                }
            )

    for condition in ["expel_bm25", "expel_dense", "skillos_bm25", "skillos_dense"]:
        values = {
            task_id: (
                per_task_gains[("qwen3-8b", condition)][task_id]
                - per_task_gains[("gpt-4o-mini", condition)][task_id]
            )
            for task_id in task_ids
        }
        lo, hi = task_bootstrap_ci(values, rng)
        rows.append(
            {
                "effect_type": "executor_x_memory_unique_prompt_equal_weight",
                "executor": "qwen3-8b_minus_gpt-4o-mini",
                "condition": condition,
                "estimate": fmt(np.mean(list(values.values()))),
                "task_boot_ci95_lo": fmt(lo),
                "task_boot_ci95_hi": fmt(hi),
                "prompt_pair_02_effect": "",
                "prompt_pair_12_effect": "",
                "duplicate_pair_seed12_disagreement_baseline": "",
                "duplicate_pair_seed12_disagreement_variant": "",
                "observed_prompt_pairs": ";".join(observed_pairs),
                "missing_prompt_pairs": ";".join(missing_pairs),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"seed_pairs": seed_pairs, "missing_pairs": missing_pairs}, sort_keys=True))
    print(OUT)


if __name__ == "__main__":
    main()
