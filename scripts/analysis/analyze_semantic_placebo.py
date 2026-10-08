#!/usr/bin/env python3
"""Analyze the frozen Paper O semantic-placebo gate."""
from __future__ import annotations

import csv
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT.parent / "ARR" / "PaperO_assets"
SEEDS = (0, 1, 2, 5)
PAIR_SEEDS = ((5,), (0,), (1, 2))
N_BOOT = 20_000
BOOT_SEED = 20260713


FILES = {
    "gpt": {
        "baseline": ("R008_nomem_s3", "R013_nomem_seed5"),
        "true": ("R012_expel_bm25", "R013_expel_bm25_seed5"),
        "placebo": ("PAPERO_PLACEBO_GPT", "PAPERO_PLACEBO_GPT_S2"),
    },
    "qwen": {
        "baseline": ("Q_nomem_s3", "Q_nomem_seed5"),
        "true": ("Q_expel_bm25_s3", "Q_expel_bm25_seed5"),
        "placebo": ("PAPERO_PLACEBO_QWEN", "PAPERO_PLACEBO_QWEN_S2"),
    },
}


def load_runs(run_ids: tuple[str, ...]) -> dict[tuple[int, int], dict]:
    rows = {}
    for run_id in run_ids:
        path = RESULTS / f"{run_id}.episodes.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seed = int(row["seed"])
            if seed not in SEEDS:
                continue
            key = (seed, int(row["task_id"]))
            if key in rows:
                raise ValueError(f"duplicate episode key {key} across {run_ids}")
            if row.get("error"):
                raise ValueError(f"episode error in {run_id} at {key}: {row['error']}")
            rows[key] = row
    return rows


def percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def stratified_bootstrap(values: dict[int, float], task_types: dict[int, str]) -> np.ndarray:
    by_type: dict[str, list[int]] = defaultdict(list)
    for task_id in values:
        by_type[task_types[task_id]].append(task_id)
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.empty(N_BOOT, dtype=float)
    groups = [np.asarray([values[t] for t in sorted(ids)]) for _, ids in sorted(by_type.items())]
    for draw in range(N_BOOT):
        sampled = [group[rng.integers(0, len(group), size=len(group))] for group in groups]
        draws[draw] = np.concatenate(sampled).mean()
    return draws


def main() -> None:
    frozen = json.loads((RESULTS / "placebo" / "frozen_tasks.json").read_text())
    task_ids = [int(task_id) for task_id in frozen["task_ids"]]
    task_types = {
        int(task_id): task_type
        for task_type, ids in frozen["by_type"].items()
        for task_id in ids
    }
    expected = {(seed, task_id) for seed in SEEDS for task_id in task_ids}

    loaded = {
        executor: {condition: load_runs(run_ids) for condition, run_ids in conditions.items()}
        for executor, conditions in FILES.items()
    }
    for executor, conditions in loaded.items():
        for condition, rows in conditions.items():
            missing = expected - set(rows)
            if missing:
                raise ValueError(f"{executor}/{condition} missing {len(missing)} frozen episodes")
            if condition == "placebo":
                realized_tokens = {int(rows[key]["memory_tokens"]) for key in expected}
                if realized_tokens != {645}:
                    raise ValueError(f"{executor}/placebo footprint mismatch: {realized_tokens}")

    per_task: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)
    task_rows = []
    for executor, conditions in loaded.items():
        means = {
            condition: {
                task_id: float(np.mean([
                    np.mean([rows[(seed, task_id)]["success"] for seed in pair_seeds])
                    for pair_seeds in PAIR_SEEDS
                ]))
                for task_id in task_ids
            }
            for condition, rows in conditions.items()
        }
        per_task[executor] = means
        for task_id in task_ids:
            task_rows.append({
                "executor": executor,
                "task_id": task_id,
                "task_type": task_types[task_id],
                "baseline": means["baseline"][task_id],
                "true": means["true"][task_id],
                "placebo": means["placebo"][task_id],
                "true_effect": means["true"][task_id] - means["baseline"][task_id],
                "placebo_effect": means["placebo"][task_id] - means["baseline"][task_id],
                "semantic_increment": means["true"][task_id] - means["placebo"][task_id],
            })

    estimates = []
    bootstrap = {}
    for executor in ("gpt", "qwen"):
        contrasts = {
            "true_effect": {t: per_task[executor]["true"][t] - per_task[executor]["baseline"][t] for t in task_ids},
            "placebo_effect": {t: per_task[executor]["placebo"][t] - per_task[executor]["baseline"][t] for t in task_ids},
            "semantic_increment": {t: per_task[executor]["true"][t] - per_task[executor]["placebo"][t] for t in task_ids},
        }
        for contrast, values in contrasts.items():
            draws = stratified_bootstrap(values, task_types)
            bootstrap[(executor, contrast)] = draws
            lo, hi = percentile_ci(draws)
            estimates.append({
                "executor": executor,
                "contrast": contrast,
                "estimate": float(np.mean(list(values.values()))),
                "ci95_lo": lo,
                "ci95_hi": hi,
            })

    interaction_values = {
        task_id: (
            per_task["qwen"]["true"][task_id] - per_task["qwen"]["placebo"][task_id]
            - per_task["gpt"]["true"][task_id] + per_task["gpt"]["placebo"][task_id]
        )
        for task_id in task_ids
    }
    interaction_draws = stratified_bootstrap(interaction_values, task_types)
    interaction_ci = percentile_ci(interaction_draws)
    interaction = {
        "executor": "qwen_minus_gpt",
        "contrast": "semantic_increment_interaction",
        "estimate": float(np.mean(list(interaction_values.values()))),
        "ci95_lo": interaction_ci[0],
        "ci95_hi": interaction_ci[1],
    }
    estimates.append(interaction)

    condition_means = {
        executor: {
            condition: float(np.mean(list(values.values())))
            for condition, values in conditions.items()
        }
        for executor, conditions in per_task.items()
    }
    prompt_pair_means = {}
    for executor, conditions in loaded.items():
        prompt_pair_means[executor] = {}
        for condition, rows in conditions.items():
            prompt_pair_means[executor][condition] = {
                pair: float(np.mean([
                    rows[(seed, task_id)]["success"]
                    for seed in seeds
                    for task_id in task_ids
                ]))
                for pair, seeds in (("01", (5,)), ("02", (0,)), ("12", (1, 2)))
            }

    gpt_sem = next(row for row in estimates if row["executor"] == "gpt" and row["contrast"] == "semantic_increment")
    qwen_sem = next(row for row in estimates if row["executor"] == "qwen" and row["contrast"] == "semantic_increment")
    if gpt_sem["ci95_lo"] > 0 and interaction["ci95_hi"] < 0:
        decision = "executor_dependent_order_sensitive_semantics_supported"
    elif gpt_sem["ci95_lo"] >= -0.05 and gpt_sem["ci95_hi"] <= 0.05:
        decision = "gpt_true_placebo_equivalent_reframe_as_prompt_effect"
    else:
        decision = "semantic_mechanism_inconclusive"

    for row in task_rows:
        row["baseline"] = f"{row['baseline']:.4f}"
        row["true"] = f"{row['true']:.4f}"
        row["placebo"] = f"{row['placebo']:.4f}"
        row["true_effect"] = f"{row['true_effect']:+.4f}"
        row["placebo_effect"] = f"{row['placebo_effect']:+.4f}"
        row["semantic_increment"] = f"{row['semantic_increment']:+.4f}"

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "semantic_placebo_task_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=task_rows[0].keys())
        writer.writeheader()
        writer.writerows(task_rows)
    with (OUT / "semantic_placebo_estimates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=estimates[0].keys())
        writer.writeheader()
        writer.writerows(estimates)

    result = {
        "protocol": "ARR_2026-07-13_PaperO_SemanticPlacebo_Protocol.md",
        "n_tasks": len(task_ids),
        "tasks_per_type": frozen["tasks_per_type"],
        "prompt_pair_seed_groups": [list(group) for group in PAIR_SEEDS],
        "bootstrap": {"kind": "stratified_task", "draws": N_BOOT, "seed": BOOT_SEED},
        "condition_means": condition_means,
        "prompt_pair_means": prompt_pair_means,
        "placebo_realized_memory_tokens": {"gpt": [645], "qwen": [645]},
        "estimates": estimates,
        "gpt_true_vs_placebo_counts": {
            "wins": sum(per_task["gpt"]["true"][t] > per_task["gpt"]["placebo"][t] for t in task_ids),
            "losses": sum(per_task["gpt"]["true"][t] < per_task["gpt"]["placebo"][t] for t in task_ids),
            "ties": sum(per_task["gpt"]["true"][t] == per_task["gpt"]["placebo"][t] for t in task_ids),
        },
        "qwen_true_vs_placebo_counts": {
            "wins": sum(per_task["qwen"]["true"][t] > per_task["qwen"]["placebo"][t] for t in task_ids),
            "losses": sum(per_task["qwen"]["true"][t] < per_task["qwen"]["placebo"][t] for t in task_ids),
            "ties": sum(per_task["qwen"]["true"][t] == per_task["qwen"]["placebo"][t] for t in task_ids),
        },
        "decision": decision,
        "qwen_semantic_increment_for_context": qwen_sem,
    }
    (OUT / "semantic_placebo_decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
