#!/usr/bin/env python3
"""Finalize the ExpeL+BM25 result over all three few-shot combinations."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ASSETS = ROOT.parent / "ARR" / "PaperO_assets"
TABLE = ASSETS / "table7_complete_prompt_axis.csv"
DECISION = ASSETS / "complete_prompt_axis_decision.json"
REPORT = ROOT.parent / "ARR" / "ARR_2026-07-12_PaperO_CompletePromptAxis_Result.md"

RUNS = {
    "gpt-4o-mini": {
        "baseline_s3": "R008_nomem_s3",
        "variant_s3": "R012_expel_bm25",
        "baseline_s5": "R013_nomem_seed5",
        "variant_s5": "R013_expel_bm25_seed5",
    },
    "qwen3-8b": {
        "baseline_s3": "Q_nomem_s3",
        "variant_s3": "Q_expel_bm25_s3",
        "baseline_s5": "Q_nomem_seed5",
        "variant_s5": "Q_expel_bm25_seed5",
    },
}


def load(run_id: str) -> dict[tuple[int, str], int]:
    with (RESULTS / f"{run_id}.episodes.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    expected = 134 if run_id.endswith("seed5") else 402
    if len(rows) != expected:
        raise ValueError(f"{run_id}: expected {expected} episodes, found {len(rows)}")
    errors = [row for row in rows if row.get("error")]
    if errors:
        raise ValueError(f"{run_id}: found {len(errors)} errors")
    keyed = {(int(row["seed"]), str(row["task_id"])): int(row["success"]) for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"{run_id}: duplicate episode keys")
    return keyed


def bootstrap(values: dict[str, float], rng: np.random.Generator) -> tuple[float, float]:
    tasks = np.array(sorted(values), dtype=object)
    draws = np.empty(20_000, dtype=float)
    for index in range(len(draws)):
        sampled = rng.choice(tasks, size=len(tasks), replace=True)
        draws[index] = np.mean([values[task] for task in sampled])
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def fmt(value: float) -> str:
    return f"{value:+.4f}"


def main() -> None:
    rng = np.random.default_rng(20260712)
    gains = {}
    rows = []
    for executor, ids in RUNS.items():
        baseline_s3 = load(ids["baseline_s3"])
        variant_s3 = load(ids["variant_s3"])
        baseline_s5 = load(ids["baseline_s5"])
        variant_s5 = load(ids["variant_s5"])
        task_ids = sorted({task_id for _, task_id in baseline_s3})
        if set(task_ids) != {task_id for _, task_id in baseline_s5}:
            raise ValueError(f"{executor}: seed-5 task set differs from main campaign")

        effects_by_pair = {
            "02": {
                task: variant_s3[(0, task)] - baseline_s3[(0, task)] for task in task_ids
            },
            "12": {
                task: np.mean(
                    [
                        variant_s3[(seed, task)] - baseline_s3[(seed, task)]
                        for seed in [1, 2]
                    ]
                )
                for task in task_ids
            },
            "01": {
                task: variant_s5[(5, task)] - baseline_s5[(5, task)] for task in task_ids
            },
        }
        task_effects = {
            task: float(np.mean([effects_by_pair[pair][task] for pair in ["01", "02", "12"]]))
            for task in task_ids
        }
        gains[executor] = task_effects
        lo, hi = bootstrap(task_effects, rng)
        rows.append(
            {
                "effect_type": "memory_gain_complete_prompt_axis",
                "executor": executor,
                "condition": "expel_bm25",
                "estimate": fmt(np.mean(list(task_effects.values()))),
                "task_boot_ci95_lo": fmt(lo),
                "task_boot_ci95_hi": fmt(hi),
                "prompt_pair_01_effect": fmt(np.mean(list(effects_by_pair["01"].values()))),
                "prompt_pair_02_effect": fmt(np.mean(list(effects_by_pair["02"].values()))),
                "prompt_pair_12_effect": fmt(np.mean(list(effects_by_pair["12"].values()))),
                "unique_tasks": len(task_ids),
            }
        )

    did = {
        task: gains["qwen3-8b"][task] - gains["gpt-4o-mini"][task]
        for task in gains["gpt-4o-mini"]
    }
    did_lo, did_hi = bootstrap(did, rng)
    did_estimate = float(np.mean(list(did.values())))
    rows.append(
        {
            "effect_type": "executor_x_memory_complete_prompt_axis",
            "executor": "qwen3-8b_minus_gpt-4o-mini",
            "condition": "expel_bm25",
            "estimate": fmt(did_estimate),
            "task_boot_ci95_lo": fmt(did_lo),
            "task_boot_ci95_hi": fmt(did_hi),
            "prompt_pair_01_effect": "",
            "prompt_pair_02_effect": "",
            "prompt_pair_12_effect": "",
            "unique_tasks": len(did),
        }
    )

    ASSETS.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    robust = did_hi < 0
    decision = {
        "material_id": "PAPERO-COMPLETE-PROMPT-AXIS-2026-07-12-01",
        "verification_status": "ANALYZED",
        "executor_x_memory_estimate": did_estimate,
        "executor_x_memory_task_boot_ci95": [did_lo, did_hi],
        "decision": (
            "stop_executor_matrix_and_integrate_prompt_robust_moderation"
            if robust
            else "downgrade_executor_moderation_and_reassess_design"
        ),
        "robust_executor_moderation": robust,
        "next_experiment": "none" if robust else "targeted_prompt_scaffold_diagnostic",
    }
    with DECISION.open("w") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
        handle.write("\n")
    row_by_executor = {row["executor"]: row for row in rows}
    report = f"""# Paper O Complete Prompt-Axis Result

## Material Passport

- **Origin:** automatic postprocess for jobs `1181912` and `1181913`
- **Date:** `2026-07-12`
- **Verification Status:** `ANALYZED`
- **Scope:** ExpeL+BM25 versus no memory, all three two-of-three few-shot pairs

## Result

| Effect | Estimate | Task-bootstrap 95% CI | Pair 01 | Pair 02 | Pair 12 |
|---|---:|---:|---:|---:|---:|
| GPT memory gain | {row_by_executor['gpt-4o-mini']['estimate']} | [{row_by_executor['gpt-4o-mini']['task_boot_ci95_lo']}, {row_by_executor['gpt-4o-mini']['task_boot_ci95_hi']}] | {row_by_executor['gpt-4o-mini']['prompt_pair_01_effect']} | {row_by_executor['gpt-4o-mini']['prompt_pair_02_effect']} | {row_by_executor['gpt-4o-mini']['prompt_pair_12_effect']} |
| Qwen memory gain | {row_by_executor['qwen3-8b']['estimate']} | [{row_by_executor['qwen3-8b']['task_boot_ci95_lo']}, {row_by_executor['qwen3-8b']['task_boot_ci95_hi']}] | {row_by_executor['qwen3-8b']['prompt_pair_01_effect']} | {row_by_executor['qwen3-8b']['prompt_pair_02_effect']} | {row_by_executor['qwen3-8b']['prompt_pair_12_effect']} |
| Qwen-minus-GPT moderation | {row_by_executor['qwen3-8b_minus_gpt-4o-mini']['estimate']} | [{row_by_executor['qwen3-8b_minus_gpt-4o-mini']['task_boot_ci95_lo']}, {row_by_executor['qwen3-8b_minus_gpt-4o-mini']['task_boot_ci95_hi']}] | -- | -- | -- |

## Automatic Decision

`{decision['decision']}`

The executor-moderation result is classified as robust only if its task-bootstrap interval is entirely below zero. No additional representation matrix is authorized by this decision rule.
"""
    REPORT.write_text(report)
    print(TABLE)
    print(DECISION)
    print(REPORT)
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
