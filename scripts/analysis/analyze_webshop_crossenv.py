#!/usr/bin/env python3
"""Analyze the minimal train-disjoint WebShop memory comparison."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ASSETS = ROOT.parent / "ARR" / "PaperO_assets"
TABLE = ASSETS / "table12_webshop_crossenv.csv"
DECISION = ASSETS / "webshop_crossenv_decision.json"
REPORT = ROOT.parent / "ARR" / "ARR_2026-07-12_PaperO_WebShop_CrossEnvironment.md"


def load(run_id: str) -> dict[str, dict]:
    path = RESULTS / f"{run_id}.episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 100:
        raise ValueError(f"{run_id}: expected 100 episodes, found {len(rows)}")
    if any(row.get("error") for row in rows):
        raise ValueError(f"{run_id}: contains errors")
    indexed = {str(row["task_id"]): row for row in rows}
    if len(indexed) != 100:
        raise ValueError(f"{run_id}: duplicate task IDs")
    return indexed


def bootstrap(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = np.empty(20_000, dtype=float)
    n = len(values)
    for index in range(len(draws)):
        draws[index] = rng.choice(values, size=n, replace=True).mean()
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    baseline = load("PAPERO_WS_Q_nomem_eval100")
    memory = load("PAPERO_WS_Q_expel_bm25_eval100")
    if set(baseline) != set(memory):
        raise ValueError("paired task sets differ")
    metadata = json.loads((RESULTS / "pools" / "webshop_expel_insights.metadata.json").read_text())
    if not metadata.get("disjoint"):
        raise ValueError("train/eval separation check failed")

    task_ids = sorted(baseline, key=int)
    reward_delta = np.array([float(memory[t]["reward"]) - float(baseline[t]["reward"]) for t in task_ids])
    success_delta = np.array([int(memory[t]["success"]) - int(baseline[t]["success"]) for t in task_ids])
    step_delta = np.array([float(memory[t]["steps"]) - float(baseline[t]["steps"]) for t in task_ids])
    prompt_delta = np.array([float(memory[t]["prompt_tokens"]) - float(baseline[t]["prompt_tokens"]) for t in task_ids])
    rng = np.random.default_rng(20260712)
    reward_ci = bootstrap(reward_delta, rng)
    success_ci = bootstrap(success_delta, rng)
    reward_wins = int((reward_delta > 0).sum())
    reward_losses = int((reward_delta < 0).sum())
    nonzero = reward_wins + reward_losses
    directional = reward_ci[0] > 0 or reward_ci[1] < 0
    heterogeneous = nonzero >= 20 and reward_wins >= 5 and reward_losses >= 5
    supported = directional or heterogeneous

    rows = [
        {
            "metric": "environment_reward",
            "baseline_mean": f"{np.mean([float(baseline[t]['reward']) for t in task_ids]):.4f}",
            "memory_mean": f"{np.mean([float(memory[t]['reward']) for t in task_ids]):.4f}",
            "paired_delta": f"{reward_delta.mean():+.4f}",
            "task_boot_ci95_lo": f"{reward_ci[0]:+.4f}",
            "task_boot_ci95_hi": f"{reward_ci[1]:+.4f}",
        },
        {
            "metric": "exact_success",
            "baseline_mean": f"{np.mean([int(baseline[t]['success']) for t in task_ids]):.4f}",
            "memory_mean": f"{np.mean([int(memory[t]['success']) for t in task_ids]):.4f}",
            "paired_delta": f"{success_delta.mean():+.4f}",
            "task_boot_ci95_lo": f"{success_ci[0]:+.4f}",
            "task_boot_ci95_hi": f"{success_ci[1]:+.4f}",
        },
        {
            "metric": "steps",
            "baseline_mean": f"{np.mean([float(baseline[t]['steps']) for t in task_ids]):.4f}",
            "memory_mean": f"{np.mean([float(memory[t]['steps']) for t in task_ids]):.4f}",
            "paired_delta": f"{step_delta.mean():+.4f}",
            "task_boot_ci95_lo": "",
            "task_boot_ci95_hi": "",
        },
        {
            "metric": "prompt_tokens",
            "baseline_mean": f"{np.mean([float(baseline[t]['prompt_tokens']) for t in task_ids]):.1f}",
            "memory_mean": f"{np.mean([float(memory[t]['prompt_tokens']) for t in task_ids]):.1f}",
            "paired_delta": f"{prompt_delta.mean():+.1f}",
            "task_boot_ci95_lo": "",
            "task_boot_ci95_hi": "",
        },
    ]
    ASSETS.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    decision = {
        "material_id": "PAPERO-WEBSHOP-CROSSENV-2026-07-12-01",
        "verification_status": "ANALYZED",
        "train_eval_disjoint": True,
        "reward_wins": reward_wins,
        "reward_losses": reward_losses,
        "reward_ties": 100 - nonzero,
        "directional_mean_effect": directional,
        "heterogeneous_task_effects": heterogeneous,
        "cross_environment_conditionality_supported": supported,
        "decision": "freeze_claim_and_write" if supported else "freeze_as_single_environment_case_study",
        "next_experiment": "none",
    }
    DECISION.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    REPORT.write_text(
        "# Paper O WebShop Cross-Environment Validation\n\n"
        "## Material Passport\n\n"
        "- **Executor:** local Qwen3-8B\n"
        "- **Evaluation:** WebShop small, sessions 0-99\n"
        "- **Memory source:** train-only sessions 500-519\n"
        "- **External API:** none\n"
        "- **Status:** `ANALYZED`\n\n"
        "## Results\n\n"
        "| Metric | No memory | ExpeL-style memory | Paired delta | Task-bootstrap 95% CI |\n"
        "|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['metric']} | {row['baseline_mean']} | {row['memory_mean']} | {row['paired_delta']} | "
            f"[{row['task_boot_ci95_lo']}, {row['task_boot_ci95_hi']}] |"
            for row in rows[:2]
        )
        + "\n\n"
        f"Reward wins/losses/ties: `{reward_wins}/{reward_losses}/{100 - nonzero}`.\n\n"
        "## Decision\n\n"
        f"`{decision['decision']}`\n\n"
        "This validation tests whether conditional memory utility appears outside ALFWorld. It does not test "
        "executor moderation in WebShop because only one executor is used.\n"
    )
    print(json.dumps(decision, sort_keys=True))
    print(TABLE)
    print(REPORT)


if __name__ == "__main__":
    main()
