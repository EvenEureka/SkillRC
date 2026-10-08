#!/usr/bin/env python3
"""Create a zero-API task-level mechanism audit for Paper O."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ASSETS = ROOT.parent / "ARR" / "PaperO_assets"
DETAIL = ASSETS / "table9_task_mechanism_cases.csv"
SUMMARY = ASSETS / "table10_mechanism_summary.csv"
CASEBOOK = ROOT.parent / "ARR" / "ARR_2026-07-12_PaperO_Mechanism_Casebook.md"

RUNS = {
    "gpt": {
        "baseline_s3": "R008_nomem_s3",
        "variant_s3": "R012_expel_bm25",
        "baseline_s5": "R013_nomem_seed5",
        "variant_s5": "R013_expel_bm25_seed5",
    },
    "qwen": {
        "baseline_s3": "Q_nomem_s3",
        "variant_s3": "Q_expel_bm25_s3",
        "baseline_s5": "Q_nomem_seed5",
        "variant_s5": "Q_expel_bm25_seed5",
    },
}


def load(run_id: str) -> dict[tuple[int, str], dict]:
    with (RESULTS / f"{run_id}.episodes.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {(int(row["seed"]), str(row["task_id"])): row for row in rows}


def prompt_average(s3: dict, s5: dict, task_id: str, field: str) -> float:
    pair_02 = float(s3[(0, task_id)][field])
    pair_12 = np.mean([float(s3[(seed, task_id)][field]) for seed in [1, 2]])
    pair_01 = float(s5[(5, task_id)][field])
    return float(np.mean([pair_01, pair_02, pair_12]))


def pair_effects(base_s3: dict, var_s3: dict, base_s5: dict, var_s5: dict, task_id: str) -> dict:
    return {
        "01": float(var_s5[(5, task_id)]["success"] - base_s5[(5, task_id)]["success"]),
        "02": float(var_s3[(0, task_id)]["success"] - base_s3[(0, task_id)]["success"]),
        "12": float(
            np.mean(
                [
                    var_s3[(seed, task_id)]["success"] - base_s3[(seed, task_id)]["success"]
                    for seed in [1, 2]
                ]
            )
        ),
    }


def sign(value: float) -> str:
    if value > 1e-12:
        return "help"
    if value < -1e-12:
        return "hurt"
    return "neutral"


def objective(text: str) -> str:
    marker = "Your task is to:"
    if marker in text:
        return text.split(marker, 1)[1].splitlines()[0].strip()
    return text.splitlines()[0].strip()


def mechanism_candidate(label: str, qwen_baseline: float, first_match_rank: int) -> str:
    if label == "gpt_help__qwen_hurt" and qwen_baseline >= 2 / 3:
        return "baseline_ceiling_plus_prompt_interference"
    if label.endswith("qwen_hurt") and first_match_rank > 10:
        return "task_rule_ordering_mismatch"
    if label == "both_hurt":
        return "generic_prompt_interference"
    if label == "both_help":
        return "task_matched_scaffolding"
    if label.startswith("gpt_help"):
        return "executor_specific_prompt_use"
    return "mixed_or_insufficient_evidence"


def main() -> None:
    loaded = {
        (executor, key): load(run_id)
        for executor, run_ids in RUNS.items()
        for key, run_id in run_ids.items()
    }
    pool = [json.loads(line) for line in (RESULTS / "pools" / "expel_insights.jsonl").open() if line.strip()]
    bm25 = BM25Okapi([item["text"].lower().split() for item in pool])
    gpt_base_s3 = loaded[("gpt", "baseline_s3")]
    task_ids = sorted({task_id for _, task_id in gpt_base_s3}, key=int)
    details = []

    for task_id in task_ids:
        reference = gpt_base_s3[(0, task_id)]
        query = reference["task_text"]
        scores = bm25.get_scores(query.lower().split())
        order = sorted(range(len(pool)), key=lambda index: scores[index], reverse=True)
        task_type = reference["task_type"]
        matched_ranks = [rank + 1 for rank, index in enumerate(order) if pool[index]["task_type"] == task_type]

        executor_stats = {}
        for executor in ["gpt", "qwen"]:
            base_s3 = loaded[(executor, "baseline_s3")]
            var_s3 = loaded[(executor, "variant_s3")]
            base_s5 = loaded[(executor, "baseline_s5")]
            var_s5 = loaded[(executor, "variant_s5")]
            effects = pair_effects(base_s3, var_s3, base_s5, var_s5, task_id)
            baseline = prompt_average(base_s3, base_s5, task_id, "success")
            variant = prompt_average(var_s3, var_s5, task_id, "success")
            executor_stats[executor] = {
                "baseline": baseline,
                "variant": variant,
                "gain": variant - baseline,
                "effects": effects,
                "steps_delta": (
                    prompt_average(var_s3, var_s5, task_id, "steps")
                    - prompt_average(base_s3, base_s5, task_id, "steps")
                ),
                "prompt_delta": (
                    prompt_average(var_s3, var_s5, task_id, "prompt_tokens")
                    - prompt_average(base_s3, base_s5, task_id, "prompt_tokens")
                ),
            }

        gpt_sign = sign(executor_stats["gpt"]["gain"])
        qwen_sign = sign(executor_stats["qwen"]["gain"])
        if gpt_sign == qwen_sign == "help":
            label = "both_help"
        elif gpt_sign == qwen_sign == "hurt":
            label = "both_hurt"
        else:
            label = f"gpt_{gpt_sign}__qwen_{qwen_sign}"
        candidate = mechanism_candidate(
            label,
            executor_stats["qwen"]["baseline"],
            min(matched_ranks),
        )
        details.append(
            {
                "task_id": task_id,
                "task_type": task_type,
                "objective": objective(query),
                "transfer_label": label,
                "mechanism_candidate": candidate,
                "gpt_baseline": f"{executor_stats['gpt']['baseline']:.4f}",
                "gpt_expel_bm25": f"{executor_stats['gpt']['variant']:.4f}",
                "gpt_gain": f"{executor_stats['gpt']['gain']:+.4f}",
                "qwen_baseline": f"{executor_stats['qwen']['baseline']:.4f}",
                "qwen_expel_bm25": f"{executor_stats['qwen']['variant']:.4f}",
                "qwen_gain": f"{executor_stats['qwen']['gain']:+.4f}",
                "gpt_pair01_effect": f"{executor_stats['gpt']['effects']['01']:+.1f}",
                "gpt_pair02_effect": f"{executor_stats['gpt']['effects']['02']:+.1f}",
                "gpt_pair12_effect": f"{executor_stats['gpt']['effects']['12']:+.1f}",
                "qwen_pair01_effect": f"{executor_stats['qwen']['effects']['01']:+.1f}",
                "qwen_pair02_effect": f"{executor_stats['qwen']['effects']['02']:+.1f}",
                "qwen_pair12_effect": f"{executor_stats['qwen']['effects']['12']:+.1f}",
                "gpt_steps_delta": f"{executor_stats['gpt']['steps_delta']:+.2f}",
                "qwen_steps_delta": f"{executor_stats['qwen']['steps_delta']:+.2f}",
                "gpt_prompt_tokens_delta": f"{executor_stats['gpt']['prompt_delta']:+.1f}",
                "qwen_prompt_tokens_delta": f"{executor_stats['qwen']['prompt_delta']:+.1f}",
                "expel_pool_items": len(pool),
                "expel_injected_items": len(pool),
                "first_task_matched_rule_rank": min(matched_ranks),
                "task_matched_rules_in_top5": sum(rank <= 5 for rank in matched_ranks),
                "mean_task_matched_rule_rank": f"{np.mean(matched_ranks):.2f}",
            }
        )

    ASSETS.mkdir(parents=True, exist_ok=True)
    with DETAIL.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)

    groups = defaultdict(list)
    for row in details:
        groups[row["transfer_label"]].append(row)
    summaries = []
    for label, rows in sorted(groups.items()):
        type_counts = Counter(row["task_type"] for row in rows)
        mechanism_counts = Counter(row["mechanism_candidate"] for row in rows)
        summaries.append(
            {
                "transfer_label": label,
                "n_tasks": len(rows),
                "task_type_counts": json.dumps(type_counts, sort_keys=True),
                "mechanism_candidate_counts": json.dumps(mechanism_counts, sort_keys=True),
                "mean_gpt_gain": f"{np.mean([float(row['gpt_gain']) for row in rows]):+.4f}",
                "mean_qwen_gain": f"{np.mean([float(row['qwen_gain']) for row in rows]):+.4f}",
                "mean_qwen_baseline": f"{np.mean([float(row['qwen_baseline']) for row in rows]):.4f}",
                "mean_first_matched_rank": f"{np.mean([int(row['first_task_matched_rule_rank']) for row in rows]):.2f}",
            }
        )
    with SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    selected = []
    for label, rows in sorted(groups.items()):
        ranked = sorted(
            rows,
            key=lambda row: abs(float(row["gpt_gain"])) + abs(float(row["qwen_gain"])),
            reverse=True,
        )
        selected.extend(ranked[:3])
    lines = [
        "# Paper O Mechanism Casebook",
        "",
        "## Material Passport",
        "",
        "- **Mode:** zero-API post-hoc mechanism audit",
        "- **Status:** `ANALYZED`, hypotheses only",
        "- **Primary condition:** ExpeL+BM25 versus no memory over three unique prompt pairs",
        "- **Key implementation fact:** all 33 ExpeL rules are injected on every task; BM25 changes their order, not their membership",
        "",
        "## Transfer Taxonomy",
        "",
        "| Label | Tasks | Mean GPT gain | Mean Qwen gain | Mean Qwen baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['transfer_label']} | {row['n_tasks']} | {row['mean_gpt_gain']} | "
            f"{row['mean_qwen_gain']} | {row['mean_qwen_baseline']} |"
        )
    lines.extend(
        [
            "",
            "## Representative Cases",
            "",
            "| Task | Type | Transfer | GPT gain | Qwen gain | Qwen baseline | First matched-rule rank | Candidate explanation |",
            "|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in selected:
        lines.append(
            f"| {row['task_id']} | {row['task_type']} | {row['transfer_label']} | {row['gpt_gain']} | "
            f"{row['qwen_gain']} | {row['qwen_baseline']} | {row['first_task_matched_rule_rank']} | "
            f"{row['mechanism_candidate']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "These labels prioritize cases for trajectory replay. They do not identify a causal mechanism. "
            "Baseline ceiling, rule ordering, and prompt interference can co-occur. Executor refers to the tested model/runtime stack.",
            "",
        ]
    )
    CASEBOOK.write_text("\n".join(lines))
    print(DETAIL)
    print(SUMMARY)
    print(CASEBOOK)


if __name__ == "__main__":
    main()
