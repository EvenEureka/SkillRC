#!/usr/bin/env python3
"""Analyze paired Qwen trajectory replays using predeclared behavioral markers."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "PAPERO_Q_mechanism_replay.jsonl"
ASSETS = ROOT.parent / "ARR" / "PaperO_assets"
TABLE = ASSETS / "table11_qwen_trajectory_mechanisms.csv"
REPORT = ROOT.parent / "ARR" / "ARR_2026-07-12_PaperO_Qwen_Trajectory_Mechanisms.md"

REQUIRED_VERB = {
    "clean": "clean",
    "cool": "cool",
    "heat": "heat",
    "examine": "use",
    "put": "put",
    "puttwo": "put",
}
INVALID_MARKERS = ["nothing happens", "not a valid", "can't", "cannot"]


def env_actions(row: dict) -> list[str]:
    return [
        step["action"].strip().lower()
        for step in row["trajectory"]
        if not step["action"].strip().lower().startswith("think:")
    ]


def trajectory_metrics(row: dict) -> dict:
    actions = env_actions(row)
    counts = Counter(actions)
    repeated = sum(max(0, count - 1) for count in counts.values())
    invalid = sum(
        any(marker in step.get("obs", "").lower() for marker in INVALID_MARKERS)
        for step in row["trajectory"]
    )
    required = REQUIRED_VERB.get(row["task_type"], "")
    return {
        "success": int(row["success"]),
        "steps": int(row["steps"]),
        "n_actions": len(actions),
        "n_unique_actions": len(counts),
        "repeat_ratio": repeated / len(actions) if actions else 0.0,
        "invalid_observations": invalid,
        "required_operation_present": int(any(action.startswith(required + " ") for action in actions)),
        "take_present": int(any(action.startswith("take ") for action in actions)),
        "actions": actions,
    }


def classify(base: dict, memory: dict) -> tuple[str, str]:
    if base["success"] == 0 and memory["success"] == 1:
        switch = "memory_rescue"
        if not base["required_operation_present"] and memory["required_operation_present"]:
            return switch, "procedural_completion"
        if memory["steps"] <= base["steps"] - 5:
            return switch, "search_efficiency"
        return switch, "rescue_unresolved"
    if base["success"] == 1 and memory["success"] == 0:
        switch = "memory_harm"
        if memory["repeat_ratio"] >= base["repeat_ratio"] + 0.15:
            return switch, "loop_amplification"
        if memory["invalid_observations"] > base["invalid_observations"]:
            return switch, "invalid_action_interference"
        if base["required_operation_present"] and not memory["required_operation_present"]:
            return switch, "procedure_displacement"
        if memory["steps"] >= base["steps"] + 5:
            return switch, "search_expansion"
        return switch, "harm_unresolved"
    if base["success"] == memory["success"] == 1:
        return "stable_success", "step_reduction" if memory["steps"] < base["steps"] else "no_success_change"
    return "stable_failure", "loop_reduction" if memory["repeat_ratio"] < base["repeat_ratio"] else "no_success_change"


def compact_actions(actions: list[str], limit: int = 8) -> str:
    if len(actions) <= limit:
        return " -> ".join(actions)
    return " -> ".join(actions[:4] + ["..."] + actions[-3:])


def main() -> None:
    with INPUT.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 54:
        raise ValueError(f"expected 54 replay episodes, found {len(rows)}")
    indexed = {(row["condition"], int(row["seed"]), str(row["task_id"])): row for row in rows}
    if len(indexed) != 54:
        raise ValueError("duplicate replay keys")

    exported = []
    for seed in [0, 1, 5]:
        for task_id in ["0", "5", "7", "30", "45", "55", "86", "113", "133"]:
            base_row = indexed[("nomem", seed, task_id)]
            memory_row = indexed[("expel_bm25", seed, task_id)]
            base = trajectory_metrics(base_row)
            memory = trajectory_metrics(memory_row)
            switch, mechanism = classify(base, memory)
            exported.append(
                {
                    "task_id": task_id,
                    "task_type": base_row["task_type"],
                    "seed": seed,
                    "switch": switch,
                    "mechanism_marker": mechanism,
                    "baseline_success": base["success"],
                    "memory_success": memory["success"],
                    "steps_delta": memory["steps"] - base["steps"],
                    "repeat_ratio_delta": f"{memory['repeat_ratio'] - base['repeat_ratio']:+.4f}",
                    "invalid_observation_delta": memory["invalid_observations"] - base["invalid_observations"],
                    "required_operation_baseline": base["required_operation_present"],
                    "required_operation_memory": memory["required_operation_present"],
                    "unique_actions_delta": memory["n_unique_actions"] - base["n_unique_actions"],
                    "baseline_actions": compact_actions(base["actions"]),
                    "memory_actions": compact_actions(memory["actions"]),
                }
            )
    ASSETS.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)

    by_switch = defaultdict(list)
    by_marker = Counter()
    for row in exported:
        by_switch[row["switch"]].append(row)
        by_marker[row["mechanism_marker"]] += 1
    lines = [
        "# Paper O Qwen Trajectory Mechanism Audit",
        "",
        "## Material Passport",
        "",
        "- **Mode:** local Qwen paired trajectory replay",
        "- **Episodes:** 54 (9 tasks x 3 prompt pairs x 2 memory conditions)",
        "- **External API:** none",
        "- **Status:** `ANALYZED`; behavioral markers, not causal identification",
        "",
        "## Aggregate Switches",
        "",
        "| Switch | N | Mean step delta | Mean repeat-ratio delta |",
        "|---|---:|---:|---:|",
    ]
    for switch, group in sorted(by_switch.items()):
        lines.append(
            f"| {switch} | {len(group)} | {np.mean([int(row['steps_delta']) for row in group]):+.2f} | "
            f"{np.mean([float(row['repeat_ratio_delta']) for row in group]):+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Predeclared Mechanism Markers",
            "",
            "| Marker | N |",
            "|---|---:|",
        ]
    )
    for marker, count in sorted(by_marker.items()):
        lines.append(f"| {marker} | {count} |")
    lines.extend(
        [
            "",
            "## Success-Switch Cases",
            "",
            "| Task | Type | Pair seed | Switch | Marker | Step delta | Required op (base/mem) |",
            "|---:|---|---:|---|---|---:|---|",
        ]
    )
    for row in exported:
        if row["switch"] in {"memory_rescue", "memory_harm"}:
            lines.append(
                f"| {row['task_id']} | {row['task_type']} | {row['seed']} | {row['switch']} | "
                f"{row['mechanism_marker']} | {row['steps_delta']:+d} | "
                f"{row['required_operation_baseline']}/{row['required_operation_memory']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The markers distinguish observable trajectory changes but do not isolate model weights, tokenizer, "
            "chat template, or attention behavior. A mechanism claim requires a controlled intervention on one of those factors.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines))
    print(TABLE)
    print(REPORT)
    print(json.dumps({"switches": {k: len(v) for k, v in by_switch.items()}, "markers": by_marker}, default=dict))


if __name__ == "__main__":
    main()
