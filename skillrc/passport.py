"""Training-free admission cards for external memory payloads."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


SCHEMA_VERSION = "memory-passport-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_episode_runs(results_dir: str | Path, run_ids: Iterable[str]) -> dict[tuple[int, int], dict]:
    rows: dict[tuple[int, int], dict] = {}
    for run_id in run_ids:
        path = Path(results_dir) / f"{run_id}.episodes.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["seed"]), int(row["task_id"]))
            if key in rows:
                raise ValueError(f"duplicate episode key {key} across {list(run_ids)}")
            if row.get("error"):
                raise ValueError(f"episode error in {run_id} at {key}: {row['error']}")
            rows[key] = row
    return rows


def _pair_statistics(deltas: list[float], z_value: float) -> dict[str, Any]:
    if len(deltas) < 2:
        raise ValueError("a passport prompt condition needs at least two paired tasks")
    average = mean(deltas)
    standard_error = stdev(deltas) / math.sqrt(len(deltas))
    return {
        "n_tasks": len(deltas),
        "mean_delta": average,
        "standard_error": standard_error,
        "lower_bound": average - z_value * standard_error,
        "wins": sum(delta > 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
    }


def build_global_passport(
    baseline: dict[tuple[int, int], dict],
    payload: dict[tuple[int, int], dict],
    pair_seeds: dict[str, tuple[int, ...]],
    task_ids: Iterable[int],
    metric: str,
    z_value: float,
    executor: dict[str, Any],
    payload_spec: dict[str, Any],
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a global card; deployment requires a positive LCB in every prompt pair."""
    selected = sorted(set(int(task_id) for task_id in task_ids))
    if not selected:
        raise ValueError("passport calibration task set is empty")
    if metric not in {"success", "reward"}:
        raise ValueError("passport metric must be 'success' or 'reward'")
    pair_stats = {}
    missing = []
    for pair, seeds in pair_seeds.items():
        deltas = []
        for task_id in selected:
            keys = [(seed, task_id) for seed in seeds]
            missing.extend(key for key in keys if key not in baseline or key not in payload)
            if any(key not in baseline or key not in payload for key in keys):
                continue
            baseline_value = mean(float(baseline[key][metric]) for key in keys)
            payload_value = mean(float(payload[key][metric]) for key in keys)
            deltas.append(payload_value - baseline_value)
        if missing:
            continue
        pair_stats[pair] = {
            "seeds": list(seeds),
            **_pair_statistics(deltas, z_value),
        }
    if missing:
        raise ValueError(f"missing {len(set(missing))} paired episode keys; first={missing[0]}")
    deploy = all(stats["lower_bound"] > 0 for stats in pair_stats.values())
    limiting_pair = min(pair_stats, key=lambda pair: pair_stats[pair]["lower_bound"])
    decision = {
        "deploy": deploy,
        "status": "deployable" if deploy else "quarantined",
        "rule": "all_prompt_pair_lower_bounds_gt_zero",
        "limiting_pair": limiting_pair,
        "limiting_lower_bound": pair_stats[limiting_pair]["lower_bound"],
        "reason": (
            "every prompt-pair lower bound is positive"
            if deploy else "at least one prompt-pair lower bound is non-positive"
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "global",
        "executor": executor,
        "payload": payload_spec,
        "calibration": {
            "metric": metric,
            "z_value": z_value,
            "task_ids": selected,
            "n_tasks": len(selected),
            "prompt_pairs": pair_stats,
            "selection": selection or {"kind": "explicit"},
        },
        "decision": decision,
        "semantic_certificate": {
            "status": "not_evaluated",
            "note": "deployment utility and placebo-based semantic certification are separate",
        },
    }


def stratified_task_sample(
    rows: dict[tuple[int, int], dict], per_type: int, seed: int
) -> tuple[list[int], dict[str, list[int]]]:
    """Select calibration tasks from metadata only, using the lowest available seed."""
    import random

    available_seed = min(run_seed for run_seed, _ in rows)
    groups: dict[str, list[int]] = defaultdict(list)
    for (run_seed, task_id), row in rows.items():
        if run_seed == available_seed:
            groups[str(row["task_type"])].append(task_id)
    rng = random.Random(seed)
    chosen_by_type = {}
    for task_type, ids in sorted(groups.items()):
        unique_ids = sorted(set(ids))
        if len(unique_ids) <= per_type:
            raise ValueError(f"{task_type} has only {len(unique_ids)} tasks for per_type={per_type}")
        chosen_by_type[task_type] = sorted(rng.sample(unique_ids, per_type))
    selected = sorted(task_id for ids in chosen_by_type.values() for task_id in ids)
    return selected, chosen_by_type
