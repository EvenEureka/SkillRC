#!/usr/bin/env python3
"""Retrospective feasibility checks for training-free memory evolution ideas.

The replay experiments only use outcomes from already-completed ALFWorld runs.
Calibration and evaluation tasks are disjoint.  Where seed-5 outcomes exist,
calibration uses prompt pairs 02/12 and evaluation uses the unseen pair 01.
These checks are intended to reject weak ideas, not to replace prospective runs.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = RESULTS / "idea_sweep"
FIGURES = ROOT.parent / "ARR" / "IdeaSweep_figures"
RNG_SEED = 20260713
PAIR_SEEDS = {"02": (0,), "12": (1, 2)}
TASK_TYPES = ("clean", "cool", "examine", "heat", "put", "puttwo")

RUNS = {
    "gpt": {
        "none": "R008_nomem_s3",
        "expel": "R012_expel_bm25",
        "skillos": "R012_skillos_bm25",
        "none_01": "R013_nomem_seed5",
        "expel_01": "R013_expel_bm25_seed5",
        "placebo": ("PAPERO_PLACEBO_GPT", "PAPERO_PLACEBO_GPT_S2"),
    },
    "qwen": {
        "none": "Q_nomem_s3",
        "expel": "Q_expel_bm25_s3",
        "skillos": "Q_skillos_bm25_s3",
        "none_01": "Q_nomem_seed5",
        "expel_01": "Q_expel_bm25_seed5",
        "placebo": ("PAPERO_PLACEBO_QWEN", "PAPERO_PLACEBO_QWEN_S2"),
    },
}


def load_runs(run_ids: str | tuple[str, ...]) -> dict[tuple[int, int], dict]:
    if isinstance(run_ids, str):
        run_ids = (run_ids,)
    rows: dict[tuple[int, int], dict] = {}
    for run_id in run_ids:
        path = RESULTS / f"{run_id}.episodes.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["seed"]), int(row["task_id"]))
            if key in rows:
                raise ValueError(f"duplicate episode {key} in {run_ids}")
            if row.get("error"):
                raise ValueError(f"episode error in {run_id} at {key}: {row['error']}")
            rows[key] = row
    return rows


def load_executor(executor: str) -> dict:
    cfg = RUNS[executor]
    raw = {arm: load_runs(cfg[arm]) for arm in ("none", "expel", "skillos")}
    test = {arm: load_runs(cfg[f"{arm}_01"]) for arm in ("none", "expel")}
    task_ids = sorted({task_id for seed, task_id in raw["none"] if seed == 0})
    expected_train = {(seed, task_id) for seed in (0, 1, 2) for task_id in task_ids}
    expected_test = {(5, task_id) for task_id in task_ids}
    for arm, rows in raw.items():
        if set(rows) != expected_train:
            raise ValueError(f"{executor}/{arm}: train key mismatch")
    for arm, rows in test.items():
        if set(rows) != expected_test:
            raise ValueError(f"{executor}/{arm}: seed-5 key mismatch")

    meta = {
        task_id: {
            "task_type": raw["none"][(0, task_id)]["task_type"],
            "task_text": raw["none"][(0, task_id)]["task_text"],
        }
        for task_id in task_ids
    }
    train = {
        arm: {
            pair: {
                task_id: float(np.mean([rows[(seed, task_id)]["success"] for seed in seeds]))
                for task_id in task_ids
            }
            for pair, seeds in PAIR_SEEDS.items()
        }
        for arm, rows in raw.items()
    }
    heldout = {
        arm: {task_id: float(rows[(5, task_id)]["success"]) for task_id in task_ids}
        for arm, rows in test.items()
    }
    return {"task_ids": task_ids, "meta": meta, "train": train, "heldout": heldout}


def by_type(task_ids: Iterable[int], meta: dict[int, dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for task_id in task_ids:
        groups[meta[task_id]["task_type"]].append(task_id)
    if set(groups) != set(TASK_TYPES):
        raise ValueError(f"unexpected task types: {sorted(groups)}")
    return {task_type: sorted(ids) for task_type, ids in groups.items()}


def split_tasks(
    groups: dict[str, list[int]], per_type: int, rng: np.random.Generator
) -> tuple[list[int], list[int]]:
    calibration: list[int] = []
    evaluation: list[int] = []
    for task_type in TASK_TYPES:
        ids = np.asarray(groups[task_type], dtype=int)
        if per_type >= len(ids):
            raise ValueError(f"per_type={per_type} leaves no {task_type} evaluation tasks")
        chosen = set(rng.choice(ids, size=per_type, replace=False).tolist())
        calibration.extend(sorted(chosen))
        evaluation.extend(task_id for task_id in ids if task_id not in chosen)
    return calibration, evaluation


def mean_score(values: dict[int, float], task_ids: Iterable[int]) -> float:
    ids = list(task_ids)
    return float(np.mean([values[task_id] for task_id in ids]))


def pair_delta(
    train: dict, arm: str, reference: str, task_ids: Iterable[int], pair: str
) -> float:
    ids = list(task_ids)
    return float(np.mean([
        train[arm][pair][task_id] - train[reference][pair][task_id]
        for task_id in ids
    ]))


def pooled_delta(
    train: dict, arm: str, reference: str, task_ids: Iterable[int]
) -> float:
    return float(np.mean([
        pair_delta(train, arm, reference, task_ids, pair) for pair in PAIR_SEEDS
    ]))


def pooled_delta_over_pairs(
    train: dict, arm: str, reference: str, task_ids: Iterable[int], pairs: Iterable[str]
) -> float:
    return float(np.mean([
        pair_delta(train, arm, reference, task_ids, pair) for pair in pairs
    ]))


def type_actions(
    calibration: list[int], meta: dict[int, dict], decide: Callable[[list[int]], str]
) -> dict[str, str]:
    groups = by_type(calibration, meta)
    return {task_type: decide(ids) for task_type, ids in groups.items()}


def score_actions(
    actions: dict[str, str], task_ids: list[int], meta: dict[int, dict],
    values: dict[str, dict[int, float]], reference: str = "none",
) -> dict[str, float]:
    selected = [
        values[actions[meta[task_id]["task_type"]]][task_id]
        for task_id in task_ids
    ]
    baseline = [values[reference][task_id] for task_id in task_ids]
    memory_rate = np.mean([
        actions[meta[task_id]["task_type"]] != reference for task_id in task_ids
    ])
    return {
        "success": float(np.mean(selected)),
        "delta_vs_none": float(np.mean(np.asarray(selected) - np.asarray(baseline))),
        "memory_rate": float(memory_rate),
    }


def constant_actions(arm: str) -> dict[str, str]:
    return {task_type: arm for task_type in TASK_TYPES}


def add_result(
    rows: list[dict], experiment: str, executor: str, per_type: int,
    replicate: int, policy: str, metrics: dict[str, float],
) -> None:
    rows.append({
        "experiment": experiment,
        "executor": executor,
        "calibration_per_type": per_type,
        "calibration_total": per_type * len(TASK_TYPES),
        "replicate": replicate,
        "policy": policy,
        **metrics,
    })


def run_prompt_consensus(data: dict[str, dict], reps: int) -> list[dict]:
    rows: list[dict] = []
    for executor, bundle in data.items():
        train, heldout, meta = bundle["train"], bundle["heldout"], bundle["meta"]
        groups = by_type(bundle["task_ids"], meta)
        for per_type in (2, 4, 6, 8):
            rng = np.random.default_rng(RNG_SEED + 1000 * per_type + (executor == "qwen"))
            for replicate in range(reps):
                calibration, evaluation = split_tasks(groups, per_type, rng)
                global_mean = "expel" if pooled_delta(train, "expel", "none", calibration) > 0 else "none"
                global_consensus = "expel" if all(
                    pair_delta(train, "expel", "none", calibration, pair) > 0
                    for pair in PAIR_SEEDS
                ) else "none"

                def decide_mean(ids: list[int]) -> str:
                    return "expel" if pooled_delta(train, "expel", "none", ids) > 0 else "none"

                def decide_consensus(ids: list[int], margin: float = 0.0) -> str:
                    return "expel" if all(
                        pair_delta(train, "expel", "none", ids, pair) > margin
                        for pair in PAIR_SEEDS
                    ) else "none"

                policies = {
                    "always_none": constant_actions("none"),
                    "always_expel": constant_actions("expel"),
                    "global_mean_gate": constant_actions(global_mean),
                    "global_pair_consensus": constant_actions(global_consensus),
                    "type_mean_gate": type_actions(calibration, meta, decide_mean),
                    "type_pair_consensus": type_actions(calibration, meta, decide_consensus),
                    "type_consensus_margin05": type_actions(
                        calibration, meta, lambda ids: decide_consensus(ids, 0.05)
                    ),
                }
                for policy, actions in policies.items():
                    metrics = score_actions(actions, evaluation, meta, heldout)
                    add_result(rows, "unseen_prompt_pair", executor, per_type, replicate, policy, metrics)
                    for task_type in TASK_TYPES:
                        rows[-1][f"select_{task_type}"] = actions[task_type]
    return rows


def run_prompt_rotation(data: dict[str, dict], reps: int) -> list[dict]:
    """Rotate the held-out prompt pair instead of privileging seed-5/pair-01."""
    rows: list[dict] = []
    all_values = {}
    for executor, bundle in data.items():
        all_values[executor] = {
            arm: {
                "01": bundle["heldout"][arm],
                "02": bundle["train"][arm]["02"],
                "12": bundle["train"][arm]["12"],
            }
            for arm in ("none", "expel")
        }

    for executor, bundle in data.items():
        meta = bundle["meta"]
        groups = by_type(bundle["task_ids"], meta)
        other_executor = "qwen" if executor == "gpt" else "gpt"
        for heldout_pair in ("01", "02", "12"):
            calibration_pairs = tuple(pair for pair in ("01", "02", "12") if pair != heldout_pair)
            target_values = {
                arm: all_values[executor][arm][heldout_pair] for arm in ("none", "expel")
            }
            for per_type in (2, 4, 6, 8):
                offset = {"01": 1, "02": 2, "12": 3}[heldout_pair]
                rng = np.random.default_rng(
                    RNG_SEED + 5000 * per_type + 10 * offset + (executor == "qwen")
                )
                for replicate in range(reps):
                    calibration, evaluation = split_tasks(groups, per_type, rng)

                    def decide(
                        source: str, ids: list[int], consensus: bool, margin: float = 0.0
                    ) -> str:
                        source_values = all_values[source]
                        deltas = [
                            mean_score(source_values["expel"][pair], ids)
                            - mean_score(source_values["none"][pair], ids)
                            for pair in calibration_pairs
                        ]
                        admit = (
                            all(delta > margin for delta in deltas)
                            if consensus else np.mean(deltas) > margin
                        )
                        return "expel" if admit else "none"

                    own_global_mean = decide(executor, calibration, False)
                    own_global_consensus = decide(executor, calibration, True)
                    transferred_global_consensus = decide(other_executor, calibration, True)

                    def lcb_decide(source: str, z_value: float = 1.2816) -> str:
                        source_values = all_values[source]
                        lower_bounds = []
                        for pair in calibration_pairs:
                            deltas = np.asarray([
                                source_values["expel"][pair][task_id]
                                - source_values["none"][pair][task_id]
                                for task_id in calibration
                            ], dtype=float)
                            standard_error = deltas.std(ddof=1) / math.sqrt(len(deltas))
                            lower_bounds.append(float(deltas.mean() - z_value * standard_error))
                        return "expel" if all(bound > 0 for bound in lower_bounds) else "none"

                    policies = {
                        "always_none": constant_actions("none"),
                        "always_expel": constant_actions("expel"),
                        "own_global_mean_gate": constant_actions(own_global_mean),
                        "own_global_pair_consensus": constant_actions(own_global_consensus),
                        "own_global_consensus_margin05": constant_actions(
                            decide(executor, calibration, True, 0.05)
                        ),
                        "own_global_lcb90": constant_actions(lcb_decide(executor)),
                        "transferred_global_consensus": constant_actions(
                            transferred_global_consensus
                        ),
                        "own_type_mean_gate": type_actions(
                            calibration, meta, lambda ids: decide(executor, ids, False)
                        ),
                        "own_type_pair_consensus": type_actions(
                            calibration, meta, lambda ids: decide(executor, ids, True)
                        ),
                        "own_type_consensus_margin10": type_actions(
                            calibration, meta, lambda ids: decide(executor, ids, True, 0.10)
                        ),
                        "transferred_type_consensus": type_actions(
                            calibration, meta, lambda ids: decide(other_executor, ids, True)
                        ),
                    }
                    for policy, actions in policies.items():
                        metrics = score_actions(actions, evaluation, meta, target_values)
                        add_result(
                            rows, "prompt_pair_rotation", executor, per_type,
                            replicate, policy, metrics,
                        )
                        rows[-1]["heldout_pair"] = heldout_pair
                        rows[-1]["calibration_pairs"] = "+".join(calibration_pairs)
    return rows


def load_placebo_data(executor: str, bundle: dict) -> dict:
    frozen = json.loads((RESULTS / "placebo" / "frozen_tasks.json").read_text())
    task_ids = sorted(int(task_id) for task_id in frozen["task_ids"])
    placebo_rows = load_runs(RUNS[executor]["placebo"])
    expected = {(seed, task_id) for seed in (0, 1, 2, 5) for task_id in task_ids}
    if set(placebo_rows) != expected:
        raise ValueError(f"{executor}/placebo key mismatch")
    train = {
        arm: {
            pair: {task_id: bundle["train"][arm][pair][task_id] for task_id in task_ids}
            for pair in PAIR_SEEDS
        }
        for arm in ("none", "expel")
    }
    train["placebo"] = {
        pair: {
            task_id: float(np.mean([
                placebo_rows[(seed, task_id)]["success"] for seed in seeds
            ]))
            for task_id in task_ids
        }
        for pair, seeds in PAIR_SEEDS.items()
    }
    heldout = {
        arm: {task_id: bundle["heldout"][arm][task_id] for task_id in task_ids}
        for arm in ("none", "expel")
    }
    meta = {task_id: bundle["meta"][task_id] for task_id in task_ids}
    return {"task_ids": task_ids, "train": train, "heldout": heldout, "meta": meta}


def run_placebo_gate(data: dict[str, dict], reps: int) -> list[dict]:
    rows: list[dict] = []
    for executor, full_bundle in data.items():
        bundle = load_placebo_data(executor, full_bundle)
        train, heldout, meta = bundle["train"], bundle["heldout"], bundle["meta"]
        groups = by_type(bundle["task_ids"], meta)
        for per_type in (2, 4, 6):
            rng = np.random.default_rng(RNG_SEED + 2000 * per_type + (executor == "qwen"))
            for replicate in range(reps):
                calibration, evaluation = split_tasks(groups, per_type, rng)

                def mean_gate(ids: list[int]) -> str:
                    return "expel" if pooled_delta(train, "expel", "none", ids) > 0 else "none"

                def pair_consensus(ids: list[int]) -> str:
                    return "expel" if all(
                        pair_delta(train, "expel", "none", ids, pair) > 0
                        for pair in PAIR_SEEDS
                    ) else "none"

                def pooled_dual(ids: list[int]) -> str:
                    useful = pooled_delta(train, "expel", "none", ids) > 0
                    semantic = pooled_delta(train, "expel", "placebo", ids) > 0
                    return "expel" if useful and semantic else "none"

                def dual_consensus(ids: list[int], margin: float = 0.0) -> str:
                    admit = all(
                        pair_delta(train, "expel", reference, ids, pair) > margin
                        for pair in PAIR_SEEDS
                        for reference in ("none", "placebo")
                    )
                    return "expel" if admit else "none"

                policies = {
                    "always_none": constant_actions("none"),
                    "always_expel": constant_actions("expel"),
                    "type_mean_gate": type_actions(calibration, meta, mean_gate),
                    "type_pair_consensus": type_actions(calibration, meta, pair_consensus),
                    "type_pooled_dual_control": type_actions(calibration, meta, pooled_dual),
                    "type_dual_consensus": type_actions(calibration, meta, dual_consensus),
                    "type_dual_margin05": type_actions(
                        calibration, meta, lambda ids: dual_consensus(ids, 0.05)
                    ),
                }
                for policy, actions in policies.items():
                    metrics = score_actions(actions, evaluation, meta, heldout)
                    add_result(rows, "placebo_admission", executor, per_type, replicate, policy, metrics)
                    for task_type in TASK_TYPES:
                        rows[-1][f"select_{task_type}"] = actions[task_type]
    return rows


def average_pairs(train: dict, arm: str) -> dict[int, float]:
    return {
        task_id: float(np.mean([train[arm][pair][task_id] for pair in PAIR_SEEDS]))
        for task_id in train[arm]["02"]
    }


def run_representation_portfolio(data: dict[str, dict], reps: int) -> list[dict]:
    rows: list[dict] = []
    arms = ("none", "expel", "skillos")
    for executor, bundle in data.items():
        train, meta = bundle["train"], bundle["meta"]
        values = {arm: average_pairs(train, arm) for arm in arms}
        groups = by_type(bundle["task_ids"], meta)
        for per_type in (2, 4, 6, 8):
            rng = np.random.default_rng(RNG_SEED + 3000 * per_type + (executor == "qwen"))
            for replicate in range(reps):
                calibration, evaluation = split_tasks(groups, per_type, rng)

                def best_arm(ids: list[int]) -> str:
                    means = {arm: mean_score(values[arm], ids) for arm in arms}
                    return max(arms, key=lambda arm: (means[arm], arm == "none"))

                def consensus_arm(ids: list[int], margin: float = 0.0) -> str:
                    eligible = ["none"]
                    for arm in ("expel", "skillos"):
                        if all(pair_delta(train, arm, "none", ids, pair) > margin for pair in PAIR_SEEDS):
                            eligible.append(arm)
                    return max(eligible, key=lambda arm: (mean_score(values[arm], ids), arm == "none"))

                policies = {
                    "always_none": constant_actions("none"),
                    "always_expel": constant_actions("expel"),
                    "always_skillos": constant_actions("skillos"),
                    "global_best_calibration": constant_actions(best_arm(calibration)),
                    "type_best_calibration": type_actions(calibration, meta, best_arm),
                    "type_consensus_portfolio": type_actions(calibration, meta, consensus_arm),
                    "type_consensus_margin05": type_actions(
                        calibration, meta, lambda ids: consensus_arm(ids, 0.05)
                    ),
                }
                for policy, actions in policies.items():
                    metrics = score_actions(actions, evaluation, meta, values)
                    add_result(rows, "representation_portfolio", executor, per_type, replicate, policy, metrics)
                    for task_type in TASK_TYPES:
                        rows[-1][f"select_{task_type}"] = actions[task_type]
    return rows


STOPWORDS = {
    "a", "an", "and", "in", "is", "it", "of", "the", "to", "two", "your",
    "task", "put", "find", "look", "at", "then",
}


def goal_tokens(task_text: str) -> set[str]:
    goal = task_text.rsplit("Your task is to:", maxsplit=1)[-1]
    return {
        token for token in re.findall(r"[a-z]+", goal.lower())
        if token not in STOPWORDS
    }


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def run_failure_cache(data: dict[str, dict], reps: int, k: int = 5) -> list[dict]:
    rows: list[dict] = []
    for executor, bundle in data.items():
        train, heldout, meta = bundle["train"], bundle["heldout"], bundle["meta"]
        groups = by_type(bundle["task_ids"], meta)
        tokens = {task_id: goal_tokens(row["task_text"]) for task_id, row in meta.items()}
        similarity = {
            (left, right): jaccard(tokens[left], tokens[right])
            for left in bundle["task_ids"] for right in bundle["task_ids"] if left != right
        }
        neighbor_order = {
            task_id: sorted(
                (other for other in bundle["task_ids"] if other != task_id),
                key=lambda other: (-similarity[(task_id, other)], other),
            )
            for task_id in bundle["task_ids"]
        }
        for per_type in (2, 4, 6, 8):
            rng = np.random.default_rng(RNG_SEED + 4000 * per_type + (executor == "qwen"))
            for replicate in range(reps):
                calibration, evaluation = split_tasks(groups, per_type, rng)
                calibration_set = set(calibration)
                actions_mean: dict[int, str] = {}
                actions_consensus: dict[int, str] = {}
                for task_id in evaluation:
                    neighbors = [
                        other for other in neighbor_order[task_id] if other in calibration_set
                    ][:k]
                    actions_mean[task_id] = "expel" if pooled_delta(
                        train, "expel", "none", neighbors
                    ) > 0 else "none"
                    actions_consensus[task_id] = "expel" if all(
                        pair_delta(train, "expel", "none", neighbors, pair) > 0
                        for pair in PAIR_SEEDS
                    ) else "none"

                for policy, actions in {
                    f"knn{k}_mean_cache": actions_mean,
                    f"knn{k}_pair_consensus": actions_consensus,
                }.items():
                    selected = [heldout[actions[task_id]][task_id] for task_id in evaluation]
                    baseline = [heldout["none"][task_id] for task_id in evaluation]
                    metrics = {
                        "success": float(np.mean(selected)),
                        "delta_vs_none": float(np.mean(np.asarray(selected) - np.asarray(baseline))),
                        "memory_rate": float(np.mean([actions[t] == "expel" for t in evaluation])),
                    }
                    add_result(rows, "failure_neighbor_cache", executor, per_type, replicate, policy, metrics)
    return rows


def run_webshop_gate(reps: int) -> list[dict]:
    """Task-disjoint replay of a global admission gate in the second environment."""
    baseline_rows = load_runs("PAPERO_WS_Q_nomem_eval100")
    memory_rows = load_runs("PAPERO_WS_Q_expel_bm25_eval100")
    if set(baseline_rows) != set(memory_rows):
        raise ValueError("WebShop paired task keys differ")
    task_ids = sorted(task_id for seed, task_id in baseline_rows if seed == 0)
    if len(task_ids) != 100:
        raise ValueError(f"expected 100 WebShop tasks, found {len(task_ids)}")
    reward = {
        "none": {task_id: float(baseline_rows[(0, task_id)]["reward"]) for task_id in task_ids},
        "expel": {task_id: float(memory_rows[(0, task_id)]["reward"]) for task_id in task_ids},
    }
    exact = {
        "none": {task_id: float(baseline_rows[(0, task_id)]["success"]) for task_id in task_ids},
        "expel": {task_id: float(memory_rows[(0, task_id)]["success"]) for task_id in task_ids},
    }
    rows: list[dict] = []
    for calibration_size in (10, 20, 30, 40):
        rng = np.random.default_rng(RNG_SEED + 6000 * calibration_size)
        for replicate in range(reps):
            calibration = sorted(rng.choice(task_ids, size=calibration_size, replace=False).tolist())
            calibration_set = set(calibration)
            evaluation = [task_id for task_id in task_ids if task_id not in calibration_set]
            deltas = np.asarray([
                reward["expel"][task_id] - reward["none"][task_id]
                for task_id in calibration
            ], dtype=float)
            standard_error = deltas.std(ddof=1) / math.sqrt(len(deltas))
            actions = {
                "always_none": "none",
                "always_expel": "expel",
                "global_mean_gate": "expel" if deltas.mean() > 0 else "none",
                "global_lcb90": (
                    "expel" if deltas.mean() - 1.2816 * standard_error > 0 else "none"
                ),
            }
            for policy, arm in actions.items():
                selected_reward = np.asarray([reward[arm][task_id] for task_id in evaluation])
                baseline_reward = np.asarray([reward["none"][task_id] for task_id in evaluation])
                selected_exact = np.asarray([exact[arm][task_id] for task_id in evaluation])
                baseline_exact = np.asarray([exact["none"][task_id] for task_id in evaluation])
                rows.append({
                    "experiment": "webshop_global_admission",
                    "executor": "qwen",
                    "calibration_per_type": calibration_size,
                    "calibration_total": calibration_size,
                    "replicate": replicate,
                    "policy": policy,
                    "success": float(selected_reward.mean()),
                    "delta_vs_none": float((selected_reward - baseline_reward).mean()),
                    "memory_rate": float(arm == "expel"),
                    "exact_success": float(selected_exact.mean()),
                    "delta_exact_vs_none": float((selected_exact - baseline_exact).mean()),
                })
    return rows


def simulate_retirement(sims: int) -> list[dict]:
    """Sanity-check sequential retirement under a stylized Bernoulli drift."""
    rng = np.random.default_rng(RNG_SEED + 5000)
    horizon, change_at, interval = 300, 150, 5
    p_good, p_bad, p_none = 0.75, 0.45, 0.60
    check_times = np.arange(interval, horizon + 1, interval)
    # A raw log(1/alpha) threshold is not alpha-valid after repeated CUSUM resets.
    # We therefore use a fixed evidence threshold and report its empirical
    # horizon-specific false-alarm rate rather than assigning it a p-value.
    threshold = 5.0
    output: list[dict] = []

    for no_drift in (True, False):
        for simulation in range(sims):
            p = np.full(len(check_times), p_good)
            if not no_drift:
                p[check_times > change_at] = p_bad
            observations = rng.binomial(1, p)
            methods: dict[str, int | None] = {"never_retire": None}

            rolling_alarm = None
            for index in range(19, len(observations)):
                if observations[index - 19:index + 1].mean() < p_none:
                    rolling_alarm = int(check_times[index])
                    break
            methods["rolling20"] = rolling_alarm

            cusum, cusum_alarm = 0.0, None
            for index, outcome in enumerate(observations):
                llr = (
                    outcome * math.log(p_bad / p_good)
                    + (1 - outcome) * math.log((1 - p_bad) / (1 - p_good))
                )
                cusum = max(0.0, cusum + llr)
                if cusum >= threshold:
                    cusum_alarm = int(check_times[index])
                    break
            methods["cusum_h5"] = cusum_alarm

            for method, alarm in methods.items():
                active = np.ones(horizon, dtype=bool)
                if alarm is not None:
                    active[alarm:] = False
                skill_prob = np.full(horizon, p_good)
                if not no_drift:
                    skill_prob[change_at:] = p_bad
                achieved = np.where(active, skill_prob, p_none).sum()
                oracle = np.maximum(skill_prob, p_none).sum()
                output.append({
                    "experiment": "drift_retirement",
                    "scenario": "stationary" if no_drift else "abrupt_drift",
                    "simulation": simulation,
                    "policy": method,
                    "alarm": alarm,
                    "false_alarm": bool(no_drift and alarm is not None),
                    "detection_delay": (
                        max(0, alarm - change_at) if (not no_drift and alarm is not None) else None
                    ),
                    "expected_regret": float(oracle - achieved),
                })
    return output


def summarize(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summary: list[dict] = []
    for group_key, group in sorted(grouped.items()):
        record = dict(zip(keys, group_key))
        for metric in ("success", "delta_vs_none", "memory_rate"):
            if metric not in group[0]:
                continue
            values = np.asarray([row[metric] for row in group], dtype=float)
            record[f"mean_{metric}"] = float(values.mean())
            record[f"split_p025_{metric}"] = float(np.quantile(values, 0.025))
            record[f"split_p975_{metric}"] = float(np.quantile(values, 0.975))
        if "delta_vs_none" in group[0]:
            record["harmful_split_rate"] = float(np.mean([
                row["delta_vs_none"] < 0 for row in group
            ]))
        summary.append(record)
    return summary


def summarize_drift(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["policy"])].append(row)
    output = []
    for (scenario, policy), group in sorted(grouped.items()):
        alarms = [row["alarm"] for row in group if row["alarm"] is not None]
        delays = [row["detection_delay"] for row in group if row["detection_delay"] is not None]
        output.append({
            "scenario": scenario,
            "policy": policy,
            "alarm_rate": len(alarms) / len(group),
            "false_alarm_rate": float(np.mean([row["false_alarm"] for row in group])),
            "mean_detection_delay": float(np.mean(delays)) if delays else None,
            "mean_expected_regret": float(np.mean([row["expected_regret"] for row in group])),
        })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def lookup(summary: list[dict], experiment: str, executor: str, per_type: int, policy: str) -> dict:
    for row in summary:
        if (
            row["experiment"] == experiment
            and row["executor"] == executor
            and row["calibration_per_type"] == per_type
            and row["policy"] == policy
        ):
            return row
    raise KeyError((experiment, executor, per_type, policy))


def plot_summary(summary: list[dict], drift: list[dict]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    })
    colors = {"gpt": "#C8523B", "qwen": "#167D8D"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

    panels = [
        (axes[0, 0], "prompt_pair_rotation", "own_global_lcb90", 4,
         "A  Prompt-consensus abstention", "Unseen-pair success delta"),
        (axes[0, 1], "placebo_admission", "type_dual_consensus", 4,
         "B  Dual-control admission", "Unseen-pair success delta"),
        (axes[1, 0], "representation_portfolio", "type_consensus_portfolio", 4,
         "C  Representation portfolio", "Held-out-task success delta"),
    ]
    for axis, experiment, policy, per_type, title, ylabel in panels:
        for index, executor in enumerate(("gpt", "qwen")):
            row = lookup(summary, experiment, executor, per_type, policy)
            y = row["mean_delta_vs_none"]
            lo = row["split_p025_delta_vs_none"]
            hi = row["split_p975_delta_vs_none"]
            axis.errorbar(
                index, y, yerr=[[max(0.0, y - lo)], [max(0.0, hi - y)]],
                fmt="o", markersize=7,
                color=colors[executor], capsize=4, linewidth=1.6,
            )
        axis.axhline(0, color="#555555", linewidth=0.9, linestyle="--")
        axis.set_xticks((0, 1), ("GPT-4o-mini", "Qwen3-8B"))
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left")
        axis.grid(axis="y", color="#D8D2C4", alpha=0.65, linewidth=0.7)

    axis = axes[1, 1]
    drift_rows = {
        (row["scenario"], row["policy"]): row for row in drift
    }
    methods = ("never_retire", "rolling20", "cusum_h5")
    labels = ("Never", "Rolling-20", "CUSUM")
    regrets = [drift_rows[("abrupt_drift", method)]["mean_expected_regret"] for method in methods]
    false_alarms = [drift_rows[("stationary", method)]["false_alarm_rate"] for method in methods]
    x = np.arange(len(methods))
    bars = axis.bar(x, regrets, color=("#B9B1A1", "#E6A33D", "#167D8D"), width=0.62)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Expected regret (300 episodes)")
    axis.set_title("D  Drift-retirement sanity check", loc="left")
    axis.grid(axis="y", color="#D8D2C4", alpha=0.65, linewidth=0.7)
    for bar, rate in zip(bars, false_alarms):
        axis.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
            f"FA {rate:.1%}", ha="center", va="bottom", fontsize=8,
        )

    fig.suptitle(
        "Training-free self-evolution: retrospective feasibility gates",
        fontsize=14, fontweight="bold", x=0.02, ha="left",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig1_training_free_idea_gates.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig1_training_free_idea_gates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--drift-sims", type=int, default=10000)
    args = parser.parse_args()
    if args.reps < 1 or args.drift_sims < 1:
        raise ValueError("simulation counts must be positive")

    data = {executor: load_executor(executor) for executor in ("gpt", "qwen")}
    replay_rows = []
    replay_rows.extend(run_prompt_consensus(data, args.reps))
    replay_rows.extend(run_placebo_gate(data, args.reps))
    replay_rows.extend(run_representation_portfolio(data, args.reps))
    replay_rows.extend(run_failure_cache(data, args.reps))
    replay_rows.extend(run_webshop_gate(args.reps))
    rotation_rows = run_prompt_rotation(data, args.reps)
    drift_rows = simulate_retirement(args.drift_sims)

    summary = summarize(
        replay_rows,
        ("experiment", "executor", "calibration_per_type", "calibration_total", "policy"),
    )
    rotation_summary = summarize(
        rotation_rows,
        (
            "experiment", "executor", "heldout_pair", "calibration_per_type",
            "calibration_total", "policy",
        ),
    )
    rotation_overall_summary = summarize(
        rotation_rows,
        ("experiment", "executor", "calibration_per_type", "calibration_total", "policy"),
    )
    summary.extend(rotation_overall_summary)
    drift_summary = summarize_drift(drift_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "replay_trials.csv", replay_rows)
    write_csv(OUT / "replay_summary.csv", summary)
    write_csv(OUT / "prompt_rotation_trials.csv", rotation_rows)
    write_csv(OUT / "prompt_rotation_summary.csv", rotation_summary)
    write_csv(OUT / "drift_trials.csv", drift_rows)
    write_csv(OUT / "drift_summary.csv", drift_summary)
    result = {
        "protocol": {
            "rng_seed": RNG_SEED,
            "replay_repetitions": args.reps,
            "drift_simulations": args.drift_sims,
            "calibration_prompt_pairs": list(PAIR_SEEDS),
            "unseen_evaluation_prompt_pair": "01 (seed 5)",
            "task_split": "stratified by six ALFWorld task types; disjoint calibration/evaluation",
            "interval_warning": "p2.5/p97.5 quantify random split sensitivity, not sampling CIs",
        },
        "replay_summary": summary,
        "prompt_rotation_summary": rotation_summary,
        "drift_summary": drift_summary,
    }
    (OUT / "idea_sweep_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_summary(summary, drift_summary)
    print(json.dumps({
        "output": str(OUT),
        "figure": str(FIGURES / "fig1_training_free_idea_gates.pdf"),
        "n_replay_rows": len(replay_rows) + len(rotation_rows),
        "n_drift_rows": len(drift_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
