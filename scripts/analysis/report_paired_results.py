#!/usr/bin/env python3
"""Summarize paired episode-level deltas between a reference run and comparators.

Usage:
  python scripts/report_paired_results.py \
      --reference results/R008_nomem_s3.episodes.jsonl \
      --compare results/R012_expel_bm25.episodes.jsonl \
      --compare results/R012_raw_dense.episodes.jsonl

Pairs are matched by (seed, task_id), so this script is intended for runs that
share the same task split and seed list.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import pstdev


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_rows(rows):
    return {(row["seed"], row["task_id"]): row for row in rows}


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def mean_ci95(vals):
    mu = mean(vals)
    if len(vals) <= 1:
        return mu, (mu, mu)
    se = pstdev(vals) / math.sqrt(len(vals))
    return mu, (mu - 1.96 * se, mu + 1.96 * se)


def compare(reference_path: Path, compare_path: Path):
    ref = index_rows(load_jsonl(reference_path))
    cmp_ = index_rows(load_jsonl(compare_path))
    keys = sorted(set(ref) & set(cmp_))
    if not keys:
        raise ValueError(f"no paired rows between {reference_path} and {compare_path}")

    missing_ref = len(set(cmp_) - set(ref))
    missing_cmp = len(set(ref) - set(cmp_))

    delta_success = []
    delta_reward = []
    delta_steps = []
    delta_prompt_tokens = []
    memory_tokens = []
    wins = losses = ties = 0
    by_task_type = defaultdict(lambda: {
        "delta_success": [],
        "delta_reward": [],
        "delta_steps": [],
        "delta_prompt_tokens": [],
        "memory_tokens": [],
        "wins": 0,
        "losses": 0,
        "ties": 0,
    })

    for key in keys:
        a = cmp_[key]
        b = ref[key]
        task_type = a.get("task_type", "unknown")

        ds = int(a["success"]) - int(b["success"])
        delta_success.append(ds)
        delta_reward.append(a["reward"] - b["reward"])
        delta_steps.append(a["steps"] - b["steps"])
        delta_prompt_tokens.append(a["prompt_tokens"] - b["prompt_tokens"])
        memory_tokens.append(a.get("memory_tokens", 0))
        by_task_type[task_type]["delta_success"].append(ds)
        by_task_type[task_type]["delta_reward"].append(a["reward"] - b["reward"])
        by_task_type[task_type]["delta_steps"].append(a["steps"] - b["steps"])
        by_task_type[task_type]["delta_prompt_tokens"].append(
            a["prompt_tokens"] - b["prompt_tokens"]
        )
        by_task_type[task_type]["memory_tokens"].append(a.get("memory_tokens", 0))

        if ds > 0:
            wins += 1
            by_task_type[task_type]["wins"] += 1
        elif ds < 0:
            losses += 1
            by_task_type[task_type]["losses"] += 1
        else:
            ties += 1
            by_task_type[task_type]["ties"] += 1

    by_type_report = {}
    for task_type, stats in sorted(by_task_type.items()):
        succ_mu, succ_ci = mean_ci95(stats["delta_success"])
        rew_mu, rew_ci = mean_ci95(stats["delta_reward"])
        step_mu, step_ci = mean_ci95(stats["delta_steps"])
        tok_mu, tok_ci = mean_ci95(stats["delta_prompt_tokens"])
        mem_mu, mem_ci = mean_ci95(stats["memory_tokens"])
        by_type_report[task_type] = {
            "paired_n": len(stats["delta_success"]),
            "delta_success": {"mean": succ_mu, "ci95": succ_ci},
            "delta_reward": {"mean": rew_mu, "ci95": rew_ci},
            "delta_steps": {"mean": step_mu, "ci95": step_ci},
            "delta_prompt_tokens": {"mean": tok_mu, "ci95": tok_ci},
            "memory_tokens": {"mean": mem_mu, "ci95": mem_ci},
            "wins_losses_ties": {
                "wins": stats["wins"],
                "losses": stats["losses"],
                "ties": stats["ties"],
            },
        }

    succ_mu, succ_ci = mean_ci95(delta_success)
    rew_mu, rew_ci = mean_ci95(delta_reward)
    step_mu, step_ci = mean_ci95(delta_steps)
    tok_mu, tok_ci = mean_ci95(delta_prompt_tokens)
    mem_mu, mem_ci = mean_ci95(memory_tokens)

    return {
        "reference": str(reference_path),
        "compare": str(compare_path),
        "paired_n": len(keys),
        "missing_only_in_reference": missing_cmp,
        "missing_only_in_compare": missing_ref,
        "delta_success": {"mean": succ_mu, "ci95": succ_ci},
        "delta_reward": {"mean": rew_mu, "ci95": rew_ci},
        "delta_steps": {"mean": step_mu, "ci95": step_ci},
        "delta_prompt_tokens": {"mean": tok_mu, "ci95": tok_ci},
        "memory_tokens": {"mean": mem_mu, "ci95": mem_ci},
        "wins_losses_ties": {"wins": wins, "losses": losses, "ties": ties},
        "by_task_type": by_type_report,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--compare", required=True, type=Path, action="append")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument(
        "--by-task-type",
        action="store_true",
        help="print per-task-type paired summaries",
    )
    args = ap.parse_args()

    reports = [compare(args.reference, path) for path in args.compare]
    if args.json:
        print(json.dumps(reports, indent=2, ensure_ascii=False))
        return

    print(f"reference: {args.reference}")
    for rep in reports:
        name = Path(rep["compare"]).name
        ds = rep["delta_success"]
        dr = rep["delta_reward"]
        st = rep["delta_steps"]
        pt = rep["delta_prompt_tokens"]
        mt = rep["memory_tokens"]
        wlt = rep["wins_losses_ties"]
        print()
        print(name)
        print(f"  paired_n: {rep['paired_n']}")
        print(f"  delta_success: {ds['mean']:+.4f} [{ds['ci95'][0]:+.4f}, {ds['ci95'][1]:+.4f}]")
        print(f"  delta_reward: {dr['mean']:+.4f} [{dr['ci95'][0]:+.4f}, {dr['ci95'][1]:+.4f}]")
        print(f"  delta_steps: {st['mean']:+.2f} [{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}]")
        print(
            "  delta_prompt_tokens: "
            f"{pt['mean']:+.1f} [{pt['ci95'][0]:+.1f}, {pt['ci95'][1]:+.1f}]"
        )
        print(f"  memory_tokens: {mt['mean']:.1f} [{mt['ci95'][0]:.1f}, {mt['ci95'][1]:.1f}]")
        print(f"  wins/losses/ties: {wlt['wins']}/{wlt['losses']}/{wlt['ties']}")
        if args.by_task_type:
            print("  by_task_type:")
            for task_type, stats in rep["by_task_type"].items():
                ds = stats["delta_success"]
                st = stats["delta_steps"]
                tok = stats["delta_prompt_tokens"]
                wlt_tt = stats["wins_losses_ties"]
                print(
                    "   "
                    f"{task_type}: n={stats['paired_n']} "
                    f"Δsucc={ds['mean']:+.4f} [{ds['ci95'][0]:+.4f}, {ds['ci95'][1]:+.4f}] "
                    f"Δsteps={st['mean']:+.2f} [{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}] "
                    f"Δptok={tok['mean']:+.1f} [{tok['ci95'][0]:+.1f}, {tok['ci95'][1]:+.1f}] "
                    f"w/l/t={wlt_tt['wins']}/{wlt_tt['losses']}/{wlt_tt['ties']}"
                )


if __name__ == "__main__":
    main()
