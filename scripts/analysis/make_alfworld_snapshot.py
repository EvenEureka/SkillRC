#!/usr/bin/env python3
"""Emit a markdown snapshot of the current ALFWorld result set.

This is intended for rapid paper planning / mentor updates. It reads the current
summary + episode files under `results/`, computes paired comparisons, and prints
one markdown report to stdout.
"""
from __future__ import annotations

import json
from pathlib import Path

from report_paired_results import compare


RESULTS = {
    "nomem": "R008_nomem_s3",
    "expel_bm25": "R012_expel_bm25",
    "expel_dense": "R012_expel_dense",
    "raw_bm25": "R012_raw_bm25",
    "raw_dense": "R012_raw_dense",
    "skillos_bm25": "R012_skillos_bm25",
    "skillos_dense": "R012_skillos_dense",
}

ORDER = [
    "nomem",
    "expel_bm25",
    "raw_dense",
    "skillos_dense",
    "raw_bm25",
    "expel_dense",
    "skillos_bm25",
]

PAIRWISE = [
    ("expel_bm25", "expel_dense", "Retriever effect within ExpeL-style insights"),
    ("raw_dense", "raw_bm25", "Retriever effect within raw trajectories"),
    ("skillos_dense", "skillos_bm25", "Retriever effect within Markdown skills"),
    ("expel_bm25", "raw_bm25", "Representation effect under BM25"),
    ("expel_dense", "raw_dense", "Representation effect under dense retrieval"),
    ("expel_bm25", "skillos_bm25", "Insight vs Markdown under BM25"),
    ("expel_dense", "skillos_dense", "Insight vs Markdown under dense retrieval"),
]


def load_summary(results_dir: Path, run_id: str):
    with (results_dir / f"{run_id}.summary.json").open() as f:
        return json.load(f)["summary"]


def mean_memory_tokens(results_dir: Path, run_id: str) -> float:
    vals = []
    with (results_dir / f"{run_id}.episodes.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            vals.append(row.get("memory_tokens", 0))
    return sum(vals) / len(vals) if vals else 0.0


def mean_ci_str(stat: dict, digits: int = 4) -> str:
    mean = stat["mean"]
    lo, hi = stat["ci95"]
    fmt = f"{{:+.{digits}f}} [{{:+.{digits}f}}, {{:+.{digits}f}}]"
    return fmt.format(mean, lo, hi)


def main():
    here = Path(__file__).resolve()
    root = here.parent.parent
    results_dir = root / "results"
    summaries = {k: load_summary(results_dir, v) for k, v in RESULTS.items()}

    print("# ALFWorld Snapshot")
    print()
    print("Current frozen-executor ALFWorld snapshot from `skilllib_rc/results/`.")
    print()

    print("## Main Table")
    print()
    print("| Variant | success | steps | prompt tokens | total cost (USD) | seed std | memory kind | realized memory tokens |")
    print("|---|---:|---:|---:|---:|---:|---|---:|")
    for key in ORDER:
        s = summaries[key]
        vals = list(s.get("by_seed", {}).values())
        if vals:
            seed_mean = sum(vals) / len(vals)
            seed_var = sum((x - seed_mean) ** 2 for x in vals) / len(vals)
            seed_std = seed_var ** 0.5
        else:
            seed_std = 0.0
        mem = s.get("memory", {})
        realized_mem = mean_memory_tokens(results_dir, RESULTS[key])
        print(
            f"| `{RESULTS[key]}` | {s['success_rate']:.4f} | {s['steps']['mean']:.2f} | "
            f"{s['prompt_tokens']['mean']:.1f} | {s['total_cost_usd']:.4f} | "
            f"{seed_std:.4f} | {mem.get('kind', 'n/a')} | {realized_mem:.1f} |"
        )
    print()

    print("## Paired vs No-Memory")
    print()
    ref = results_dir / f"{RESULTS['nomem']}.episodes.jsonl"
    print("| Variant | Δsuccess | Δsteps | Δprompt tokens | wins/losses/ties |")
    print("|---|---:|---:|---:|---:|")
    for key in ORDER[1:]:
        rep = compare(ref, results_dir / f"{RESULTS[key]}.episodes.jsonl")
        ds = rep["delta_success"]
        st = rep["delta_steps"]
        pt = rep["delta_prompt_tokens"]
        wlt = rep["wins_losses_ties"]
        print(
            f"| `{RESULTS[key]}` | {mean_ci_str(ds)} | {mean_ci_str(st, 2)} | "
            f"{mean_ci_str(pt, 1)} | {wlt['wins']}/{wlt['losses']}/{wlt['ties']} |"
        )
    print()

    print("## Pairwise Interaction Checks")
    print()
    print("| Comparison | Interpretation | Δsuccess | Δsteps | Δprompt tokens |")
    print("|---|---|---:|---:|---:|")
    for left, right, label in PAIRWISE:
        rep = compare(
            results_dir / f"{RESULTS[right]}.episodes.jsonl",
            results_dir / f"{RESULTS[left]}.episodes.jsonl",
        )
        print(
            f"| `{RESULTS[left]}` vs `{RESULTS[right]}` | {label} | "
            f"{mean_ci_str(rep['delta_success'])} | {mean_ci_str(rep['delta_steps'], 2)} | "
            f"{mean_ci_str(rep['delta_prompt_tokens'], 1)} |"
        )
    print()

    print("## Task-Type Utility vs No-Memory")
    print()
    best = compare(ref, results_dir / f"{RESULTS['expel_bm25']}.episodes.jsonl")
    print("| Task type | Δsuccess (`expel_bm25` - `nomem`) | Δsteps | wins/losses/ties |")
    print("|---|---:|---:|---:|")
    for task_type, stats in best["by_task_type"].items():
        wlt = stats["wins_losses_ties"]
        print(
            f"| `{task_type}` | {mean_ci_str(stats['delta_success'])} | "
            f"{mean_ci_str(stats['delta_steps'], 2)} | "
            f"{wlt['wins']}/{wlt['losses']}/{wlt['ties']} |"
        )
    print()

    print("## Caveats")
    print()
    print("- `expel_insights.jsonl` is much more compact than the raw and Markdown pools, so the current comparisons jointly reflect representation *and* realized memory footprint.")
    print("- The nominal injected-memory budget is `2048`, but realized memory tokens differ substantially by variant because some pools are too small to fill that budget.")
    print("- This means the current results support a strong *compactness + utility* story, but not yet a strict equal-budget representation-only claim.")
    print("- WebShop is not yet a reliable second environment in this workspace, so the current evidence is ALFWorld-only.")


if __name__ == "__main__":
    main()
