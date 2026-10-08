#!/usr/bin/env python3
"""Export Paper O table/figure assets as CSV files.

This script turns the current ALFWorld result set into machine-readable assets
for manuscript tables and plots. It is intentionally narrow in scope: Paper O's
current evidence base is the frozen-executor ALFWorld study plus the matched-
footprint pilots.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from report_paired_results import compare, load_jsonl


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
    ("expel_bm25", "expel_dense", "Retriever effect within insight memories"),
    ("raw_dense", "raw_bm25", "Retriever effect within raw trajectories"),
    ("skillos_dense", "skillos_bm25", "Retriever effect within Markdown skills"),
    ("expel_bm25", "raw_bm25", "Representation effect under BM25"),
    ("expel_dense", "raw_dense", "Representation effect under dense retrieval"),
    ("expel_bm25", "skillos_bm25", "Insight vs Markdown under BM25"),
    ("expel_dense", "skillos_dense", "Insight vs Markdown under dense retrieval"),
]

PILOT_RUNS = [
    "PILOT_eqfp_nomem_s0_types6",
    "PILOT_eqfp_expel_bm25_s0_types6",
    "PILOT_eqfp_raw_bm25_s0_types6",
    "PILOT_eqfp_skillos_bm25_s0_types6",
    "PILOT_eqfp_nomem_s0_types12",
    "PILOT_eqfp_expel_bm25_s0_types12",
    "PILOT_eqfp_raw_bm25_s0_types12",
    "PILOT_eqfp_skillos_bm25_s0_types12",
]

EXECUTOR_TRANSFER = {
    "gpt-4o-mini": {
        "baseline": "R008_nomem_s3",
        "variants": [
            ("expel", "dense", "R012_expel_dense"),
            ("skillos", "dense", "R012_skillos_dense"),
            ("expel", "bm25", "R012_expel_bm25"),
            ("skillos", "bm25", "R012_skillos_bm25"),
        ],
    },
    "qwen3-8b": {
        "baseline": "Q_nomem_s3",
        "variants": [
            ("expel", "dense", "Q_expel_s3"),
            ("skillos", "dense", "Q_skillos_s3"),
            ("expel", "bm25", "Q_expel_bm25_s3"),
            ("skillos", "bm25", "Q_skillos_bm25_s3"),
        ],
    },
}


def load_summary(results_dir: Path, run_id: str):
    return json.loads((results_dir / f"{run_id}.summary.json").read_text())["summary"]


def mean_memory_tokens(results_dir: Path, run_id: str) -> float:
    rows = load_jsonl(results_dir / f"{run_id}.episodes.jsonl")
    vals = [row.get("memory_tokens", 0.0) for row in rows]
    return sum(vals) / len(vals) if vals else 0.0


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_main_table(results_dir: Path, out_dir: Path):
    rows = []
    ref = results_dir / f"{RESULTS['nomem']}.episodes.jsonl"
    for key in ORDER:
        run_id = RESULTS[key]
        s = load_summary(results_dir, run_id)
        row = {
            "variant_key": key,
            "run_id": run_id,
            "success_rate": f"{s['success_rate']:.4f}",
            "mean_steps": f"{s['steps']['mean']:.2f}",
            "mean_prompt_tokens": f"{s['prompt_tokens']['mean']:.1f}",
            "total_cost_usd": f"{s['total_cost_usd']:.4f}",
            "memory_kind": s["memory"]["kind"],
            "realized_memory_tokens": f"{mean_memory_tokens(results_dir, run_id):.1f}",
        }
        if key == "nomem":
            row.update(
                {
                    "paired_delta_success_mean": "",
                    "paired_delta_success_ci95_lo": "",
                    "paired_delta_success_ci95_hi": "",
                }
            )
        else:
            rep = compare(ref, results_dir / f"{run_id}.episodes.jsonl")
            row.update(
                {
                    "paired_delta_success_mean": f"{rep['delta_success']['mean']:+.4f}",
                    "paired_delta_success_ci95_lo": f"{rep['delta_success']['ci95'][0]:+.4f}",
                    "paired_delta_success_ci95_hi": f"{rep['delta_success']['ci95'][1]:+.4f}",
                }
            )
        rows.append(row)
    write_csv(
        out_dir / "table1_main_results.csv",
        [
            "variant_key",
            "run_id",
            "success_rate",
            "paired_delta_success_mean",
            "paired_delta_success_ci95_lo",
            "paired_delta_success_ci95_hi",
            "mean_steps",
            "mean_prompt_tokens",
            "total_cost_usd",
            "memory_kind",
            "realized_memory_tokens",
        ],
        rows,
    )


def export_paired_vs_nomem(results_dir: Path, out_dir: Path):
    ref = results_dir / f"{RESULTS['nomem']}.episodes.jsonl"
    rows = []
    for key in ORDER[1:]:
        run_id = RESULTS[key]
        rep = compare(ref, results_dir / f"{run_id}.episodes.jsonl")
        rows.append(
            {
                "variant_key": key,
                "run_id": run_id,
                "paired_n": rep["paired_n"],
                "delta_success_mean": f"{rep['delta_success']['mean']:+.4f}",
                "delta_success_ci95_lo": f"{rep['delta_success']['ci95'][0]:+.4f}",
                "delta_success_ci95_hi": f"{rep['delta_success']['ci95'][1]:+.4f}",
                "delta_steps_mean": f"{rep['delta_steps']['mean']:+.2f}",
                "delta_steps_ci95_lo": f"{rep['delta_steps']['ci95'][0]:+.2f}",
                "delta_steps_ci95_hi": f"{rep['delta_steps']['ci95'][1]:+.2f}",
                "delta_prompt_tokens_mean": f"{rep['delta_prompt_tokens']['mean']:+.1f}",
                "delta_prompt_tokens_ci95_lo": f"{rep['delta_prompt_tokens']['ci95'][0]:+.1f}",
                "delta_prompt_tokens_ci95_hi": f"{rep['delta_prompt_tokens']['ci95'][1]:+.1f}",
                "wins": rep["wins_losses_ties"]["wins"],
                "losses": rep["wins_losses_ties"]["losses"],
                "ties": rep["wins_losses_ties"]["ties"],
            }
        )
    write_csv(
        out_dir / "figure1_paired_vs_nomem.csv",
        list(rows[0].keys()),
        rows,
    )


def export_tasktype(results_dir: Path, out_dir: Path):
    ref = results_dir / f"{RESULTS['nomem']}.episodes.jsonl"
    rep = compare(ref, results_dir / f"{RESULTS['expel_bm25']}.episodes.jsonl")
    baseline = load_summary(results_dir, RESULTS["nomem"])["by_task_type"]
    rows = []
    for task_type, stats in rep["by_task_type"].items():
        rows.append(
            {
                "task_type": task_type,
                "baseline_success_rate": f"{baseline[task_type]['success_rate']:.4f}",
                "delta_success_mean": f"{stats['delta_success']['mean']:+.4f}",
                "delta_success_ci95_lo": f"{stats['delta_success']['ci95'][0]:+.4f}",
                "delta_success_ci95_hi": f"{stats['delta_success']['ci95'][1]:+.4f}",
                "delta_steps_mean": f"{stats['delta_steps']['mean']:+.2f}",
                "delta_steps_ci95_lo": f"{stats['delta_steps']['ci95'][0]:+.2f}",
                "delta_steps_ci95_hi": f"{stats['delta_steps']['ci95'][1]:+.2f}",
                "wins": stats["wins_losses_ties"]["wins"],
                "losses": stats["wins_losses_ties"]["losses"],
                "ties": stats["wins_losses_ties"]["ties"],
            }
        )
    rows.sort(key=lambda r: r["task_type"])
    write_csv(out_dir / "figure2_tasktype_best_vs_nomem.csv", list(rows[0].keys()), rows)


def export_pairwise(results_dir: Path, out_dir: Path):
    rows = []
    for left, right, label in PAIRWISE:
        rep = compare(
            results_dir / f"{RESULTS[right]}.episodes.jsonl",
            results_dir / f"{RESULTS[left]}.episodes.jsonl",
        )
        rows.append(
            {
                "left_variant_key": left,
                "left_run_id": RESULTS[left],
                "right_variant_key": right,
                "right_run_id": RESULTS[right],
                "interpretation": label,
                "delta_success_mean": f"{rep['delta_success']['mean']:+.4f}",
                "delta_success_ci95_lo": f"{rep['delta_success']['ci95'][0]:+.4f}",
                "delta_success_ci95_hi": f"{rep['delta_success']['ci95'][1]:+.4f}",
                "delta_steps_mean": f"{rep['delta_steps']['mean']:+.2f}",
                "delta_steps_ci95_lo": f"{rep['delta_steps']['ci95'][0]:+.2f}",
                "delta_steps_ci95_hi": f"{rep['delta_steps']['ci95'][1]:+.2f}",
                "delta_prompt_tokens_mean": f"{rep['delta_prompt_tokens']['mean']:+.1f}",
                "delta_prompt_tokens_ci95_lo": f"{rep['delta_prompt_tokens']['ci95'][0]:+.1f}",
                "delta_prompt_tokens_ci95_hi": f"{rep['delta_prompt_tokens']['ci95'][1]:+.1f}",
            }
        )
    write_csv(out_dir / "table2_interactions.csv", list(rows[0].keys()), rows)


def export_pilots(results_dir: Path, out_dir: Path):
    rows = []
    for run_id in PILOT_RUNS:
        s = load_summary(results_dir, run_id)
        rows.append(
            {
                "run_id": run_id,
                "success_rate": f"{s['success_rate']:.4f}",
                "mean_steps": f"{s['steps']['mean']:.2f}",
                "mean_prompt_tokens": f"{s['prompt_tokens']['mean']:.1f}",
                "mean_memory_tokens": f"{s['memory_tokens']['mean']:.1f}",
                "total_cost_usd": f"{s['total_cost_usd']:.4f}",
            }
        )
    write_csv(out_dir / "appendix_tableA1_matched_footprint.csv", list(rows[0].keys()), rows)


def export_task_reallocation(results_dir: Path, out_dir: Path):
    run_ids = [
        "PILOT_eqfp_nomem_s0_types12",
        "PILOT_eqfp_expel_bm25_s0_types12",
        "PILOT_eqfp_raw_bm25_s0_types12",
        "PILOT_eqfp_skillos_bm25_s0_types12",
    ]
    by_run = {}
    for run_id in run_ids:
        rows = load_jsonl(results_dir / f"{run_id}.episodes.jsonl")
        by_run[run_id] = {
            row["task_id"]: {
                "task_type": row["task_type"],
                "success": int(row["success"]),
                "steps": row["steps"],
            }
            for row in rows
        }
    task_ids = sorted(by_run[run_ids[0]])
    rows = []
    for task_id in task_ids:
        row = {
            "task_id": task_id,
            "task_type": by_run[run_ids[0]][task_id]["task_type"],
        }
        for run_id in run_ids:
            row[f"{run_id}_success"] = by_run[run_id][task_id]["success"]
            row[f"{run_id}_steps"] = by_run[run_id][task_id]["steps"]
        rows.append(row)
    write_csv(out_dir / "appendix_task_reallocation_types12.csv", list(rows[0].keys()), rows)


def export_executor_transfer(results_dir: Path, out_dir: Path):
    rows = []
    task_rows = []
    for executor, spec in EXECUTOR_TRANSFER.items():
        baseline_id = spec["baseline"]
        baseline_summary = load_summary(results_dir, baseline_id)
        baseline_path = results_dir / f"{baseline_id}.episodes.jsonl"
        rows.append(
            {
                "executor": executor,
                "representation": "none",
                "retriever": "none",
                "run_id": baseline_id,
                "success_rate": f"{baseline_summary['success_rate']:.4f}",
                "paired_delta_success_mean": "",
                "paired_delta_success_ci95_lo": "",
                "paired_delta_success_ci95_hi": "",
                "mean_steps": f"{baseline_summary['steps']['mean']:.2f}",
                "mean_prompt_tokens": f"{baseline_summary['prompt_tokens']['mean']:.1f}",
                "realized_memory_tokens": "0.0",
            }
        )
        for representation, retriever, run_id in spec["variants"]:
            summary_path = results_dir / f"{run_id}.summary.json"
            episodes_path = results_dir / f"{run_id}.episodes.jsonl"
            if not summary_path.exists() or not episodes_path.exists():
                continue
            summary = load_summary(results_dir, run_id)
            if summary.get("n") != 402 or summary.get("n_error") != 0:
                continue
            report = compare(baseline_path, episodes_path)
            rows.append(
                {
                    "executor": executor,
                    "representation": representation,
                    "retriever": retriever,
                    "run_id": run_id,
                    "success_rate": f"{summary['success_rate']:.4f}",
                    "paired_delta_success_mean": f"{report['delta_success']['mean']:+.4f}",
                    "paired_delta_success_ci95_lo": f"{report['delta_success']['ci95'][0]:+.4f}",
                    "paired_delta_success_ci95_hi": f"{report['delta_success']['ci95'][1]:+.4f}",
                    "mean_steps": f"{summary['steps']['mean']:.2f}",
                    "mean_prompt_tokens": f"{summary['prompt_tokens']['mean']:.1f}",
                    "realized_memory_tokens": f"{mean_memory_tokens(results_dir, run_id):.1f}",
                }
            )
            for task_type, stats in report["by_task_type"].items():
                task_rows.append(
                    {
                        "executor": executor,
                        "representation": representation,
                        "retriever": retriever,
                        "run_id": run_id,
                        "task_type": task_type,
                        "baseline_success_rate": f"{baseline_summary['by_task_type'][task_type]['success_rate']:.4f}",
                        "delta_success_mean": f"{stats['delta_success']['mean']:+.4f}",
                        "delta_success_ci95_lo": f"{stats['delta_success']['ci95'][0]:+.4f}",
                        "delta_success_ci95_hi": f"{stats['delta_success']['ci95'][1]:+.4f}",
                    }
                )
    write_csv(out_dir / "table3_executor_transfer.csv", list(rows[0].keys()), rows)
    write_csv(out_dir / "figure4_executor_tasktype_transfer.csv", list(task_rows[0].keys()), task_rows)


def export_readme(out_dir: Path):
    text = """# Paper O Assets

Machine-readable assets exported from `skilllib_rc/results/` for the current Paper O draft.

Files:

- `table1_main_results.csv`: main result table values
- `figure1_paired_vs_nomem.csv`: paired episode-level deltas versus the no-memory baseline
- `figure2_tasktype_best_vs_nomem.csv`: task-type deltas for `expel_bm25` versus `nomem`
- `table2_interactions.csv`: pairwise retriever/representation interaction checks
- `appendix_tableA1_matched_footprint.csv`: 6-task and 12-task matched-footprint pilot summaries
- `appendix_task_reallocation_types12.csv`: task-instance success swaps for the 12-task pilot
- `table3_executor_transfer.csv`: executor-specific paired transfer results; completed Qwen BM25 runs are included automatically
- `figure4_executor_tasktype_transfer.csv`: task-type transfer values by executor, representation, and retriever
"""
    (out_dir / "README.md").write_text(text)


def main():
    here = Path(__file__).resolve()
    project_root = here.parent.parent.parent
    results_dir = project_root / "skilllib_rc" / "results"
    out_dir = project_root / "ARR" / "PaperO_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_main_table(results_dir, out_dir)
    export_paired_vs_nomem(results_dir, out_dir)
    export_tasktype(results_dir, out_dir)
    export_pairwise(results_dir, out_dir)
    export_pilots(results_dir, out_dir)
    export_task_reallocation(results_dir, out_dir)
    export_executor_transfer(results_dir, out_dir)
    export_readme(out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
