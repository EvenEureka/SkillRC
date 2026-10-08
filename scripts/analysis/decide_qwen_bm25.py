#!/usr/bin/env python3
"""Classify the Qwen BM25 transfer result from paired confidence intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from report_paired_results import compare


def classify(interval: list[float]) -> str:
    lo, hi = interval
    if lo > 0:
        return "positive"
    if hi < 0:
        return "negative"
    return "null"


def summarize(reference: Path, candidate: Path) -> dict:
    report = compare(reference, candidate)
    return {
        "paired_n": report["paired_n"],
        "delta_success": report["delta_success"],
        "delta_prompt_tokens": report["delta_prompt_tokens"],
        "memory_tokens": report["memory_tokens"],
        "wins_losses_ties": report["wins_losses_ties"],
        "classification": classify(report["delta_success"]["ci95"]),
    }


def choose_next(expel: str, skillos: str) -> dict:
    if expel == "positive":
        return {
            "finding": "retriever_specific_transfer",
            "next": "matched_footprint_qwen_bm25",
            "reason": "Compact insight utility transfers under BM25 but not dense retrieval.",
        }
    if skillos == "positive":
        return {
            "finding": "representation_retriever_interaction",
            "next": "freeze_qwen_matrix_and_analyze_interaction",
            "reason": "Markdown utility changes sign across retrievers on Qwen.",
        }
    return {
        "finding": "executor_dependent_memory_utility",
        "next": "stop_qwen_matrix_and_integrate_executor_moderation",
        "reason": "Neither tested BM25 memory establishes a transferable positive gain.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--baseline", default="Q_nomem_s3")
    parser.add_argument("--expel", default="Q_expel_bm25_s3")
    parser.add_argument("--skillos", default="Q_skillos_bm25_s3")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = args.results_dir / f"{args.baseline}.episodes.jsonl"
    expel = summarize(reference, args.results_dir / f"{args.expel}.episodes.jsonl")
    skillos = summarize(reference, args.results_dir / f"{args.skillos}.episodes.jsonl")
    payload = {
        "baseline": args.baseline,
        "conditions": {
            "expel_bm25": {"run_id": args.expel, **expel},
            "skillos_bm25": {"run_id": args.skillos, **skillos},
        },
        "decision": choose_next(expel["classification"], skillos["classification"]),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
