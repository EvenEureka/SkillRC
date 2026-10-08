#!/usr/bin/env python3
"""ABANDONED pre-outcome word-shuffle prototype; retained for audit only.

The placebo preserves every insight's whitespace-token multiset. Because the
evaluation retriever tokenizes with ``lower().split()``, this preserves every
BM25 document score and therefore the complete retrieval ranking exactly.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import numpy as np
from rank_bm25 import BM25Okapi

from skillrc.tokens import count_tokens, trim_to_budget


TASK_TYPES = ("clean", "cool", "examine", "heat", "put", "puttwo")


def adjacent_overlap(original: list[str], candidate: list[str]) -> int:
    original_bigrams = set(zip(original, original[1:]))
    return sum(pair in original_bigrams for pair in zip(candidate, candidate[1:]))


def scramble(tokens: list[str], rng: random.Random) -> tuple[list[str], int]:
    """Choose a deterministic shuffle with minimal retained local word order."""
    best = list(tokens)
    best_overlap = adjacent_overlap(tokens, best)
    for _ in range(5000):
        candidate = list(tokens)
        rng.shuffle(candidate)
        overlap = adjacent_overlap(tokens, candidate)
        if overlap < best_overlap:
            best, best_overlap = candidate, overlap
        if overlap == 0:
            break
    return best, best_overlap


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=Path("data/pools/expel_insights.jsonl"))
    parser.add_argument("--episodes", type=Path, default=Path("results/R008_nomem_s3.episodes.jsonl"))
    parser.add_argument("--out-pool", type=Path, default=Path("data/pools/expel_insights_wordshuffle.jsonl"))
    parser.add_argument("--out-audit", type=Path, default=Path("results/placebo/semantic_placebo_audit.json"))
    parser.add_argument("--out-tasks", type=Path, default=Path("results/placebo/frozen_tasks.json"))
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--tasks-per-type", type=int, default=10)
    args = parser.parse_args()

    pool = load_jsonl(args.pool)
    rng = random.Random(args.seed)
    placebo = []
    item_audit = []
    for index, item in enumerate(pool):
        words = item["text"].split()
        if not words or words[0] != f"[{item['task_type']}]":
            raise ValueError(f"unexpected task tag in pool item {index}: {item}")
        shuffled, overlap = scramble(words[1:], rng)
        corrupted_text = " ".join([words[0], *shuffled])
        if Counter(item["text"].lower().split()) != Counter(corrupted_text.lower().split()):
            raise AssertionError(f"BM25 token multiset changed for item {index}")
        placebo.append({**item, "text": corrupted_text})
        item_audit.append({
            "index": index,
            "task_type": item["task_type"],
            "retained_adjacent_bigrams": overlap,
            "n_adjacent_bigrams": max(0, len(words) - 2),
            "original": item["text"],
            "placebo": corrupted_text,
        })

    # Freeze task membership from seed-0 metadata only; success/reward is never read.
    tasks_by_type: dict[str, list[int]] = defaultdict(list)
    seen = set()
    for episode in load_jsonl(args.episodes):
        key = int(episode["task_id"])
        if key in seen:
            continue
        seen.add(key)
        tasks_by_type[episode["task_type"]].append(key)
    if set(tasks_by_type) != set(TASK_TYPES):
        raise ValueError(f"unexpected task types: {sorted(tasks_by_type)}")

    task_rng = random.Random(args.seed)
    selected_by_type = {
        task_type: sorted(task_rng.sample(tasks_by_type[task_type], args.tasks_per_type))
        for task_type in TASK_TYPES
    }
    selected = sorted(task for values in selected_by_type.values() for task in values)

    args.out_pool.parent.mkdir(parents=True, exist_ok=True)
    args.out_audit.parent.mkdir(parents=True, exist_ok=True)
    args.out_pool.write_text("".join(json.dumps(item) + "\n" for item in placebo))
    args.out_tasks.write_text(json.dumps({
        "selection_seed": args.seed,
        "selection_source": str(args.episodes),
        "selection_uses_outcomes": False,
        "tasks_per_type": args.tasks_per_type,
        "by_type": selected_by_type,
        "task_ids": selected,
    }, indent=2) + "\n")

    original_context = "\n\n".join(item["text"] for item in pool)
    placebo_context = "\n\n".join(item["text"] for item in placebo)
    target_budget = count_tokens(original_context, "gpt-4o-mini")
    trimmed_placebo = trim_to_budget(placebo_context, target_budget, "gpt-4o-mini", keep="head")

    original_bm25 = BM25Okapi([item["text"].lower().split() for item in pool])
    placebo_bm25 = BM25Okapi([item["text"].lower().split() for item in placebo])
    query_rows = {}
    for episode in load_jsonl(args.episodes):
        query_rows.setdefault(int(episode["task_id"]), episode["task_text"])
    ranking_matches = 0
    max_score_difference = 0.0
    complete_items_after_trim = []
    for query in query_rows.values():
        tokens = query.lower().split()
        original_scores = original_bm25.get_scores(tokens)
        placebo_scores = placebo_bm25.get_scores(tokens)
        max_score_difference = max(
            max_score_difference,
            float(np.max(np.abs(original_scores - placebo_scores))),
        )
        original_order = sorted(range(len(pool)), key=lambda i: original_scores[i], reverse=True)
        placebo_order = sorted(range(len(placebo)), key=lambda i: placebo_scores[i], reverse=True)
        ranking_matches += original_order == placebo_order
        ranked_context = "\n\n".join(placebo[index]["text"] for index in placebo_order)
        retained = trim_to_budget(ranked_context, target_budget, "gpt-4o-mini", keep="head")
        complete = 0
        for index in placebo_order:
            prefix = "\n\n".join(placebo[i]["text"] for i in placebo_order[:complete + 1])
            if retained.startswith(prefix):
                complete += 1
            else:
                break
        complete_items_after_trim.append(complete)
    audit = {
        "seed": args.seed,
        "construction": "within-insight whitespace-token shuffle; task tag held first",
        "bm25_invariance": "exact per-document lower().split() token multiset",
        "n_items": len(pool),
        "retained_adjacent_bigrams": sum(item["retained_adjacent_bigrams"] for item in item_audit),
        "total_adjacent_bigrams": sum(item["n_adjacent_bigrams"] for item in item_audit),
        "context_tokens": {
            model: {
                "original": count_tokens(original_context, model),
                "placebo": count_tokens(placebo_context, model),
            }
            for model in ("gpt-4o-mini", "qwen3-8b")
        },
        "matched_evaluation_budget": target_budget,
        "placebo_tokens_after_budget": count_tokens(trimmed_placebo, "gpt-4o-mini"),
        "placebo_characters_retained": len(trimmed_placebo) / len(placebo_context),
        "bm25_runtime_audit": {
            "queries": len(query_rows),
            "exact_ranking_matches": ranking_matches,
            "max_absolute_score_difference": max_score_difference,
        },
        "complete_items_after_645_token_trim": {
            "min": min(complete_items_after_trim),
            "max": max(complete_items_after_trim),
            "mean": float(np.mean(complete_items_after_trim)),
        },
        "items": item_audit,
    }
    args.out_audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({
        "pool": str(args.out_pool),
        "tasks": selected,
        "n_tasks": len(selected),
        "retained_adjacent_bigrams": audit["retained_adjacent_bigrams"],
        "context_tokens": audit["context_tokens"],
        "matched_evaluation_budget": target_budget,
        "placebo_characters_retained": audit["placebo_characters_retained"],
        "bm25_runtime_audit": audit["bm25_runtime_audit"],
        "complete_items_after_645_token_trim": audit["complete_items_after_645_token_trim"],
    }, indent=2))


if __name__ == "__main__":
    main()
