#!/usr/bin/env python3
"""Build the final matched-order-control semantic placebo for Paper O."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from skillrc.tokens import count_tokens


TASK_TYPES = ("clean", "cool", "examine", "heat", "put", "puttwo")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def adjacent_overlap(original: list[str], candidate: list[str]) -> int:
    original_bigrams = set(zip(original, original[1:]))
    return sum(pair in original_bigrams for pair in zip(candidate, candidate[1:]))


def scramble(tokens: list[str], rng: random.Random) -> list[str]:
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
    if best_overlap != 0:
        raise RuntimeError(f"could not remove all adjacent bigrams: {tokens}")
    return best


def context(docs: list[list[str]]) -> str:
    return "\n\n".join(" ".join(doc) for doc in docs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=Path("data/pools/expel_insights.jsonl"))
    parser.add_argument("--episodes", type=Path, default=Path("results/R008_nomem_s3.episodes.jsonl"))
    parser.add_argument("--out-pool", type=Path, default=Path("data/pools/expel_insights_order_placebo.jsonl"))
    parser.add_argument("--out-audit", type=Path, default=Path("results/placebo/order_placebo_audit.json"))
    parser.add_argument("--out-tasks", type=Path, default=Path("results/placebo/frozen_tasks.json"))
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--tasks-per-type", type=int, default=10)
    args = parser.parse_args()

    pool = load_jsonl(args.pool)
    rng = random.Random(args.seed)
    docs = []
    body_tokens = 0
    for index, item in enumerate(pool):
        words = item["text"].split()
        if not words or words[0] != f"[{item['task_type']}]":
            raise ValueError(f"unexpected task tag in pool item {index}: {item}")
        body = words[1:]
        docs.append([words[0], *scramble(body, rng)])
        body_tokens += len(body)

    original_context = "\n\n".join(item["text"] for item in pool)
    target_tokens = count_tokens(original_context, "gpt-4o-mini")
    preadjust_tokens = count_tokens(context(docs), "gpt-4o-mini")
    current_tokens = preadjust_tokens
    replacements = []
    # Replace the minimum number found by a deterministic greedy pass needed to
    # restore the original BPE footprint. Item count and word positions stay fixed.
    for doc_index, doc in enumerate(docs):
        for word_index in range(1, len(doc)):
            if current_tokens == target_tokens:
                break
            original_word = doc[word_index]
            doc[word_index] = "x"
            proposed = count_tokens(context(docs), "gpt-4o-mini")
            if target_tokens <= proposed < current_tokens:
                replacements.append({
                    "doc_index": doc_index,
                    "word_index": word_index,
                    "original_word": original_word,
                })
                current_tokens = proposed
            else:
                doc[word_index] = original_word
        if current_tokens == target_tokens:
            break
    if current_tokens != target_tokens:
        raise RuntimeError(f"could not match footprint: target={target_tokens}, got={current_tokens}")

    placebo = [{**item, "text": " ".join(docs[index])} for index, item in enumerate(pool)]
    for source, corrupted in zip(pool, placebo):
        if len(source["text"].split()) != len(corrupted["text"].split()):
            raise AssertionError("document word count changed")
        if source["task_type"] != corrupted["task_type"]:
            raise AssertionError("reference and payload pools are not index aligned")
        source_body = source["text"].split()[1:]
        corrupted_body = corrupted["text"].split()[1:]
        if adjacent_overlap(source_body, corrupted_body) != 0:
            raise AssertionError("an original adjacent body bigram survived")

    args.out_pool.parent.mkdir(parents=True, exist_ok=True)
    args.out_pool.write_text("".join(json.dumps(item) + "\n" for item in placebo))
    if not args.episodes.exists():
        # The released configs already pin the frozen 60-task sample; the episode
        # log is only needed to re-derive it.
        print(f"wrote {args.out_pool}; {args.episodes} not found, skipping task selection")
        print(json.dumps({"target_tokens": target_tokens, "preadjust_tokens": preadjust_tokens,
                          "body_tokens": body_tokens, "nonce_replacements": len(replacements),
                          "retained_body_fraction": round(1 - len(replacements) / body_tokens, 4)}))
        return
    episodes = load_jsonl(args.episodes)
    tasks_by_type: dict[str, list[int]] = defaultdict(list)
    seen = set()
    for episode in episodes:
        task_id = int(episode["task_id"])
        if task_id in seen:
            continue
        seen.add(task_id)
        tasks_by_type[episode["task_type"]].append(task_id)
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

    audit = {
        "seed": args.seed,
        "construction": "within-insight word shuffle plus minimal nonce footprint adjustment",
        "n_items": len(pool),
        "all_items_complete": True,
        "document_word_counts_preserved": True,
        "body_tokens": body_tokens,
        "body_tokens_replaced_by_nonce": len(replacements),
        "lexical_retention_fraction": (body_tokens - len(replacements)) / body_tokens,
        "retained_original_adjacent_bigrams": 0,
        "context_tokens": {
            "original": target_tokens,
            "shuffled_before_adjustment": preadjust_tokens,
            "placebo": count_tokens(context(docs), "gpt-4o-mini"),
        },
        "retrieval_control": {
            "kind": "index-aligned reference ranking",
            "ranking_corpus": str(args.pool),
            "payload_corpus": str(args.out_pool),
            "expected_order_identity": "exact by construction: both conditions score the original pool",
        },
        "nonce_replacements": replacements,
        "examples": [
            {"original": pool[index]["text"], "placebo": placebo[index]["text"]}
            for index in range(3)
        ],
    }
    args.out_audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({
        "pool": str(args.out_pool),
        "n_tasks": len(selected),
        "body_tokens_replaced_by_nonce": len(replacements),
        "lexical_retention_fraction": audit["lexical_retention_fraction"],
        "retained_original_adjacent_bigrams": 0,
        "context_tokens": audit["context_tokens"],
    }, indent=2))


if __name__ == "__main__":
    main()
