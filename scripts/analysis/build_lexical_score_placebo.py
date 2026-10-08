#!/usr/bin/env python3
"""ABANDONED pre-outcome repeated-nonce prototype; retained for audit only.

Tokens that occur in any ALFWorld benchmark query are preserved. All other body
tokens are replaced by nonce strings while document word counts stay fixed. The
evaluation ranks an index-aligned copy of the original pool and returns these
corrupted payloads, exactly preserving the true condition's BM25 order. Nonces
are selected to exactly match the original context's BPE length.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from skillrc.tokens import count_tokens


TASK_TYPES = ("clean", "cool", "examine", "heat", "put", "puttwo")
NONCES = ("zxq", "qzxv", "blorf", "xyzzy", "quux", "x-x", "zzzz", "qwerty", "xqzvjk", "000000")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def context(docs: list[list[str]]) -> str:
    return "\n\n".join(" ".join(doc) for doc in docs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=Path("data/pools/expel_insights.jsonl"))
    parser.add_argument("--episodes", type=Path, default=Path("results/R008_nomem_s3.episodes.jsonl"))
    parser.add_argument("--out-pool", type=Path, default=Path("data/pools/expel_insights_lexical_placebo.jsonl"))
    parser.add_argument("--out-audit", type=Path, default=Path("results/placebo/lexical_placebo_audit.json"))
    parser.add_argument("--out-tasks", type=Path, default=Path("results/placebo/frozen_tasks.json"))
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--tasks-per-type", type=int, default=10)
    args = parser.parse_args()

    pool = load_jsonl(args.pool)
    episodes = load_jsonl(args.episodes)
    query_rows = {}
    for episode in episodes:
        query_rows.setdefault(int(episode["task_id"]), episode["task_text"])
    query_vocab = {token.lower() for query in query_rows.values() for token in query.split()}
    if any(nonce in query_vocab for nonce in ("x", *NONCES)):
        raise ValueError("a nonce unexpectedly occurs in the benchmark query vocabulary")

    docs = []
    replaceable = []
    preserved_body = 0
    for doc_index, item in enumerate(pool):
        words = item["text"].split()
        if not words or words[0] != f"[{item['task_type']}]":
            raise ValueError(f"unexpected task tag in pool item {doc_index}: {item}")
        placebo_words = [words[0]]
        for word_index, word in enumerate(words[1:], start=1):
            if word.lower() in query_vocab:
                placebo_words.append(word)
                preserved_body += 1
            else:
                placebo_words.append("x")
                replaceable.append((doc_index, word_index))
        docs.append(placebo_words)

    target_tokens = count_tokens("\n\n".join(item["text"] for item in pool), "gpt-4o-mini")
    current_tokens = count_tokens(context(docs), "gpt-4o-mini")
    expanded_nonces = 0
    # Greedily increase nonce BPE cost without changing whitespace-token count.
    for doc_index, word_index in replaceable:
        if current_tokens == target_tokens:
            break
        old = docs[doc_index][word_index]
        best = None
        for nonce in NONCES:
            docs[doc_index][word_index] = nonce
            proposed = count_tokens(context(docs), "gpt-4o-mini")
            increase = proposed - current_tokens
            if 0 < increase <= target_tokens - current_tokens:
                candidate = (increase, nonce, proposed)
                if best is None or candidate < best:
                    best = candidate
        docs[doc_index][word_index] = old
        if best is not None:
            docs[doc_index][word_index] = best[1]
            current_tokens = best[2]
            expanded_nonces += 1
    if current_tokens != target_tokens:
        raise RuntimeError(f"could not match footprint: target={target_tokens}, got={current_tokens}")

    placebo = [{**item, "text": " ".join(docs[index])} for index, item in enumerate(pool)]
    for original, corrupted in zip(pool, placebo):
        if len(original["text"].split()) != len(corrupted["text"].split()):
            raise AssertionError("document whitespace-token length changed")

    if any(source["task_type"] != corrupted["task_type"] for source, corrupted in zip(pool, placebo)):
        raise AssertionError("reference and payload pools are not index-aligned by task type")

    # Freeze task membership from type metadata only; no outcome field is read.
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

    body_tokens = preserved_body + len(replaceable)
    audit = {
        "seed": args.seed,
        "construction": "preserve benchmark-query vocabulary; replace other body tokens with nonces",
        "query_construction_uses_outcomes": False,
        "n_items": len(pool),
        "all_items_complete": True,
        "document_word_counts_preserved": True,
        "body_tokens": body_tokens,
        "body_tokens_preserved": preserved_body,
        "body_tokens_replaced": len(replaceable),
        "body_replacement_fraction": len(replaceable) / body_tokens,
        "expanded_nonces_for_bpe_match": expanded_nonces,
        "context_tokens": {
            "original": target_tokens,
            "placebo": count_tokens(context(docs), "gpt-4o-mini"),
        },
        "retrieval_control": {
            "kind": "index-aligned reference ranking",
            "ranking_corpus": str(args.pool),
            "payload_corpus": str(args.out_pool),
            "queries_covered": len(query_rows),
            "expected_order_identity": "exact by construction: both conditions score the original pool",
        },
        "examples": [
            {"original": pool[index]["text"], "placebo": placebo[index]["text"]}
            for index in range(3)
        ],
    }
    args.out_audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({
        "pool": str(args.out_pool),
        "n_tasks": len(selected),
        "body_replacement_fraction": audit["body_replacement_fraction"],
        "context_tokens": audit["context_tokens"],
        "retrieval_control": audit["retrieval_control"],
    }, indent=2))


if __name__ == "__main__":
    main()
