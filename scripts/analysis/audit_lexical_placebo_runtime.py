#!/usr/bin/env python3
"""Verify reference ranking, item coverage, and tokenizer footprint for the placebo."""
from __future__ import annotations

import json
from pathlib import Path

from skillrc.config import RunConfig
from skillrc.memory import build_memory


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    episodes = {}
    for line in (ROOT / "results" / "R008_nomem_s3.episodes.jsonl").read_text().splitlines():
        row = json.loads(line)
        episodes.setdefault(int(row["task_id"]), row["task_text"])

    placebo_cfg = RunConfig.load(str(ROOT / "configs" / "papero_placebo_gpt.yaml"))
    true_cfg = RunConfig.load(str(ROOT / "configs" / "papero_placebo_gpt.yaml"))
    true_cfg.demo_pool = "data/pools/expel_insights.jsonl"
    true_cfg.rank_reference_pool = None
    placebo = build_memory(placebo_cfg)
    true = build_memory(true_cfg)

    exact_orders = 0
    true_tokens = set()
    placebo_tokens = set()
    for query in episodes.values():
        true_order = [true.pool.index(item) for item in true._rank_bm25(query)]
        placebo_order = [placebo.pool.index(item) for item in placebo._rank_bm25(query)]
        exact_orders += true_order == placebo_order
        true.retrieve(query)
        placebo.retrieve(query)
        true_tokens.add(true.last_tokens)
        placebo_tokens.add(placebo.last_tokens)

    audit = {
        "queries": len(episodes),
        "exact_order_matches": exact_orders,
        "true_items": len(true.pool),
        "placebo_items": len(placebo.pool),
        "true_realized_tokens": sorted(true_tokens),
        "placebo_realized_tokens": sorted(placebo_tokens),
    }
    if audit != {
        "queries": 134,
        "exact_order_matches": 134,
        "true_items": 33,
        "placebo_items": 33,
        "true_realized_tokens": [645],
        "placebo_realized_tokens": [645],
    }:
        raise AssertionError(audit)
    out = ROOT / "results" / "placebo" / "order_placebo_runtime_audit.json"
    out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
