#!/usr/bin/env python3
"""Order placebo for the WebShop memory.

Same construction as scripts/analysis/build_order_placebo.py (its scramble() is reused with the same seed,
so every item's words are shuffled until no original adjacent word pair survives), then the minimal nonce
substitution restores the true pool's footprint. The ALFWorld builder matched the gpt-4o-mini footprint
and the Qwen3-8B footprint happened to agree; here the greedy pass of the irrelevant-control builder is
used so that both tokenizers are matched by construction (document order first, then seeded orders).
Outputs: data/pools/webshop1010_insights_order_placebo.jsonl, results/placebo/webshop1010_order_placebo_audit.json"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

from skillrc.tokens import count_tokens

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plc = _load("plc", "scripts/analysis/build_order_placebo.py")
alf = _load("alf_ctl", "scripts/analysis/build_irrelevant_control.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=ROOT / "data/pools/webshop1010_insights.jsonl")
    ap.add_argument("--out-pool", type=Path, default=ROOT / "data/pools/webshop1010_insights_order_placebo.jsonl")
    ap.add_argument("--out-audit", type=Path, default=ROOT / "results/placebo/webshop1010_order_placebo_audit.json")
    ap.add_argument("--seed", type=int, default=20260713)
    ap.add_argument("--max-tries", type=int, default=400)
    a = ap.parse_args()
    pool = plc.load_jsonl(a.pool)
    rng = random.Random(a.seed)
    docs, body_words = [], 0
    for i, item in enumerate(pool):
        w = item["text"].split()
        assert w and w[0] == f"[{item['task_type']}]", f"unexpected tag in item {i}"
        docs.append([w[0], *plc.scramble(w[1:], rng)])
        body_words += len(w) - 1
    original = "\n\n".join(it["text"] for it in pool)
    target, qtarget = count_tokens(original, "gpt-4o-mini"), alf.qwen_count(original)
    pre_g, pre_q = alf.both_counts(docs)
    base = [(d, k) for d in range(len(docs)) for k in range(1, len(docs[d]))]
    orng, best, tries = random.Random(a.seed), None, 0
    for t in range(a.max_tries):
        order = list(base) if t == 0 else orng.sample(base, len(base))
        adj, (g, q), reps = alf.greedy_adjust(docs, target, qtarget, order)
        tries += 1
        dist = abs(g - target) + abs(q - qtarget)
        if best is None or dist < best[3] or (dist == best[3] and len(reps) < len(best[2])):
            best = (adj, (g, q), reps, dist)
        if dist == 0:
            break
    adj, (g, q), reps, dist = best
    if dist:
        raise SystemExit(f"footprint not matched: gpt {g}/{target}, qwen {q}/{qtarget} (before: {pre_g}/{pre_q})")
    placebo = [{**it, "text": " ".join(adj[i])} for i, it in enumerate(pool)]
    for src, p in zip(pool, placebo):
        assert len(src["text"].split()) == len(p["text"].split())
        assert plc.adjacent_overlap(src["text"].split()[1:], p["text"].split()[1:]) == 0
    a.out_pool.write_text("".join(json.dumps(x) + "\n" for x in placebo))
    audit = {"seed": a.seed, "construction": "within-insight word shuffle (no original adjacent pair) plus minimal nonce footprint adjustment under both tokenizers",
             "n_items": len(pool), "body_words": body_words, "nonce_replacements": len(reps),
             "retained_body_fraction": round(1 - len(reps) / body_words, 4), "tries": tries,
             "retained_original_adjacent_bigrams": 0,
             "context_tokens": {"gpt_4o_mini": {"target": target, "before": pre_g, "placebo": g},
                                "qwen3_8b": {"target": qtarget, "before": pre_q, "placebo": q}},
             "retrieval_control": {"kind": "index-aligned reference ranking", "ranking_corpus": str(a.pool.resolve().relative_to(ROOT)),
                                   "payload_corpus": str(a.out_pool.resolve().relative_to(ROOT))},
             "replacements": reps, "examples": [{"original": pool[i]["text"], "placebo": placebo[i]["text"]} for i in range(min(3, len(pool)))]}
    a.out_audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k not in ("replacements", "examples")}, indent=2))


if __name__ == "__main__":
    main()
