#!/usr/bin/env python3
"""Coherent, task-irrelevant control for the WebShop memory.

Same construction as the ALFWorld control (scripts/analysis/build_irrelevant_control.py), whose
footprint-matching functions are reused: every item keeps its tag and body word count, the body is a
well-formed imperative sentence about choral singing practice (one line per item, index-aligned, in
data/pools/webshop1010_irrelevant_bodies.txt), and a deterministic greedy pass replaces the fewest body
words with the nonce "x" until the footprint equals the true pool's under the gpt-4o-mini and Qwen3-8B
tokenizers. The audit reports, for the true pool and the control, the share of body words that occur in
the 500 test instructions.
Outputs: data/pools/webshop1010_insights_irrelevant_coherent.jsonl, results/placebo/webshop1010_irrelevant_audit.json"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from pathlib import Path

from skillrc.tokens import count_tokens

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("alf_ctl", ROOT / "scripts/analysis/build_irrelevant_control.py")
alf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(alf)

FORBIDDEN = {"product", "products", "item", "items", "search", "searching", "click", "buy", "buying", "purchase",
             "price", "prices", "priced", "dollar", "dollars", "cost", "cheap", "size", "sizes", "color", "colors",
             "colour", "option", "options", "attribute", "attributes", "page", "pages", "result", "results",
             "shop", "shopping", "cart", "order", "brand", "pack", "count", "ounce", "inch", "beauty", "electronics",
             "fashion", "garden", "grocery", "listing", "description", "instruction", "instructions", "query",
             "black", "white", "queen", "king", "match", "matches", "requirement", "requirements", "quantity"}
STOP = set("a an the and or but if then so of to in on at by for with from as is are be been was were it its this that "
           "these those your you each every all any no not do does than into over under before after while when once "
           "until about up down out off very more most less only also just can will would should must may".split())


def words(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1]  # drop the possessive "s"


def overlap_share(bodies, vocab):
    toks = [w for b in bodies for w in words(b) if w not in STOP]
    return round(sum(w in vocab for w in toks) / max(1, len(toks)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=ROOT / "data/pools/webshop1010_insights.jsonl")
    ap.add_argument("--bodies", type=Path, default=ROOT / "data/pools/webshop1010_irrelevant_bodies.txt")
    ap.add_argument("--out-pool", type=Path, default=ROOT / "data/pools/webshop1010_insights_irrelevant_coherent.jsonl")
    ap.add_argument("--out-audit", type=Path, default=ROOT / "results/placebo/webshop1010_irrelevant_audit.json")
    ap.add_argument("--seed", type=int, default=20260713)
    ap.add_argument("--max-tries", type=int, default=400)
    a = ap.parse_args()
    pool = alf.load_jsonl(a.pool)
    bodies = [b.strip() for b in a.bodies.read_text().splitlines() if b.strip()]
    assert len(bodies) == len(pool), f"{len(bodies)} bodies for {len(pool)} items"
    docs, body_words, problems = [], 0, []
    for i, (item, body) in enumerate(zip(pool, bodies)):
        w = item["text"].split()
        tag, orig = w[0], w[1:]
        assert tag == f"[{item['task_type']}]", f"unexpected tag {tag}"
        nb = body.split()
        if len(nb) != len(orig):
            problems.append(f"item {i} [{item['task_type']}]: {len(nb)} words, need {len(orig)}: {body}")
        bad = set(words(body)) & FORBIDDEN
        if bad:
            problems.append(f"item {i}: forbidden {bad}: {body}")
        docs.append([tag, *nb])
        body_words += len(nb)
    if problems:
        raise SystemExit("\n".join(problems))
    instr = json.loads((ROOT / "data/summaries/webshop_2026-10-10/instructions_seed0_1k.json").read_text())["instructions"]
    vocab = {w for t in range(500) for w in words(instr[str(t)]["instruction"]) if w not in STOP}
    original_context = "\n\n".join(it["text"] for it in pool)
    target, qtarget = count_tokens(original_context, "gpt-4o-mini"), alf.qwen_count(original_context)
    pre_g, pre_q = alf.both_counts(docs)
    if pre_g < target or pre_q < qtarget:
        raise SystemExit(f"control shorter than target before adjustment: gpt {pre_g}/{target}, qwen {pre_q}/{qtarget}; lengthen sentences")
    base = [(d, k) for d in range(len(docs)) for k in range(1, len(docs[d]))]
    rng, best, tries = random.Random(a.seed), None, 0
    for t in range(a.max_tries):
        order = list(base) if t == 0 else rng.sample(base, len(base))
        adj, (g, q), reps = alf.greedy_adjust(docs, target, qtarget, order)
        tries += 1
        dist = abs(g - target) + abs(q - qtarget)
        if best is None or dist < best[3] or (dist == best[3] and len(reps) < len(best[2])):
            best = (adj, (g, q), reps, dist)
        if dist == 0:
            break
    adj, (g, q), reps, dist = best
    if dist:
        raise SystemExit(f"footprint not matched: gpt {g}/{target}, qwen {q}/{qtarget}")
    control = [{**it, "text": " ".join(adj[i])} for i, it in enumerate(pool)]
    a.out_pool.write_text("".join(json.dumps(x) + "\n" for x in control))
    audit = {"seed": a.seed, "domain": "choral singing practice", "n_items": len(control), "body_words": body_words,
             "nonce_replacements": len(reps), "retained_body_fraction": round(1 - len(reps) / body_words, 4), "tries": tries,
             "context_tokens": {"gpt_4o_mini": {"target": target, "before": pre_g, "control": g},
                                "qwen3_8b": {"target": qtarget, "before": pre_q, "control": q}},
             "test_instruction_vocab_overlap": {"true_pool": overlap_share([" ".join(it["text"].split()[1:]) for it in pool], vocab),
                                                "control": overlap_share(bodies, vocab)},
             "forbidden_words": sorted(FORBIDDEN), "replacements": reps,
             "examples": [{"original": pool[i]["text"], "control": control[i]["text"]} for i in range(min(3, len(pool)))]}
    a.out_audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k not in ("replacements", "examples", "forbidden_words")}, indent=2))


if __name__ == "__main__":
    main()
