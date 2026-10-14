#!/usr/bin/env python3
"""W1-B GENERIC control: footprint matching of the trajectory-free
rule list written by generic_rules_w1b_2026-10-13.py to one focal pool. CPU only; no model is called.

Input: results/placebo/w1b_generic_raw.json (per focal item: tag, word count n and up to 6 exact-length variants).
1. One variant per item is chosen: seeded random choices, then hill-climbing by single-item swaps, to the least total
   excess over the focal pool's footprint under both the gpt-4o-mini (o200k) and the Qwen3 tokenizer, keeping the
   feasibility rule of the gardening builder (>= both targets; equal on both or on neither).
2. greedy_adjust (build_irrelevant_control.py) replaces the fewest body words with the nonce "x" until both footprints
   equal the focal pool's.
The output is index-aligned with the focal pool: same tags, same body word count per item. It ranks by the true pool
(rank_reference_pool) at run time. Outputs (never overwritten): data/pools/w1b_<focal>_generic.jsonl and
results/placebo/w1b_<focal>_generic_audit.json. Exit 3 means every choice is shorter than the focal pool (protocol fallback)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

from skillrc.tokens import count_tokens

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


alf = _load("alf_ctl", "scripts/analysis/build_irrelevant_control.py")
prb = _load("prb_ctl", "scripts/analysis/build_irrelevant_control_poolrep.py")
FOCAL = {"focal": ("data/pools/expel_insights.jsonl", "data/pools/expel_insights_irrelevant_coherent.jsonl"),
         "gref": ("data/pools/src1011_qwen_insights.jsonl", "data/pools/poolrep_gref_irrelevant.jsonl")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", required=True, choices=list(FOCAL))
    ap.add_argument("--raw", type=Path, default=ROOT / "results/placebo/w1b_generic_raw.json")
    ap.add_argument("--lengthfix", type=Path, default=ROOT / "results/placebo/w1b_generic_lengthfix.json",
                    help="deviation D1: variants from the word-count feedback loop, keyed by (type, n, base rule)")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--search", type=int, default=3000)
    ap.add_argument("--climb-starts", type=int, default=8)
    ap.add_argument("--climb", type=int, default=2000)
    ap.add_argument("--max-tries", type=int, default=10)
    a = ap.parse_args()
    out_pool = ROOT / f"data/pools/w1b_{a.focal}_generic.jsonl"
    out_audit = ROOT / f"results/placebo/w1b_{a.focal}_generic_audit.json"
    for p in (out_pool, out_audit):
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
    pool_path, garden_path = FOCAL[a.focal]
    pool = alf.load_jsonl(ROOT / pool_path)
    raw = json.loads(a.raw.read_text())
    plan = sorted(raw["focal"][a.focal]["plan"], key=lambda p: p["focal_index"])
    fixed = {}
    if a.lengthfix.exists():
        for it in json.loads(a.lengthfix.read_text())["items"]:
            fixed[(it["type"], it["n"], it["base_rule"])] = it["variants"]
    assert [p["focal_index"] for p in plan] == list(range(len(pool))), "plan must cover every focal item once"
    tags, lens, variants = [], [], []
    for p, it in zip(plan, pool):
        w = it["text"].split()
        assert w[0] == f"[{it['task_type']}]" and p["type"] == it["task_type"] and p["n"] == len(w) - 1
        vs = []
        for v in list(p["variants"]) + list(fixed.get((p["type"], p["n"], p["base_rule"]), [])):
            if len(v.split()) == p["n"] and v not in vs:
                vs.append(v)
        if not vs:
            sys.exit(f"item {p['focal_index']} has no exact-length variant")
        tags.append(w[0])
        lens.append(p["n"])
        variants.append(vs)
    target, qtarget = count_tokens("\n\n".join(it["text"] for it in pool), "gpt-4o-mini"), alf.qwen_count("\n\n".join(it["text"] for it in pool))
    rng = random.Random(a.seed + (1 if a.focal == "gref" else 0))

    def counts(choice):
        return alf.both_counts([[tags[i], *variants[i][choice[i]].split()] for i in range(len(pool))])

    def feasible(g, q):
        return g >= target and q >= qtarget and ((g == target) == (q == qtarget))

    seen, cands, below = set(), [], 0
    for _ in range(a.search):
        choice = tuple(rng.randrange(len(v)) for v in variants)
        if choice in seen:
            continue
        seen.add(choice)
        g, q = counts(choice)
        if feasible(g, q):
            cands.append(((g - target) + (q - qtarget), choice, (g, q)))
        elif g < target or q < qtarget:
            below += 1
    search_kind = "uniform"
    if not cands:
        # Second stage, as in the gardening builder: density-biased choices (each item's variants ordered by o200k tokens per
        # word plus noise of a random spread, from "densest first" to "effectively random"). Runs only if the uniform
        # search found no choice at or above both targets.
        search_kind = "uniform, then density-biased"
        dens = [[count_tokens(v, "gpt-4o-mini") / max(1, len(v.split())) for v in vs] for vs in variants]
        for _ in range(a.search):
            spread = 10 ** rng.uniform(-1.5, 1.5)
            choice = tuple(max(range(len(vs)), key=lambda j: dens[i][j] + rng.uniform(0.0, spread)) for i, vs in enumerate(variants))
            if choice in seen:
                continue
            seen.add(choice)
            g, q = counts(choice)
            if feasible(g, q):
                cands.append(((g - target) + (q - qtarget), choice, (g, q)))
            elif g < target or q < qtarget:
                below += 1
    if not cands:
        print(json.dumps({"target": [target, qtarget], "choices_tried": len(seen), "below_target": below}))
        sys.exit(3)
    cands.sort(key=lambda x: x[0])

    def climb(choice, g, q):
        choice = list(choice)
        for _ in range(a.climb):
            i = rng.randrange(len(pool))
            if len(variants[i]) < 2:
                continue
            j = rng.randrange(len(variants[i]))
            if j == choice[i]:
                continue
            trial = choice[:i] + [j] + choice[i + 1:]
            g2, q2 = counts(trial)
            if feasible(g2, q2) and (g2 - target) + (q2 - qtarget) < (g - target) + (q - qtarget):
                choice, g, q = trial, g2, q2
        return ((g - target) + (q - qtarget), tuple(choice), (g, q))

    cands = sorted((climb(c, g, q) for _, c, (g, q) in cands[:a.climb_starts]), key=lambda x: x[0])
    nrng, chosen, tries = random.Random(a.seed), None, 0
    for excess, choice, (pre_g, pre_q) in cands:
        docs = [[tags[i], *variants[i][choice[i]].split()] for i in range(len(pool))]
        base = [(d, k) for d in range(len(docs)) for k in range(1, len(docs[d]))]
        for t in range(a.max_tries):
            order = list(base) if t == 0 else nrng.sample(base, len(base))
            adj, (g, q), reps = alf.greedy_adjust(docs, target, qtarget, order)
            tries += 1
            if g == target and q == qtarget:
                chosen = (adj, reps, choice, pre_g, pre_q)
                break
        if chosen:
            break
    if chosen is None:
        sys.exit("footprint not matched")
    adj, reps, choice, pre_g, pre_q = chosen
    control = [{**it, "text": " ".join(adj[i])} for i, it in enumerate(pool)]
    for s, c in zip(pool, control):
        assert len(s["text"].split()) == len(c["text"].split()) and s["task_type"] == c["task_type"]
        assert s["text"].split()[0] == c["text"].split()[0]
    ctx = "\n\n".join(c["text"] for c in control)
    assert count_tokens(ctx, "gpt-4o-mini") == target and alf.qwen_count(ctx) == qtarget
    vocab = prb.test_vocab()
    garden = alf.load_jsonl(ROOT / garden_path)
    body = lambda items: [" ".join(x["text"].split()[1:]) for x in items]  # noqa: E731
    out_pool.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in control))
    audit = {"focal": a.focal, "focal_pool": pool_path, "control": str(out_pool.relative_to(ROOT)), "kind": "generic rules, no trajectories",
             "writer": raw["distiller"], "lengthfix_file": str(a.lengthfix.relative_to(ROOT)) if a.lengthfix.exists() else None, "n_items": len(control), "body_words": sum(lens),
             "variants_per_item": [len(v) for v in variants], "variant_chosen": list(choice),
             "search": search_kind, "choices_tried": len(seen), "below_target_choices": below, "tries": tries,
             "nonce_replacements": len(reps), "retained_body_fraction": round(1 - len(reps) / sum(lens), 4),
             "context_tokens": {"gpt_4o_mini_o200k": {"target": target, "before": pre_g, "control": target},
                                "qwen3": {"target": qtarget, "before": pre_q, "control": qtarget}},
             "both_tokenizers_match": True,
             "test_game_vocab_overlap": {"true_pool": prb.overlap_share(body(pool), vocab),
                                         "generic": prb.overlap_share(body(control), vocab),
                                         "gardening_control": prb.overlap_share(body(garden), vocab)},
             "retrieval_control": {"kind": "index-aligned reference ranking", "ranking_corpus": pool_path,
                                   "payload_corpus": str(out_pool.relative_to(ROOT))},
             "replacements": reps,
             "pairs": [{"true": pool[i]["text"], "generic": control[i]["text"]} for i in range(len(pool))]}
    out_audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k not in ("pairs", "replacements")}, indent=2))


if __name__ == "__main__":
    main()
