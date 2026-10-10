#!/usr/bin/env python3
"""Build the coherent, task-irrelevant control memory (2026-10-10 rerun of the placebo gate).

The order placebo keeps the 33 items, their task tags, indices, word counts, retrieval order and
645-token footprint, but scrambles the words. It therefore confounds "no task content" with
"incoherent text". This control keeps everything the placebo keeps and restores coherence: every
item is a well-formed imperative sentence with the SAME word count as the original insight, under the
SAME task tag, but about an unrelated activity (vegetable gardening). None of the sentences mentions an
ALFWorld object, receptacle, appliance or task verb.

Footprint matching follows the placebo builder exactly: a deterministic greedy pass replaces the
minimum number of body words with the nonce "x" until the GPT-4o-mini footprint equals the true
memory's; the Qwen3-8B footprint is then audited and, if it differs, the greedy pass is re-run with
a different word order until both tokenizers agree (the placebo matched both at 645/645).
Outputs: data/pools/expel_insights_irrelevant_coherent.jsonl and results/placebo/irrelevant_control_audit.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from skillrc.tokens import count_tokens

# Same tag and the same number of body words as the corresponding item of data/pools/expel_insights.jsonl.
BODIES = [
    "Water the seedlings early in the morning before the sun climbs high.",
    "Loosen the soil around each bed with a fork so that roots spread very easily and drain well.",
    "Always pull weeds while they are small so they never set seed across the whole garden.",
    "Mulch the beds with fresh straw after planting to keep moisture in and weeds down.",
    "Rotate crops every season: beans, then leafy greens, then roots, to rest the tired soil.",
    "Start tomato seeds indoors about six weeks before the last expected spring frost.",
    "Pinch off the lower leaves of tomato plants to improve air flow.",
    "If the leaves turn yellow, test the soil before adding more fertilizer.",
    "After the first harvest, feed plants with compost tea to encourage a second flush.",
    "Stake tall plants before storms arrive so the stems do not snap.",
    "Keep a simple log of sowing dates, rainfall, and the first blooms.",
    "Choose plants suited to the local climate and then group them by their water needs.",
    "Thin crowded seedlings to the strongest one per cell once true leaves appear.",
    "Harvest herbs in the morning after the dew dries but before the sun strengthens.",
    "Always label each row with the variety name and the sowing date.",
    "If a plant wilts in the afternoon but recovers by evening, it is probably just hot, not thirsty.",
    "Save seed only from the healthiest plants so the next season starts out strong.",
    "Prune fruit trees in late winter while the branches are still completely bare.",
    "Sharpen the pruning shears each spring so that every cut heals quickly.",
    "Use a rain gauge to decide when to irrigate instead of watering on a schedule.",
    "Turn the compost pile every two weeks to keep it breaking down.",
    "Plant marigolds along the garden borders to deter aphids naturally.",
    "Sow carrots directly into very fine, stone-free soil, because transplanting disturbs the taproot and usually produces forked, stunted roots.",
    "If frost threatens after planting, cover the young plants overnight with a light garden fleece.",
    "Always water at the base of the plant rather than over the leaves.",
    "Once the pods swell, pick peas every other day to keep the vines producing new pods.",
    "Let the soil dry slightly between waterings so that the roots grow deep and strong.",
    "When transplanting young seedlings, handle them by the leaves, never by the thin, fragile stem.",
    "Start by testing the soil pH and amending it before the first seeds go in.",
    "Check the undersides of leaves weekly, since pests usually hide there first.",
    "Once the garlic tops fall over, lift the bulbs carefully and cure them in a dry, airy, shaded place.",
    "If the first sowing fails to germinate, resow a fresh batch two weeks later, as cold soil often delays sprouting.",
    "Keep a record of which varieties thrived so that next year's planting plan improves.",
]
# ALFWorld vocabulary that must not appear in the control (objects, receptacles, appliances, task verbs).
FORBIDDEN = {"fridge", "microwave", "sink", "countertop", "counter", "drawer", "cabinet", "desk", "lamp",
             "shelf", "sofa", "armchair", "toilet", "bathtub", "garbagecan", "mug", "egg", "pillow", "book",
             "apple", "bowl", "plate", "cup", "knife", "clean", "heat", "cool", "examine", "put", "take",
             "container", "containers", "location", "locations", "item", "items", "object", "objects"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def context(docs: list[list[str]]) -> str:
    return "\n\n".join(" ".join(doc) for doc in docs)


def qwen_count(text: str) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    return len(tok.encode(text, add_special_tokens=False))


_QWEN = None


def qwen_tok():
    global _QWEN
    if _QWEN is None:
        from transformers import AutoTokenizer
        _QWEN = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    return _QWEN


def both_counts(docs) -> tuple[int, int]:
    text = context(docs)
    return count_tokens(text, "gpt-4o-mini"), len(qwen_tok().encode(text, add_special_tokens=False))


def greedy_adjust(docs: list[list[str]], target: int, qtarget: int, order: list[tuple[int, int]]):
    """Replace body words with the nonce in the given order until BOTH footprints equal their targets.
    A replacement is kept only if it moves neither tokenizer below its target and reduces at least one."""
    docs = [list(d) for d in docs]
    g, q = both_counts(docs)
    replacements = []
    for doc_index, word_index in order:
        if g == target and q == qtarget:
            break
        original = docs[doc_index][word_index]
        docs[doc_index][word_index] = "x"
        pg, pq = both_counts(docs)
        if pg >= target and pq >= qtarget and (pg < g or pq < q) and not (pg > g or pq > q):
            replacements.append({"doc_index": doc_index, "word_index": word_index, "original_word": original})
            g, q = pg, pq
        else:
            docs[doc_index][word_index] = original
    return docs, (g, q), replacements


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=Path("data/pools/expel_insights.jsonl"))
    ap.add_argument("--out-pool", type=Path, default=Path("data/pools/expel_insights_irrelevant_coherent.jsonl"))
    ap.add_argument("--out-audit", type=Path, default=Path("results/placebo/irrelevant_control_audit.json"))
    ap.add_argument("--seed", type=int, default=20260713)
    ap.add_argument("--max-tries", type=int, default=400)
    a = ap.parse_args()
    pool = load_jsonl(a.pool)
    assert len(pool) == len(BODIES) == 33
    docs, body_words = [], 0
    for item, body in zip(pool, BODIES):
        words = item["text"].split()
        tag, orig_body = words[0], words[1:]
        new_body = body.split()
        if len(new_body) != len(orig_body):
            raise ValueError(f"word count {len(new_body)} != {len(orig_body)} for [{item['task_type']}]: {body}")
        if tag != f"[{item['task_type']}]":
            raise ValueError(f"unexpected tag {tag}")
        bad = {w.strip(".,:;()'\"").lower() for w in new_body} & FORBIDDEN
        if bad:
            raise ValueError(f"forbidden ALFWorld vocabulary {bad} in: {body}")
        docs.append([tag, *new_body])
        body_words += len(new_body)
    original_context = "\n\n".join(item["text"] for item in pool)
    target = count_tokens(original_context, "gpt-4o-mini")
    qwen_target = qwen_count(original_context)
    pre_gpt, pre_qwen = count_tokens(context(docs), "gpt-4o-mini"), qwen_count(context(docs))
    if pre_gpt < target:
        raise RuntimeError(f"control is shorter than the target under GPT ({pre_gpt} < {target}); lengthen the sentences")
    # deterministic first pass in document order, then seeded random orders until both tokenizers agree
    base_order = [(d, w) for d in range(len(docs)) for w in range(1, len(docs[d]))]
    rng = random.Random(a.seed)
    tries = 0
    best = None
    for t in range(a.max_tries):
        order = list(base_order) if t == 0 else rng.sample(base_order, len(base_order))
        adj, (g, q), reps = greedy_adjust(docs, target, qwen_target, order)
        tries += 1
        dist = abs(g - target) + abs(q - qwen_target)
        if best is None or dist < best[3] or (dist == best[3] and len(reps) < len(best[2])):
            best = (adj, q, reps, dist, g)
        if dist == 0:
            break
    adj, q, reps, dist, g = best
    if g != target:
        raise RuntimeError(f"could not match the GPT footprint: {g} != {target}")
    control = [{**item, "text": " ".join(adj[i])} for i, item in enumerate(pool)]
    for src, ctl in zip(pool, control):
        assert len(src["text"].split()) == len(ctl["text"].split()), "word count changed"
        assert src["task_type"] == ctl["task_type"]
    a.out_pool.parent.mkdir(parents=True, exist_ok=True)
    a.out_pool.write_text("".join(json.dumps(x) + "\n" for x in control))
    audit = {"seed": a.seed, "construction": "coherent task-irrelevant (gardening) sentences with the original word counts and tags, "
                                              "plus the placebo's minimal nonce footprint adjustment",
             "n_items": len(control), "body_words": body_words, "nonce_replacements": len(reps),
             "retained_body_fraction": round(1 - len(reps) / body_words, 4), "tries": tries,
             "context_tokens": {"gpt_4o_mini": {"target": target, "before_adjustment": pre_gpt, "control": count_tokens(context(adj), "gpt-4o-mini")},
                                "qwen3_8b_native": {"target": qwen_target, "before_adjustment": pre_qwen, "control": q}},
             "both_tokenizers_match": q == qwen_target,
             "retrieval_control": {"kind": "index-aligned reference ranking", "ranking_corpus": str(a.pool), "payload_corpus": str(a.out_pool)},
             "replacements": reps, "examples": [{"original": pool[i]["text"], "control": control[i]["text"]} for i in range(3)]}
    a.out_audit.parent.mkdir(parents=True, exist_ok=True)
    a.out_audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k not in ("replacements", "examples")}, indent=2))
    if q != qwen_target:
        raise SystemExit(f"Qwen footprint {q} != {qwen_target}; increase --max-tries or edit a sentence")


if __name__ == "__main__":
    main()
