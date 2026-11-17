#!/usr/bin/env python3
"""Block D payloads for the WebShop release check (engineering/GATE_DESIGN_v3.md §5, v4 §5).

From the published WebShop pool PUB (data/pools/webshop1010_insights.jsonl, 30 rules) build four payloads in which
six rules are replaced IN PLACE by choral-singing sentences, everything else byte-identical:

  ABL      the six frozen suspects (lines 4, 11, 16, 21, 27, 30): rules that direct another search, a revised
           search or pagination after inadequate results and contain no search-count, step-budget or stopping rule
  COMP1-3  six non-suspect rules each, category profile beauty 1 / electronics 1 / fashion 1 / garden 1 / grocery 2,
           total removed body words within +-5% of the suspects' total, rejection sampling with seeds 20261116 + i

Matching: each replacement has the replaced rule's body word count; the whole payload ("\\n\\n".join of the items, the
injected block) has PUB's token count under o200k (harness budget) and the Qwen3 tokenizer (byte-identical across
Qwen3-8B/14B/32B). Search over choral realisations of the right length, then replace body words of the
replacements with the nonce "x" only if needed. Per-item token differences are reported, not required to be zero.

Outputs (never overwritten): data/pools/gate_ws_2026_11/PUB_{ABL,COMP1,COMP2,COMP3}.jsonl and
data/pools/gate_ws_2026_11/payload_audit.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from skillrc.tokens import count_tokens  # noqa: E402

POOL = ROOT / "data/pools/webshop1010_insights.jsonl"
CHORAL_PUB = ROOT / "data/pools/webshop1010_insights_irrelevant_coherent.jsonl"   # published control, index-aligned
BANK = ROOT / "data/pools/r2_choral_bank.txt"
OUT = ROOT / "data/pools/gate_ws_2026_11"
SUSPECT_LINES = [4, 11, 16, 21, 27, 30]           # 1-based, frozen in GATE_DESIGN_v2 table B
CRITERION = ("explicitly directs another search, a revised search, or pagination after inadequate results, and "
             "contains no search-count, step-budget or stopping rule; excluded: initial-query construction, filters, "
             "inspecting current results")
PROFILE = {"beauty": 1, "electronics": 1, "fashion": 1, "garden": 1, "grocery": 2}
SEED = 20261116
TOL = 0.05


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def body(text: str) -> str:
    return text.split(" ", 1)[1]


def load(p: Path):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def qwen_counter():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    return lambda t: len(tok.encode(t, add_special_tokens=False))


def load_bank_module():
    spec = importlib.util.spec_from_file_location("prb", ROOT / "scripts/analysis/build_irrelevant_control_poolrep.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sample_complements(pool, suspects, n=3):
    target = sum(len(body(pool[i - 1]["text"]).split()) for i in suspects)
    lo, hi = (1 - TOL) * target, (1 + TOL) * target
    by_cat = {c: [i for i in range(1, len(pool) + 1) if i not in suspects and pool[i - 1]["task_type"] == c]
              for c in PROFILE}
    eligible = []
    for combo in itertools.product(*[list(itertools.combinations(by_cat[c], k)) for c, k in PROFILE.items()]):
        lines = sorted(x for part in combo for x in part)
        tot = sum(len(body(pool[i - 1]["text"]).split()) for i in lines)
        if lo <= tot <= hi:
            eligible.append((tuple(lines), tot))
    chosen, log = [], []
    for k in range(1, n + 1):
        rng = random.Random(SEED + k)
        tries = 0
        while True:
            tries += 1
            lines = sorted(x for c, m in PROFILE.items() for x in rng.sample(by_cat[c], m))
            tot = sum(len(body(pool[i - 1]["text"]).split()) for i in lines)
            if lo <= tot <= hi and tuple(lines) not in [c[0] for c in chosen]:
                chosen.append((tuple(lines), tot))
                log.append({"name": f"COMP{k}", "seed": SEED + k, "tries": tries, "lines": lines, "body_words": tot})
                break
    overlap = {f"COMP{a + 1}-COMP{b + 1}": len(set(chosen[a][0]) & set(chosen[b][0]))
               for a, b in itertools.combinations(range(n), 2)}
    return target, (lo, hi), len(eligible), log, overlap


def build_payload(pool, lines, candidates, cnt, rng, search=40000):
    texts = [it["text"] for it in pool]
    target = tuple(c("\n\n".join(texts)) for c in cnt)
    idx = [i - 1 for i in lines]
    tags = {i: texts[i].split(" ", 1)[0] for i in idx}

    def assemble(bodies):
        t = list(texts)
        for i, b in zip(idx, bodies):
            t[i] = f"{tags[i]} {b}"
        return t

    def counts(t):
        return tuple(c("\n\n".join(t)) for c in cnt)

    best = None
    seen = set()
    for _ in range(search):
        bodies = tuple(rng.choice(candidates[i]) for i in idx)
        if bodies in seen:
            continue
        seen.add(bodies)
        v = counts(assemble(bodies))
        if v == target:
            best = (0, bodies, v)
            break
        if all(x >= y for x, y in zip(v, target)) and (best is None or sum(v) - sum(target) < best[0]):
            best = (sum(v) - sum(target), bodies, v)
    if best is None:
        raise SystemExit(f"no realisation reaches the targets {target} for lines {lines}")
    bodies = [b.split() for b in best[1]]
    v = best[2]
    nonce = 0
    while v != target:          # greedy nonce trimming inside the replacement bodies only
        improved = False
        for bi, ws in enumerate(bodies):
            for wi, w in enumerate(ws):
                if w == "x":
                    continue
                trial = [list(x) for x in bodies]
                trial[bi][wi] = "x"
                v2 = counts(assemble([" ".join(x) for x in trial]))
                if all(a >= b for a, b in zip(v2, target)) and all(a <= b for a, b in zip(v2, v)) and v2 != v:
                    bodies, v, nonce, improved = trial, v2, nonce + 1, True
                    break
            if improved or v == target:
                break
        if not improved:
            raise SystemExit(f"nonce trimming stuck at {v} (target {target}) for lines {lines}")
    final = assemble([" ".join(b) for b in bodies])
    per_item = []
    for i in idx:
        per_item.append({"line": i + 1, "replaced_sha256": sha(texts[i]), "replaced": texts[i], "replacement": final[i],
                         "body_words": [len(body(texts[i]).split()), len(body(final[i]).split())],
                         "token_diff": {n: c(final[i]) - c(texts[i]) for n, c in zip(("o200k", "qwen3"), cnt)}})
    return final, {"target_tokens": dict(zip(("o200k", "qwen3"), target)), "payload_tokens": dict(zip(("o200k", "qwen3"), v)),
                   "nonce_words": nonce, "items": per_item}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    outs = [OUT / f"PUB_{n}.jsonl" for n in ("ABL", "COMP1", "COMP2", "COMP3")] + [OUT / "payload_audit.json"]
    for p in outs:
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
    pool = load(POOL)
    assert len(pool) == 30
    choral_pub = load(CHORAL_PUB)
    assert len(choral_pub) == 30
    prb = load_bank_module()
    bank = prb.load_bank(BANK)
    sys.path.insert(0, str(ROOT / "scripts/webshop"))
    spec = importlib.util.spec_from_file_location("ws", ROOT / "scripts/webshop/build_irrelevant_control_webshop.py")
    ws = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ws)
    forbidden = set(ws.FORBIDDEN)
    by_len = {}
    for t in bank:
        for n, ss in t["by_len"].items():
            by_len.setdefault(n, set()).update(ss)
    candidates = {}
    for i, it in enumerate(pool):
        n = len(body(it["text"]).split())
        c = set(by_len.get(n, set())) | {body(choral_pub[i]["text"])}
        c = sorted(s for s in c if not (set(prb.words(s)) & forbidden) and len(s.split()) == n)
        candidates[i] = c
    cnt = [lambda t: count_tokens(t, "gpt-4o-mini"), qwen_counter()]
    suspects = SUSPECT_LINES
    target_words, window, n_eligible, comps, overlap = sample_complements(pool, suspects)
    plan = [("ABL", suspects)] + [(c["name"], c["lines"]) for c in comps]
    audit = {"source_pool": str(POOL.relative_to(ROOT)), "source_sha256": hashlib.sha256(POOL.read_bytes()).hexdigest(),
             "criterion": CRITERION, "suspect_lines": suspects,
             "suspects": [{"line": i, "text": pool[i - 1]["text"], "sha256": sha(pool[i - 1]["text"])} for i in suspects],
             "complement_rule": {"profile": PROFILE, "suspect_body_words": target_words, "window": window,
                                 "n_eligible_subsets": n_eligible, "draws": comps, "pairwise_overlap": overlap},
             "payloads": {}}
    for k, (name, lines) in enumerate(plan):
        rng = random.Random(SEED * 10 + k)
        final, rec = build_payload(pool, lines, candidates, cnt, rng)
        with (OUT / f"PUB_{name}.jsonl").open("w") as f:
            for it, t in zip(pool, final):
                f.write(json.dumps({**it, "text": t}, ensure_ascii=False) + "\n")
        untouched = [i + 1 for i in range(30) if i + 1 not in lines and final[i] != pool[i]["text"]]
        assert not untouched, untouched
        rec["sha256"] = hashlib.sha256((OUT / f"PUB_{name}.jsonl").read_bytes()).hexdigest()
        rec["lines"] = lines
        audit["payloads"][name] = rec
        print(name, lines, rec["payload_tokens"], "target", rec["target_tokens"], "nonce", rec["nonce_words"])
    (OUT / "payload_audit.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
