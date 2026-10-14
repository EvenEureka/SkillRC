#!/usr/bin/env python3
"""Injection audit for the pool-replication configs, run before any episode.

For every TRUE/IRR pair in configs/poolrep_2026_10_12 it builds the harness memory exactly as the runner does
(skillrc.memory.build_memory) and retrieves for the initial observation of each of the 134 test games (the query
the ReAct loop uses). It checks that the control injects the same item indices in the same order as the true pool,
and counts the injected context under o200k (the harness budget tokenizer) and, via a subprocess in the
tokenizer env, under the Qwen3-8B tokenizer (the reader's).
Run with the harness env (needs rank_bm25); --tok-python must have transformers.
Output: results/placebo/poolrep_injection_audit.json"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from skillrc.config import RunConfig  # noqa: E402
from skillrc.memory import build_memory  # noqa: E402

QWEN_COUNT = ("import json,sys; from transformers import AutoTokenizer; "
              "t=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', local_files_only=True); "
              "print(json.dumps([len(t.encode(s, add_special_tokens=False)) for s in json.load(sys.stdin)]))")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-python", default=os.environ.get("SKILLRC_TOK_PY", sys.executable),
                    help="python with transformers and the Qwen3 tokenizer available (default: this python)")
    ap.add_argument("--out", type=Path, default=ROOT / "results/placebo/poolrep_injection_audit.json")
    a = ap.parse_args()
    rows = [json.loads(x) for x in (ROOT / "results/SRC1011_NONE_QWEN.episodes.jsonl").read_text().splitlines() if x.strip()]
    queries = {int(r["task_id"]): r["task_text"] for r in rows if int(r["seed"]) == 0}
    assert sorted(queries) == list(range(134))
    cfgdir = ROOT / "configs/poolrep_2026_10_12"
    arms = {}
    for p in ("GREF", "Q32R0", "Q32R1", "Q32R2", "Q32R3"):
        arms[p] = (cfgdir / f"PRP1012_{p}_TRUE_QWEN.yaml", cfgdir / f"PRP1012_{p}_IRR_QWEN.yaml")
    texts, report = [], {}
    for p, (ct, ci) in arms.items():
        mt, mi = build_memory(RunConfig.load(str(ct))), build_memory(RunConfig.load(str(ci)))
        idx_t = {it["text"]: k for k, it in enumerate(mt.pool)}
        idx_i = {it["text"]: k for k, it in enumerate(mi.pool)}
        rec = {"n_items": len(mt.pool), "token_budget": mt.token_budget, "order_mismatch_games": 0,
               "o200k_mismatch_games": 0, "items_injected": set(), "pairs": []}
        for g, q in sorted(queries.items()):
            ctx_t, tok_t = mt.retrieve(q), mt.last_tokens
            ctx_i, tok_i = mi.retrieve(q), mi.last_tokens
            order_t = [idx_t[s] for s in ctx_t.split("\n\n")]
            order_i = [idx_i[s] for s in ctx_i.split("\n\n")]
            rec["order_mismatch_games"] += order_t != order_i
            rec["o200k_mismatch_games"] += tok_t != tok_i
            rec["items_injected"].add(len(order_t))
            rec["pairs"].append((len(texts), len(texts) + 1, tok_t, tok_i))
            texts += [ctx_t, ctx_i]
        report[p] = rec
    qwen = json.loads(subprocess.run([a.tok_python, "-c", QWEN_COUNT], input=json.dumps(texts), text=True,
                                     capture_output=True, check=True, env=dict(os.environ)).stdout)
    out = {}
    for p, rec in report.items():
        qt = [qwen[i] for i, _, _, _ in rec["pairs"]]
        qi = [qwen[j] for _, j, _, _ in rec["pairs"]]
        out[p] = {"n_items": rec["n_items"], "token_budget_o200k": rec["token_budget"],
                  "items_injected_per_game": sorted(rec["items_injected"]),
                  "o200k_true": sorted({t for _, _, t, _ in rec["pairs"]}), "o200k_control": sorted({c for _, _, _, c in rec["pairs"]}),
                  "o200k_mismatch_games": rec["o200k_mismatch_games"],
                  "qwen3_8b_true": sorted(set(qt)), "qwen3_8b_control": sorted(set(qi)),
                  "qwen3_8b_mismatch_games": sum(x != y for x, y in zip(qt, qi)),
                  "qwen3_8b_max_abs_diff": max(abs(x - y) for x, y in zip(qt, qi)),
                  "order_mismatch_games": rec["order_mismatch_games"], "games": len(rec["pairs"])}
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))
    bad = [p for p, v in out.items() if v["order_mismatch_games"] or v["o200k_mismatch_games"] or v["items_injected_per_game"] != [v["n_items"]]]
    if bad:
        sys.exit(f"injection audit FAILED for {bad}")


if __name__ == "__main__":
    main()
