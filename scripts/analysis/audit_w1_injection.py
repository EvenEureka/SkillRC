#!/usr/bin/env python3
"""Injection audit for W1-A (crossover study) and W1-B (generic-rule study), run before any episode.

For every TRUE/payload pair it builds the harness memory exactly as the runner does (skillrc.memory.build_memory) from
the Qwen3-8B configs and retrieves for the initial observation of each of the 134 test games. It requires the payload
to inject the same item indices in the same order as the true pool, every item to be injected, and equal lengths
under o200k (the harness budget tokenizer) and the Qwen3 tokenizer (shared by Qwen3-8B/14B/32B; counted in a
subprocess of the tokenizer env). It also checks that every Q14/Q32 config equals its Q8 config except for
run_id, milestone and model. On success it writes the sentinel configs/w1_2026_10_13/eval/AUDIT_OK, which the eval
jobs require. Output: results/placebo/w1_injection_audit.json"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from skillrc.config import RunConfig  # noqa: E402
from skillrc.memory import build_memory  # noqa: E402

QWEN_COUNT = ("import json,sys; from transformers import AutoTokenizer; "
              "t=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', local_files_only=True); "
              "print(json.dumps([len(t.encode(s, add_special_tokens=False)) for s in json.load(sys.stdin)]))")
EVAL = ROOT / "configs/w1_2026_10_13/eval"


def pairs():
    out = {}
    for s in ("GPT", "Q8", "Q32"):
        for d in ("D32", "D8"):
            for r in (1, 2, 3):
                p = f"{s}_{d}_R{r}"
                out[p] = (f"W1A1013_{p}_TRUE_Q8", f"W1A1013_{p}_IRR_Q8")
    out["GPTREF"] = ("W1A1013_GPTREF_TRUE_Q8", "W1A1013_GPTREF_IRR_Q8")
    out["GREF"] = ("W1A1013_GREF_TRUE_Q8", "W1A1013_GREF_IRR_Q8")
    out["FOCAL_IRR"] = ("W1B1013_FOCAL_TRUE_Q8", "W1B1013_FOCAL_IRR_Q8")
    out["FOCAL_GENERIC"] = ("W1B1013_FOCAL_TRUE_Q8", "W1B1013_FOCAL_GENERIC_Q8")
    out["FOCAL_IRR2"] = ("W1B1013_FOCAL_TRUE_Q8", "W1B1013_FOCAL_IRR2_Q8")
    out["GREF_GENERIC"] = ("W1A1013_GREF_TRUE_Q8", "W1B1013_GREF_GENERIC_Q8")
    out["GREF_IRR2"] = ("W1A1013_GREF_TRUE_Q8", "W1B1013_GREF_IRR2_Q8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-python", default=os.environ.get("SKILLRC_TOK_PY", sys.executable),
                    help="python with transformers and the Qwen3 tokenizer available (default: this python)")
    ap.add_argument("--out", type=Path, default=ROOT / "results/placebo/w1_injection_audit.json")
    a = ap.parse_args()
    sentinel = EVAL / "AUDIT_OK"
    if sentinel.exists():
        sys.exit(f"{sentinel} already exists; the audit was already passed")
    rows = [json.loads(x) for x in (ROOT / "results/SRC1011_NONE_QWEN.episodes.jsonl").read_text().splitlines() if x.strip()]
    queries = {int(r["task_id"]): r["task_text"] for r in rows if int(r["seed"]) == 0}
    assert sorted(queries) == list(range(134))
    # Q14 / Q32 configs must equal the Q8 config except run_id, milestone, model
    cross = []
    for f in sorted(EVAL.glob("W1*_Q8.yaml")):
        base = yaml.safe_load(f.read_text())
        for rd, model in (("Q14", "qwen3-14b"), ("Q32", "qwen3-32b")):
            g = f.with_name(f.name[:-len("_Q8.yaml")] + f"_{rd}.yaml")
            if not g.exists():
                if not (rd == "Q32" and f.name.startswith("W1B")):
                    cross.append(f"missing {g.name}")
                continue
            other = yaml.safe_load(g.read_text())
            strip = lambda c: {k: v for k, v in c.items() if k not in ("run_id", "milestone", "model")}  # noqa: E731
            if strip(base) != strip(other) or other["model"] != model:
                cross.append(f"{g.name} differs from {f.name} beyond run_id/milestone/model")
    texts, report = [], {}
    for name, (rt, rp) in pairs().items():
        ct, cp = RunConfig.load(str(EVAL / f"{rt}.yaml")), RunConfig.load(str(EVAL / f"{rp}.yaml"))
        mt, mp = build_memory(ct), build_memory(cp)
        idx_t = {it["text"]: k for k, it in enumerate(mt.pool)}
        idx_p = {it["text"]: k for k, it in enumerate(mp.pool)}
        rec = {"n_items": len(mt.pool), "token_budget": mt.token_budget, "order_mismatch_games": 0,
               "o200k_mismatch_games": 0, "items_injected": set(), "pairs": [],
               "payload_ranks_by_true_pool": getattr(cp, "rank_reference_pool", None) == ct.demo_pool}
        for g, q in sorted(queries.items()):
            ctx_t, tok_t = mt.retrieve(q), mt.last_tokens
            ctx_p, tok_p = mp.retrieve(q), mp.last_tokens
            order_t = [idx_t[s] for s in ctx_t.split("\n\n")]
            order_p = [idx_p[s] for s in ctx_p.split("\n\n")]
            rec["order_mismatch_games"] += order_t != order_p
            rec["o200k_mismatch_games"] += tok_t != tok_p
            rec["items_injected"].add(len(order_t))
            rec["pairs"].append((len(texts), len(texts) + 1, tok_t, tok_p))
            texts += [ctx_t, ctx_p]
        report[name] = rec
    env = dict(os.environ)
    qwen = json.loads(subprocess.run([a.tok_python, "-c", QWEN_COUNT], input=json.dumps(texts), text=True,
                                     capture_output=True, check=True, env=env).stdout)
    out = {"cross_reader_config_problems": cross, "pairs": {}}
    for name, rec in report.items():
        qt = [qwen[i] for i, _, _, _ in rec["pairs"]]
        qp = [qwen[j] for _, j, _, _ in rec["pairs"]]
        out["pairs"][name] = {
            "n_items": rec["n_items"], "token_budget_o200k": rec["token_budget"],
            "payload_ranks_by_true_pool": rec["payload_ranks_by_true_pool"],
            "items_injected_per_game": sorted(rec["items_injected"]),
            "o200k_true": sorted({t for _, _, t, _ in rec["pairs"]}), "o200k_payload": sorted({c for _, _, _, c in rec["pairs"]}),
            "o200k_mismatch_games": rec["o200k_mismatch_games"],
            "qwen3_true": sorted(set(qt)), "qwen3_payload": sorted(set(qp)),
            "qwen3_mismatch_games": sum(x != y for x, y in zip(qt, qp)),
            "order_mismatch_games": rec["order_mismatch_games"], "games": len(rec["pairs"])}
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    bad = [n for n, v in out["pairs"].items()
           if v["order_mismatch_games"] or v["o200k_mismatch_games"] or v["qwen3_mismatch_games"]
           or v["items_injected_per_game"] != [v["n_items"]] or not v["payload_ranks_by_true_pool"]]
    for n, v in out["pairs"].items():
        print(f"{n:16s} items {v['n_items']:3d} budget {v['token_budget_o200k']} o200k {v['o200k_true']}/{v['o200k_payload']} "
              f"qwen3 {v['qwen3_true']}/{v['qwen3_payload']} mism order/o200k/qwen3 "
              f"{v['order_mismatch_games']}/{v['o200k_mismatch_games']}/{v['qwen3_mismatch_games']}")
    if bad or cross:
        sys.exit(f"injection audit FAILED: pairs {bad}; config problems {cross}")
    sentinel.write_text(json.dumps({"audit": str(a.out.relative_to(ROOT)), "pairs": len(out["pairs"])}) + "\n")
    print(f"AUDIT OK ({len(out['pairs'])} pairs) -> {sentinel}")


if __name__ == "__main__":
    main()
