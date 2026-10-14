#!/usr/bin/env python3
"""Configs for W1-A (crossover study) and W1-B (generic-rule study).

  collect  Qwen3-32B source collection on the 140 valid_seen games, seed 0, in 10 shards of 14 games:
           configs/source_2026_10_11/SRC1011_COLLECT_QWEN.yaml with only run_id, milestone, model and task_ids changed.
  eval     Reader configs for Qwen3-8B / Qwen3-14B / Qwen3-32B: every setting copied from
           configs/poolrep_2026_10_12/PRP1012_{NONE,GREF_TRUE}_QWEN.yaml; only run_id, milestone, model, demo_pool,
           token_budget and rank_reference_pool change. Every pool and payload is injected whole (token_budget = the true
           pool's o200k footprint, which every payload must equal); payload arms rank by the true pool.
           Writes configs/w1_2026_10_13/eval/<RUN>.yaml and the per-server job lists jobs_<READER>_S<k>.txt.
Local models only: every config must point to localhost and use BM25; nothing here calls a model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skillrc.tokens import count_tokens  # noqa: E402

OUT = ROOT / "configs/w1_2026_10_13"
READERS = {"Q8": "qwen3-8b", "Q14": "qwen3-14b", "Q32": "qwen3-32b"}
SOURCES, DISTILLERS, REPS = ("gpt", "q8", "q32"), ("d32", "d8"), (1, 2, 3)


def w1a_pools():
    """arm prefix -> (true pool, control payload)"""
    pools = {}
    for s in SOURCES:
        for d in DISTILLERS:
            for r in REPS:
                stem = f"data/pools/w1a_{s}_{d}_r{r}"
                pools[f"{s.upper()}_{d.upper()}_R{r}"] = (f"{stem}_insights.jsonl", f"{stem}_irrelevant.jsonl")
    pools["GPTREF"] = ("data/pools/src1011_gpt_insights.jsonl", "data/pools/w1a_gptref_irrelevant.jsonl")
    pools["GREF"] = ("data/pools/src1011_qwen_insights.jsonl", "data/pools/poolrep_gref_irrelevant.jsonl")
    return pools


# W1-B: arm -> (true pool used for ranking and budget, payload)
W1B = {"FOCAL_TRUE": ("data/pools/expel_insights.jsonl", None),
       "FOCAL_IRR": ("data/pools/expel_insights.jsonl", "data/pools/expel_insights_irrelevant_coherent.jsonl"),
       "FOCAL_GENERIC": ("data/pools/expel_insights.jsonl", "data/pools/w1b_focal_generic.jsonl"),
       "FOCAL_IRR2": ("data/pools/expel_insights.jsonl", "data/pools/w1b_focal_astronomy.jsonl"),
       "GREF_GENERIC": ("data/pools/src1011_qwen_insights.jsonl", "data/pools/w1b_gref_generic.jsonl"),
       "GREF_IRR2": ("data/pools/src1011_qwen_insights.jsonl", "data/pools/w1b_gref_astronomy.jsonl")}


def footprint(path: str) -> int:
    items = [json.loads(x) for x in (ROOT / path).read_text().splitlines() if x.strip()]
    return count_tokens("\n\n".join(it["text"] for it in items), "gpt-4o-mini")


def write(path: Path, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if yaml.safe_load(path.read_text()) != cfg:
            sys.exit(f"refusing to overwrite a different {path}")
        return
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=110))


def check_local(cfg):
    assert cfg["api_key"] == "EMPTY" and cfg["provider"] == "openai"
    assert cfg["base_url"].startswith(("http://localhost", "http://127.0.0.1")), cfg["base_url"]
    assert cfg.get("retriever", "bm25") == "bm25" or cfg.get("memory") == "none"


def collect():
    base = yaml.safe_load((ROOT / "configs/source_2026_10_11/SRC1011_COLLECT_QWEN.yaml").read_text())
    assert base["alfworld_split"] == "eval_in_distribution" and base["task_ids"] == list(range(140)) and base["seeds"] == [0]
    names = []
    for k in range(10):
        cfg = dict(base, run_id=f"W1A1013_COLLECT_Q32_P{k + 1:02d}", milestone="w1a-2026-10-13-collect", model="qwen3-32b",
                   task_ids=list(range(14 * k, 14 * k + 14)))
        check_local(cfg)
        p = OUT / "collect" / f"{cfg['run_id']}.yaml"
        write(p, cfg)
        names.append(str(p.relative_to(ROOT)))
    print(":".join(names))


def eval_():
    src = ROOT / "configs/poolrep_2026_10_12"
    none = yaml.safe_load((src / "PRP1012_NONE_QWEN.yaml").read_text())
    true = yaml.safe_load((src / "PRP1012_GREF_TRUE_QWEN.yaml").read_text())
    for c in (none, true):
        assert c["task_ids"] == list(range(134)) and c["seeds"] == [0, 1, 5] and c["model"] == "qwen3-8b"
    budgets, missing = {}, []
    arms = {}  # arm -> (study, cfg-without-model)
    arms["NONE"] = ("W1A", dict(none))
    for prefix, (tp, cp) in w1a_pools().items():
        if not (ROOT / tp).exists() or not (ROOT / cp).exists():
            missing += [p for p in (tp, cp) if not (ROOT / p).exists()]
            continue
        b = footprint(tp)
        assert footprint(cp) == b, f"{prefix}: control footprint {footprint(cp)} != pool {b}"
        budgets[prefix] = b
        arms[f"{prefix}_TRUE"] = ("W1A", dict(true, demo_pool=tp, token_budget=b))
        arms[f"{prefix}_IRR"] = ("W1A", dict(true, demo_pool=cp, token_budget=b, rank_reference_pool=tp))
    if "Q8_D32_R1_TRUE" in arms:
        arms["Q8_D32_R1_TRUEDUP"] = ("W1A", dict(arms["Q8_D32_R1_TRUE"][1]))
    for arm, (tp, pp) in W1B.items():
        need = [p for p in (tp, pp) if p and not (ROOT / p).exists()]
        if need:
            missing += need
            continue
        b = footprint(tp)
        cfg = dict(true, demo_pool=tp, token_budget=b)
        if pp:
            assert footprint(pp) == b, f"{arm}: payload footprint {footprint(pp)} != pool {b}"
            cfg = dict(cfg, demo_pool=pp, rank_reference_pool=tp)
        cfg.pop("rank_reference_pool", None) if not pp else None
        arms[arm] = ("W1B", cfg)
    if missing:
        sys.exit("missing pool/payload files (build them first):\n  " + "\n  ".join(sorted(set(missing))))
    s1 = (["NONE", "GREF_TRUE", "GREF_IRR", "GPTREF_TRUE", "GPTREF_IRR"]
          + [f"GPT_{d.upper()}_R{r}_{c}" for d in DISTILLERS for r in REPS for c in ("TRUE", "IRR")]
          + ["Q8_D32_R1_TRUEDUP"] + list(W1B))
    s2 = [f"{s.upper()}_{d.upper()}_R{r}_{c}" for s in ("q8", "q32") for d in DISTILLERS for r in REPS for c in ("TRUE", "IRR")]
    w1a_arms = [a for a in arms if arms[a][0] == "W1A"]
    assert sorted(w1a_arms + list(W1B)) == sorted(s1 + s2) and len(w1a_arms) == 42, (len(w1a_arms), len(s1), len(s2))
    jobs = {}
    for reader, model in READERS.items():
        for arm, (study, cfg0) in arms.items():
            if study == "W1B" and reader == "Q32":
                continue
            run = f"{study}1013_{arm}_{reader}"
            cfg = dict(cfg0, run_id=run, milestone=f"{study.lower()}-2026-10-13", model=model)
            check_local(cfg)
            assert cfg["task_ids"] == list(range(134)) and cfg["seeds"] == [0, 1, 5]
            write(OUT / "eval" / f"{run}.yaml", cfg)
        if reader == "Q32":
            lists = {"S1": [a for a in s1 + s2 if a not in W1B]}
        else:
            lists = {"S1": s1, "S2": s2}
        for srv, arm_list in lists.items():
            paths = []
            for arm in arm_list:
                study = arms[arm][0]
                paths.append(f"configs/w1_2026_10_13/eval/{study}1013_{arm}_{reader}.yaml")
            jobs[f"{reader}_{srv}"] = paths
            f = OUT / "eval" / f"jobs_{reader}_{srv}.txt"
            txt = ":".join(paths) + "\n"
            if f.exists() and f.read_text() != txt:
                sys.exit(f"refusing to overwrite a different {f}")
            f.write_text(txt)
    print("budgets (o200k):", json.dumps(budgets))
    for k, v in jobs.items():
        print(k, len(v), "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["collect", "eval"])
    a = ap.parse_args()
    collect() if a.what == "collect" else eval_()


if __name__ == "__main__":
    main()
