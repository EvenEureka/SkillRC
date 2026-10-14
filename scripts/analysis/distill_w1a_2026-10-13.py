#!/usr/bin/env python3
"""W1-A: balanced replicate subsamples from three sources, each
distilled by a LOCAL distiller (Qwen3-32B "d32" or Qwen3-8B "d8") served by vLLM on localhost.

Pipeline = the source swap's (distill_alfworld_pool.py): serialise() of each successful trajectory, INSIGHT_PROMPT and
_bullets of skillrc/build_skills.py, one call per task type, temperature 0, 512 completion tokens, items tagged [<type>].
Only the subsample and the distiller differ.
  sources   gpt = results/SRC1011_COLLECT_GPT (gpt-4o-mini), q8 = results/SRC1011_COLLECT_QWEN (Qwen3-8B),
            q32 = results/W1A1013_COLLECT_Q32_P01..P10 (Qwen3-32B, new); 140 valid_seen games each, seed 0
  balance   k_t = min(6, smallest success count of type t over the three sources)
  subsample source index s (gpt 1, q8 2, q32 3), replicate r in 1..3: for each type in TYPES order,
            random.Random(20261013 + 10*s + r).sample(successes of the type in game order, k_t), sorted by game id
Outputs (never overwritten): results/placebo/w1a_subsamples.json (once; later runs must reproduce it exactly),
data/pools/w1a_<src>_<dist>_r<r>_insights.jsonl, results/placebo/w1a_<src>_<dist>_r<r>_audit.json.
--base-url must be http://localhost or http://127.0.0.1; the key is a dummy."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from skillrc.build_skills import INSIGHT_PROMPT, _bullets
from skillrc.llm import LLMClient
from skillrc.tokens import count_tokens

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("src_distill", ROOT / "scripts/analysis/distill_alfworld_pool.py")
src = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(src)
TYPES = src.TYPES
SOURCES = {"gpt": (1, ["SRC1011_COLLECT_GPT"]), "q8": (2, ["SRC1011_COLLECT_QWEN"]),
           "q32": (3, [f"W1A1013_COLLECT_Q32_P{k:02d}" for k in range(1, 11)])}
DISTILLERS = {"d32": {"model": "Qwen/Qwen3-32B", "revision": "9216db5781bf21249d130ec9da846c4624c16137", "served_as": "qwen3-32b",
                      "server": "vLLM 0.24, tensor parallel 2, bf16, max-model-len 16384"},
              "d8": {"model": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218", "served_as": "qwen3-8b",
                     "server": "vLLM 0.24, tensor parallel 1, bf16, max-model-len 16384"}}
SEED_BASE, MAX_PER_TYPE, REPS = 20261013, 6, (1, 2, 3)
MANIFEST = ROOT / "results/placebo/w1a_subsamples.json"


def local_only(url: str) -> str:
    if urlparse(url).hostname not in ("localhost", "127.0.0.1"):
        sys.exit(f"refusing non-local base_url {url!r}: this study uses local models only")
    return url


def load_source(name):
    rows = []
    for run in SOURCES[name][1]:
        p = ROOT / f"results/{run}.episodes.jsonl"
        if not p.exists():
            sys.exit(f"missing source run {p}")
        rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    ids = sorted(int(r["task_id"]) for r in rows)
    assert ids == list(range(140)), f"{name}: expected the 140 valid_seen games once each, got {len(ids)} rows"
    rows.sort(key=lambda r: int(r["task_id"]))
    wins = collections.defaultdict(list)
    for r in rows:
        if r["success"] and not r.get("error"):
            wins[r["task_type"]].append(r)
    return rows, wins


def subsamples():
    data = {s: load_source(s) for s in SOURCES}
    k = {t: min(MAX_PER_TYPE, *(len(data[s][1][t]) for s in SOURCES)) for t in TYPES}
    plan = {"k_per_type": k, "trajectories_per_pool": sum(k.values()),
            "successes": {s: {t: len(data[s][1][t]) for t in TYPES} for s in SOURCES},
            "errors": {s: sum(1 for r in data[s][0] if r.get("error")) for s in SOURCES}, "subsamples": {}}
    chosen = {}
    for s, (si, _) in SOURCES.items():
        for r in REPS:
            rng = random.Random(SEED_BASE + 10 * si + r)
            pick = {}
            for t in TYPES:
                pick[t] = sorted(rng.sample(data[s][1][t], k[t]), key=lambda x: int(x["task_id"]))
            chosen[(s, r)] = pick
            plan["subsamples"][f"{s}_r{r}"] = {"seed": SEED_BASE + 10 * si + r,
                                               "games": {t: [int(x["task_id"]) for x in pick[t]] for t in TYPES}}
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        if old != plan:
            sys.exit(f"{MANIFEST} exists and differs from the subsamples computed now")
    else:
        MANIFEST.write_text(json.dumps(plan, indent=2) + "\n")
    return plan, chosen


def make_llm(a, served_as) -> LLMClient:
    return LLMClient(model=served_as, provider="openai", temperature=0.0, max_completion_tokens=512, max_retries=5,
                     base_url=local_only(a.base_url), api_key="EMPTY",
                     extra_body={"chat_template_kwargs": {"enable_thinking": False}})


def call(llm, task_type, trajs):
    resp = llm.complete([{"role": "user", "content": INSIGHT_PROMPT.format(
        task_type=task_type, trajs="\n\n---\n\n".join(src.serialise(r) for r in trajs))}])
    finish = resp.raw.choices[0].finish_reason if resp.raw is not None else None
    return resp, finish


def smoke(a):
    d = DISTILLERS[a.distiller]
    llm = make_llm(a, d["served_as"])
    _, wins = load_source("q8")
    t0 = time.time()
    resp, finish = call(llm, "put", wins["put"][:1])
    bullets = _bullets(resp.text)
    print(f"[smoke {a.distiller}] {len(bullets)} bullets finish={finish} {time.time() - t0:.1f}s\n{resp.text}", flush=True)
    if "<think>" in resp.text or not bullets:
        sys.exit("[smoke] FAILED: thinking leaked or no '- ' bullets parsed")
    print("[smoke] OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--distiller", required=True, choices=list(DISTILLERS))
    ap.add_argument("--sources", default="gpt,q8,q32")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    local_only(a.base_url)
    if a.smoke:
        smoke(a)
        return
    d = DISTILLERS[a.distiller]
    plan, chosen = subsamples()
    print("k per type:", plan["k_per_type"], "successes:", plan["successes"], flush=True)
    todo = [(s, r) for s in a.sources.split(",") for r in REPS]
    for s, r in todo:
        for p in (ROOT / f"data/pools/w1a_{s}_{a.distiller}_r{r}_insights.jsonl", ROOT / f"results/placebo/w1a_{s}_{a.distiller}_r{r}_audit.json"):
            if p.exists():
                sys.exit(f"refusing to overwrite {p}")
    llm = make_llm(a, d["served_as"])
    jobs = {(s, r, t): chosen[(s, r)][t] for s, r in todo for t in TYPES}
    results = {}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(call, llm, key[2], trajs): key for key, trajs in jobs.items()}
        for f in cf.as_completed(futs):
            results[futs[f]] = f.result()
    for s, r in todo:
        name = f"w1a_{s}_{a.distiller}_r{r}"
        items, audit = [], {"pool": name, "source": s, "source_runs": SOURCES[s][1], "replicate": r,
                            "subsample_seed": SEED_BASE + 10 * SOURCES[s][0] + r, "k_per_type": plan["k_per_type"],
                            "distiller": dict(d, enable_thinking=False), "temperature": 0.0, "max_completion_tokens": 512, "types": {}}
        for t in TYPES:
            resp, finish = results[(s, r, t)]
            if "<think>" in resp.text:
                sys.exit(f"thinking leaked for {name}/{t}")
            bullets = _bullets(resp.text)
            if not bullets:
                sys.exit(f"no bullets parsed for {name}/{t}:\n{resp.text}")
            items += [{"text": f"[{t}] {b}", "task_type": t} for b in bullets]
            audit["types"][t] = {"games_used": [int(x["task_id"]) for x in chosen[(s, r)][t]], "n_insights": len(bullets),
                                 "finish_reason": finish, "prompt_tokens": resp.prompt_tokens,
                                 "completion_tokens": resp.completion_tokens, "response": resp.text}
        ctx = "\n\n".join(it["text"] for it in items)
        audit.update({"n_items": len(items), "footprint_o200k": count_tokens(ctx, "gpt-4o-mini"),
                      "body_words": sum(len(it["text"].split()) - 1 for it in items),
                      "finish_reasons": sorted({v["finish_reason"] for v in audit["types"].values()})})
        (ROOT / f"data/pools/{name}_insights.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
        (ROOT / f"results/placebo/{name}_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(f"{name}: {len(items)} items, {audit['footprint_o200k']} o200k, per type "
              f"{[audit['types'][t]['n_insights'] for t in TYPES]}, finish {audit['finish_reasons']}", flush=True)


if __name__ == "__main__":
    main()
