#!/usr/bin/env python3
"""Pool-replication study: distil ALFWorld insight pools from the Qwen3-8B valid_seen source collection
(SRC1011_COLLECT_QWEN) with a LOCAL Qwen3-32B served by vLLM.

Everything except the distiller and the subsample is the pipeline of distill_alfworld_pool.py:
serialise() for the trajectories, INSIGHT_PROMPT and _bullets of skillrc/build_skills.py, one call per task type,
six successful trajectories per type in game order, temperature 0, 512 completion tokens, items tagged [<type>].

Pools:  q32r0 = the original subsample (first six successes per type, the games of src1011_qwen_pool_audit.json)
        q32r<r>, r >= 1 = random six per type, random.Random(20261012 + r), types in TYPES order, sorted by game id
No hosted endpoint can be reached: --base-url must be http://localhost or http://127.0.0.1, the key is a dummy.
Outputs (never overwritten): data/pools/poolrep_<pool>_insights.jsonl, results/placebo/poolrep_<pool>_audit.json"""
from __future__ import annotations

import argparse
import collections
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
SOURCE_RUN = "SRC1011_COLLECT_QWEN"
SEED_BASE = 20261012
PER_TYPE = 6


def local_only(url: str) -> str:
    host = urlparse(url).hostname
    if host not in ("localhost", "127.0.0.1"):
        sys.exit(f"refusing non-local base_url {url!r}: this study uses local models only")
    return url


def successes():
    rows = [json.loads(x) for x in (ROOT / f"results/{SOURCE_RUN}.episodes.jsonl").read_text().splitlines() if x.strip()]
    rows.sort(key=lambda r: int(r["task_id"]))
    wins = collections.defaultdict(list)
    for r in rows:
        if r["success"] and not r.get("error"):
            wins[r["task_type"]].append(r)
    return rows, wins


def subsample(pool: str, wins) -> dict:
    if pool == "q32r0":
        ref = json.loads((ROOT / "results/placebo/src1011_qwen_pool_audit.json").read_text())["types"]
        out = {t: wins[t][:PER_TYPE] for t in TYPES}
        for t in TYPES:  # must be exactly the games the original GPT-distilled pool used
            assert [int(r["task_id"]) for r in out[t]] == ref[t]["games_used"], (t, ref[t]["games_used"])
        return out
    r = int(pool.removeprefix("q32r"))
    rng = random.Random(SEED_BASE + r)
    out = {}
    for t in TYPES:
        chosen = rng.sample(wins[t], PER_TYPE)
        out[t] = sorted(chosen, key=lambda x: int(x["task_id"]))
    return out


def make_llm(a) -> LLMClient:
    return LLMClient(model=a.model, provider="openai", temperature=0.0, max_completion_tokens=512, max_retries=5,
                     base_url=local_only(a.base_url), api_key="EMPTY",
                     extra_body={"chat_template_kwargs": {"enable_thinking": False}})


def call(llm, task_type, chosen):
    resp = llm.complete([{"role": "user", "content": INSIGHT_PROMPT.format(
        task_type=task_type, trajs="\n\n---\n\n".join(src.serialise(r) for r in chosen))}])
    finish = resp.raw.choices[0].finish_reason if resp.raw is not None else None
    return resp, finish


def smoke(a):
    """Two tiny calls through the full path (prompt, local server, thinking off, bullet parser)."""
    llm = make_llm(a)
    _, wins = successes()
    for t in ("put", "clean"):
        t0 = time.time()
        resp, finish = call(llm, t, wins[t][:1])
        bullets = _bullets(resp.text)
        print(f"[smoke] {t}: {len(bullets)} bullets, finish={finish}, prompt_tokens={resp.prompt_tokens}, "
              f"completion_tokens={resp.completion_tokens}, {time.time() - t0:.1f}s", flush=True)
        print(resp.text, flush=True)
        if "<think>" in resp.text or not bullets:
            sys.exit("[smoke] FAILED: thinking leaked or no '- ' bullets parsed")
    print("[smoke] OK", flush=True)


def build(a, pool: str):
    out = ROOT / f"data/pools/poolrep_{pool}_insights.jsonl"
    aud_path = ROOT / f"results/placebo/poolrep_{pool}_audit.json"
    for p in (out, aud_path):
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
    llm = make_llm(a)
    rows, wins = successes()
    chosen_by_type = subsample(pool, wins)
    items = []
    audit = {"pool": pool, "source_run": SOURCE_RUN, "episodes": len(rows), "successes": sum(len(v) for v in wins.values()),
             "distiller": {"model": "Qwen/Qwen3-32B", "revision": a.revision, "served_as": a.model,
                           "server": "vLLM 0.24 (local, tensor parallel 2)", "enable_thinking": False},
             "temperature": 0.0, "max_completion_tokens": 512, "per_type": PER_TYPE,
             "subsample": "first six per type in game order (original)" if pool == "q32r0" else
                          f"random.Random({SEED_BASE + int(pool.removeprefix('q32r'))}).sample per type in {list(TYPES)} order, sorted by game id",
             "types": {}}
    for t in TYPES:
        chosen = chosen_by_type[t]
        resp, finish = call(llm, t, chosen)
        if "<think>" in resp.text:
            sys.exit(f"thinking leaked for {pool}/{t}")
        bullets = _bullets(resp.text)
        if not bullets:
            sys.exit(f"no bullets parsed for {pool}/{t}:\n{resp.text}")
        items += [{"text": f"[{t}] {b}", "task_type": t} for b in bullets]
        audit["types"][t] = {"successes": len(wins[t]), "games_used": [int(r["task_id"]) for r in chosen],
                             "n_insights": len(bullets), "finish_reason": finish, "prompt_tokens": resp.prompt_tokens,
                             "completion_tokens": resp.completion_tokens, "response": resp.text}
        print(pool, t, {k: v for k, v in audit["types"][t].items() if k != "response"}, flush=True)
    ctx = "\n\n".join(it["text"] for it in items)
    audit.update({"n_items": len(items), "footprint_o200k": count_tokens(ctx, "gpt-4o-mini"),
                  "body_words": sum(len(it["text"].split()) - 1 for it in items)})
    out.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    aud_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"{pool}: {len(items)} insights, {audit['footprint_o200k']} o200k tokens -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--revision", default="9216db5781bf21249d130ec9da846c4624c16137")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pools", default="q32r0,q32r1,q32r2,q32r3")
    a = ap.parse_args()
    local_only(a.base_url)
    if a.smoke:
        smoke(a)
        return
    for pool in a.pools.split(","):
        build(a, pool)


if __name__ == "__main__":
    main()
