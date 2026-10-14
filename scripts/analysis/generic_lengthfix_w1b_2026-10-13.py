#!/usr/bin/env python3
"""W1-B deviation D1: exact-length variants of the generic base rules
by a word-count feedback loop, because single-shot rewrites by Qwen3-32B came out 3-4 words short (65 of 5,189 exact).

The base list (temperature 0, task description only) and the pairing in results/placebo/w1b_generic_raw.json are kept
unchanged; only the length normalisation of step 3 changes. For every paired item (type, word count n, base rule):
  a chain starts with the planned REWRITE_PROMPT (temperature 0.7, top-p 0.95, seed per chain); while the reply
  has m != n words, the same conversation gets "That rule has m words. It must have exactly n words. Add|Remove k
  word(s) ..." (at most 5 corrections). Up to 12 chains per item; up to 6 distinct exact variants are kept (the base
  rule counts if it already has n words, and so do exact variants found in the first pass).
Local Qwen3-32B only (--base-url must be localhost). Output (never overwritten): results/placebo/w1b_generic_lengthfix.json"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import zlib
from pathlib import Path
from urllib.parse import urlparse

from skillrc.llm import LLMClient

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/placebo/w1b_generic_raw.json"
OUT = ROOT / "results/placebo/w1b_generic_lengthfix.json"
FIX = ("That rule has {m} words. It must have exactly {n} words. {verb} {k} word{s} while keeping its meaning, and output "
       "ONLY the corrected rule on one line, without a leading dash.")
CHAINS, ROUNDS, KEEP = 12, 5, 6


def local_only(url):
    if urlparse(url).hostname not in ("localhost", "127.0.0.1"):
        sys.exit(f"refusing non-local base_url {url!r}: local models only")
    return url


def clean_line(text):
    for ln in (text or "").splitlines():
        s = ln.strip().lstrip("-*• ").strip().strip('"').strip()
        if s:
            return s
    return ""


def chain(a, prompt, t, n, rule, c):
    seed = zlib.crc32(f"fix|{t}|{n}|{rule}|{c}".encode()) % 2**31
    llm = LLMClient(model="qwen3-32b", provider="openai", temperature=0.7, max_completion_tokens=96, max_retries=5,
                    base_url=local_only(a.base_url), api_key="EMPTY",
                    extra_body={"seed": seed, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": False}})
    msgs = [{"role": "user", "content": prompt.format(task_type=t, n=n, rule=rule)}]
    log = []
    for r in range(ROUNDS + 1):
        resp = llm.complete(msgs)
        s = clean_line(resp.text)
        m = len(s.split())
        log.append({"round": r, "text": s, "words": m})
        if m == n and "<think>" not in resp.text:
            return {"seed": seed, "rounds": log, "ok": True, "text": s}
        k = abs(n - m)
        msgs += [{"role": "assistant", "content": s},
                 {"role": "user", "content": FIX.format(m=m, n=n, verb="Add" if m < n else "Remove", k=k, s="" if k == 1 else "s")}]
    return {"seed": seed, "rounds": log, "ok": False, "text": None}


def item(a, prompt, key, prior):
    t, n, rule = key
    found = list(prior)
    chains = []
    for c in range(CHAINS):
        if len(found) >= KEEP:
            break
        r = chain(a, prompt, t, n, rule, c)
        chains.append(r)
        if r["ok"] and r["text"] not in found:
            found.append(r["text"])
    return {"variants": found[:KEEP], "chains": chains}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    local_only(a.base_url)
    if OUT.exists():
        sys.exit(f"refusing to overwrite {OUT}")
    raw = json.loads(RAW.read_text())
    prompt = raw["prompts"]["rewrite"]
    keys = {}
    for f, d in raw["focal"].items():
        for p in d["plan"]:
            k = (p["type"], p["n"], p["base_rule"])
            keys.setdefault(k, list(p["variants"]))
    res = {}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(item, a, prompt, k, prior): k for k, prior in keys.items()}
        for fu in cf.as_completed(futs):
            res[futs[fu]] = fu.result()
    out = {"deviation": "D1", "fix_prompt": FIX, "chains_per_item": CHAINS,
           "max_corrections": ROUNDS, "temperature": 0.7, "top_p": 0.95, "keep": KEEP,
           "items": [{"type": k[0], "n": k[1], "base_rule": k[2], **v} for k, v in res.items()]}
    failed = [(k[0], k[1]) for k, v in res.items() if not v["variants"]]
    out["failed_items"] = failed
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("variants per item:", sorted(len(v["variants"]) for v in res.values()), flush=True)
    if failed:
        sys.exit(f"still no exact variant for {failed}")
    print("LENGTHFIX_DONE", flush=True)


if __name__ == "__main__":
    main()
