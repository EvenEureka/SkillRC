#!/usr/bin/env python3
"""W1-B: a generic ALFWorld rule list written by a LOCAL Qwen3-32B
from a task description only (no trajectory, no game observation, no pool item), then rewritten item by item to the
exact body word counts of each focal pool. The footprint is matched afterwards on CPU (build_generic_control_w1b.py).

  1. base list   one call per task type, temperature 0, 512 tokens, 6 rules (follow-up call if fewer)
  2. pairing     per focal pool and type: the n_t base rules whose word counts are closest to the focal items'
                 (minimum total |difference| over all subsets, ties by generation order), paired in sorted order of word count
  3. exact length the base rule if it already has n words, plus up to 6 distinct exact-length rewrites
                 (temperature 0.8, top-p 0.95, seed per attempt, up to 48 attempts; then 48 more at temperature 1.0)
Outputs (never overwritten): data/pools/w1b_generic_base.jsonl, results/placebo/w1b_generic_raw.json
(base responses, pairing and every rewrite attempt). --base-url must be localhost."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import sys
import zlib
from pathlib import Path
from urllib.parse import urlparse

from skillrc.build_skills import _bullets
from skillrc.llm import LLMClient

ROOT = Path(__file__).resolve().parents[2]
TYPES = ("clean", "cool", "examine", "heat", "put", "puttwo")
FOCAL = {"focal": "data/pools/expel_insights.jsonl", "gref": "data/pools/src1011_qwen_insights.jsonl"}
N_BASE = 6
ENV = ("ALFWorld is a text-based household environment. At the start of an episode the agent is told which receptacles it "
       "can see in the room (for example cabinets, drawers, countertops, shelves, a fridge, a microwave, a sinkbasin, a "
       "stoveburner, a garbagecan, a desklamp), each with a number. Objects lie on or inside receptacles, and closed "
       "receptacles must be opened to see inside. The agent acts with text commands: go to [receptacle], open [receptacle], "
       "close [receptacle], take [object] from [receptacle], put [object] in/on [receptacle], clean [object] with "
       "[receptacle], heat [object] with [receptacle], cool [object] with [receptacle], use [object], examine [object or "
       "receptacle], look, inventory. An episode has a limited number of steps.")
GOALS = {
    "clean": 'the goal reads like "put a clean [object] in [receptacle]" or "clean some [object] and put it in [receptacle]"',
    "cool": 'the goal reads like "put a cool [object] in [receptacle]" or "cool some [object] and put it in [receptacle]"',
    "examine": 'the goal reads like "look at [object] under the desklamp" or "examine the [object] with the desklamp"',
    "heat": 'the goal reads like "put a hot [object] in [receptacle]" or "heat some [object] and put it in [receptacle]"',
    "put": 'the goal reads like "put a [object] in [receptacle]" or "put some [object] on [receptacle]"',
    "puttwo": 'the goal reads like "put two [object] in [receptacle]" or "find two [object] and put them in [receptacle]"'}
BASE_PROMPT = (ENV + "\n\nTask type '{task_type}': {goal}\n\n"
               "Write {n} concise, GENERAL rules (heuristics) that would help an agent solve this task type. Each rule: one "
               "imperative line, no specific object numbers or room layouts. Output ONLY a bullet list, each line starting with '- '.")
MORE_PROMPT = (BASE_PROMPT + "\n\nYou already wrote these rules:\n{given}\n\nWrite {k} further rules, different from those. "
               "Output ONLY a bullet list, each line starting with '- '.")
REWRITE_PROMPT = ("Rewrite the following rule for ALFWorld '{task_type}' tasks so that it has exactly {n} words (count every "
                  "whitespace-separated token as a word). Keep its meaning; no specific object numbers or room layouts. "
                  "Output ONLY the rewritten rule on one line, without a leading dash.\n\nRule: {rule}")


def local_only(url):
    if urlparse(url).hostname not in ("localhost", "127.0.0.1"):
        sys.exit(f"refusing non-local base_url {url!r}: local models only")
    return url


def client(a, temperature, max_tokens, extra=None):
    body = {"chat_template_kwargs": {"enable_thinking": False}, **(extra or {})}
    return LLMClient(model="qwen3-32b", provider="openai", temperature=temperature, max_completion_tokens=max_tokens,
                     max_retries=5, base_url=local_only(a.base_url), api_key="EMPTY", extra_body=body)


def nwords(s):
    return len(s.split())


def clean_line(text):
    for ln in (text or "").splitlines():
        s = ln.strip().lstrip("-*• ").strip().strip('"').strip()
        if s:
            return s
    return ""


def base_list(a, raw):
    llm = client(a, 0.0, 512)
    out = {}
    for t in TYPES:
        resp = llm.complete([{"role": "user", "content": BASE_PROMPT.format(task_type=t, goal=GOALS[t], n=N_BASE)}])
        rules = _bullets(resp.text)
        entry = {"response": resp.text, "follow_up": None}
        if len(rules) < N_BASE:
            given = "\n".join(f"- {r}" for r in rules)
            resp2 = llm.complete([{"role": "user", "content": MORE_PROMPT.format(
                task_type=t, goal=GOALS[t], n=N_BASE, given=given, k=N_BASE - len(rules))}])
            entry["follow_up"] = resp2.text
            rules += _bullets(resp2.text)
        if "<think>" in resp.text or len(rules) < N_BASE:
            sys.exit(f"base list for {t} failed: {len(rules)} rules\n{resp.text}")
        out[t] = rules[:N_BASE]
        entry["rules"] = out[t]
        raw["base"][t] = entry
        print(f"[base] {t}: {[nwords(r) for r in out[t]]} words", flush=True)
    return out


def pair(base, focal_items):
    """per type: choose the n_t base rules closest in word count, pair in sorted order of word count."""
    plan = []
    for t in TYPES:
        idx = [i for i, it in enumerate(focal_items) if it["task_type"] == t]
        lens = sorted((nwords(focal_items[i]["text"]) - 1, i) for i in idx)
        best = None
        for combo in itertools.combinations(range(len(base[t])), len(idx)):
            bl = sorted((nwords(base[t][j]), j) for j in combo)
            cost = sum(abs(x[0] - y[0]) for x, y in zip(bl, lens))
            key = (cost, combo)
            if best is None or key < best[0]:
                best = (key, bl)
        for (bn, j), (n, i) in zip(best[1], lens):
            plan.append({"type": t, "focal_index": i, "n": n, "base_index": j, "base_rule": base[t][j], "base_words": bn})
    return sorted(plan, key=lambda p: p["focal_index"])


def variants(a, job):
    """exact-length variants for one (type, n, base rule)"""
    t, n, rule = job["type"], job["n"], job["base_rule"]
    found = [rule] if nwords(rule) == n else []
    attempts = []
    for phase, temp in ((0, 0.8), (1, 1.0)):
        if phase == 1 and found:
            break
        for k in range(48):
            if len(found) >= 6:
                break
            seed = zlib.crc32(f"{t}|{n}|{rule}|{phase}|{k}".encode()) % 2**31
            llm = client(a, temp, 96, {"seed": seed, "top_p": 0.95})
            resp = llm.complete([{"role": "user", "content": REWRITE_PROMPT.format(task_type=t, n=n, rule=rule)}])
            s = clean_line(resp.text)
            ok = nwords(s) == n and "<think>" not in resp.text and "\n" not in s
            attempts.append({"phase": phase, "temperature": temp, "seed": seed, "text": s, "words": nwords(s), "ok": ok})
            if ok and s not in found:
                found.append(s)
    return {"variants": found, "attempts": attempts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    local_only(a.base_url)
    out_base, out_raw = ROOT / "data/pools/w1b_generic_base.jsonl", ROOT / "results/placebo/w1b_generic_raw.json"
    for p in (out_base, out_raw):
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
    raw = {"distiller": {"model": "Qwen/Qwen3-32B", "revision": "9216db5781bf21249d130ec9da846c4624c16137", "enable_thinking": False},
           "prompts": {"base": BASE_PROMPT, "more": MORE_PROMPT, "rewrite": REWRITE_PROMPT, "goals": GOALS},
           "base": {}, "focal": {}}
    base = base_list(a, raw)
    jobs, plans = {}, {}
    for f, path in FOCAL.items():
        items = [json.loads(x) for x in (ROOT / path).read_text().splitlines() if x.strip()]
        plans[f] = pair(base, items)
        for p in plans[f]:
            jobs.setdefault((p["type"], p["n"], p["base_rule"]), p)
    res = {}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(variants, a, p): key for key, p in jobs.items()}
        for fu in cf.as_completed(futs):
            res[futs[fu]] = fu.result()
    failed = []
    for f, plan in plans.items():
        for p in plan:
            r = res[(p["type"], p["n"], p["base_rule"])]
            p["variants"] = r["variants"]
            p["n_attempts"] = len(r["attempts"])
            if not r["variants"]:
                failed.append((f, p["focal_index"], p["type"], p["n"]))
        raw["focal"][f] = {"pool": FOCAL[f], "plan": plan}
    raw["attempts"] = {f"{k[0]}|{k[1]}|{k[2]}": v["attempts"] for k, v in res.items()}
    raw["failed_items"] = failed
    out_base.write_text("".join(json.dumps({"text": f"[{t}] {r}", "task_type": t}, ensure_ascii=False) + "\n"
                                for t in TYPES for r in base[t]))
    out_raw.write_text(json.dumps(raw, indent=2) + "\n")
    for f, plan in plans.items():
        print(f"[{f}] items {len(plan)}, variants per item {[len(p['variants']) for p in plan]}", flush=True)
    if failed:
        sys.exit(f"no exact-length variant for {failed}: manual decision needed (record as a deviation)")
    print("GENERIC_RULES_DONE", flush=True)


if __name__ == "__main__":
    main()
