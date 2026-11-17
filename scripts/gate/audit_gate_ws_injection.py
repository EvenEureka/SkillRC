#!/usr/bin/env python3
"""Injection audit for the WebShop release-check configs, run before any episode.

For every memory config in configs/gate_ws_2026_11 it builds the harness memory exactly as the runner does and
retrieves for the initial observation of every session of that config. Requirements:
  * all 30 items are injected (whole pool) and the o200k length equals PUB's (524);
  * a payload config injects the same item indices in the same order as the TRUE config of its panel;
  * the injected block has PUB's Qwen3 token count (529) for every session;
  * the payload differs from PUB exactly at its declared lines (data/pools/gate_ws_2026_11/payload_audit.json).
Writes results/gate_ws_2026_11/injection_audit.json and, on success, configs/gate_ws_2026_11/AUDIT_OK.
The initial observation is rebuilt from the seeded instruction map with the template of a logged r2 observation;
ranking uses the TRUE pool's text, so order equality does not depend on the query wording.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from skillrc.config import RunConfig  # noqa: E402
from skillrc.memory import build_memory  # noqa: E402

CFG = ROOT / "configs/gate_ws_2026_11"
OUT = ROOT / "results/gate_ws_2026_11/injection_audit.json"
INSTR = ROOT / "data/summaries/webshop_2026-10-10/instructions_seed0_1k.json"
R2_LOG = ROOT / "results/R2031_P_L_NONE_Q8.episodes.jsonl"
PAYLOAD_AUDIT = ROOT / "data/pools/gate_ws_2026_11/payload_audit.json"


def template():
    r = json.loads(R2_LOG.read_text().splitlines()[0])
    instr = json.loads(INSTR.read_text())["instructions"][str(r["task_id"])]["instruction"]
    t = r["task_text"]
    assert instr in t, "instruction not found in the logged observation"
    return t.replace(instr, "{INSTR}")


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    q3 = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731
    sentinel = CFG / "AUDIT_OK"
    if sentinel.exists():
        sys.exit(f"{sentinel} already exists")
    tpl = template()
    instr = json.loads(INSTR.read_text())["instructions"]
    declared = {k: v["lines"] for k, v in json.loads(PAYLOAD_AUDIT.read_text())["payloads"].items()}
    pub = [json.loads(x)["text"] for x in (ROOT / "data/pools/webshop1010_insights.jsonl").read_text().splitlines() if x.strip()]
    q3_target = q3("\n\n".join(pub))
    report, bad = {"template": tpl, "qwen3_target": q3_target, "configs": {}}, []
    true_cfg = {"T": RunConfig.load(str(CFG / "GWS_T_TRUE_R1_Q8.yaml")), "D": RunConfig.load(str(CFG / "GWS_D_TRUE_R1_Q8.yaml"))}
    true_mem = {p: build_memory(c) for p, c in true_cfg.items()}
    for f in sorted(CFG.glob("GWS_*.yaml")):
        c = RunConfig.load(str(f))
        if c.memory == "none":
            continue
        panel = f.name.split("_")[1]
        m = build_memory(c)
        mt = true_mem[panel]
        idx_t = {it["text"]: k for k, it in enumerate(mt.pool)}
        idx_m = {it["text"]: k for k, it in enumerate(m.pool)}
        kind = f.name.split("_")[2]
        diff_lines = [i + 1 for i, (a, b) in enumerate(zip(pub, [it["text"] for it in m.pool])) if a != b]
        exp_lines = [] if kind == "TRUE" else declared[kind]
        rec = {"n_tasks": len(c.task_ids), "order_mismatch": 0, "n_items": set(), "o200k": set(), "qwen3": set(),
               "diff_lines": diff_lines, "declared_lines": exp_lines}
        for t in c.task_ids:
            q = tpl.replace("{INSTR}", instr[str(t)]["instruction"])
            ctx_t, ctx = mt.retrieve(q), m.retrieve(q)
            ot = [idx_t[s] for s in ctx_t.split("\n\n")]
            om = [idx_m[s] for s in ctx.split("\n\n")]
            rec["order_mismatch"] += ot != om
            rec["n_items"].add(len(om))
            rec["o200k"].add(m.last_tokens)
            rec["qwen3"].add(q3(ctx))
        for k in ("n_items", "o200k", "qwen3"):
            rec[k] = sorted(rec[k])
        ok = (rec["order_mismatch"] == 0 and rec["n_items"] == [30] and rec["o200k"] == [524]
              and rec["qwen3"] == [q3_target] and diff_lines == sorted(exp_lines))
        rec["ok"] = ok
        report["configs"][f.name] = rec
        if not ok:
            bad.append(f.name)
        print(f.name, "OK" if ok else "FAIL", {k: rec[k] for k in ("order_mismatch", "n_items", "o200k", "qwen3")})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({**report, "failed": bad}, indent=1) + "\n")
    if bad:
        sys.exit(f"audit failed: {bad}")
    sentinel.write_text("injection audit passed; see results/gate_ws_2026_11/injection_audit.json\n")
    print("AUDIT_OK")


if __name__ == "__main__":
    main()
