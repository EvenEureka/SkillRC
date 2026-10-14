#!/usr/bin/env python3
"""Configs for the 2026-10-10 phase-2 runs (the five-reader study).

A  the 74 unseen games outside the 60-game rerun: {none,true,placebo,irrelevant} x {gpt-4o-mini, Qwen3-8B}
   (together with the morning rerun this completes all 134 games in one day; the 74 alone are an
   out-of-sample replication)
B  a second gpt-4o-mini run of all 134 games x 4 conditions (hosted run-to-run noise floor)
C  three further executors on all 134 games x 4 conditions: gpt-4.1-mini-2025-04-14, Llama-3.1-8B-Instruct,
   Qwen3-14B (thinking off)
Everything except run_id, task_ids and the executor fields is copied from configs/rerun_2026_10_10/."""
import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "configs/rerun_2026_10_10"
OUT = ROOT / "configs/phase2_2026_10_10"
OUT.mkdir(exist_ok=True)
gate = json.loads((ROOT / "data/summaries/rerun_2026-10-10/frozen_tasks_rerun_2026-10-10.json").read_text())["task_ids"]
assert len(gate) == 60
complement = [i for i in range(134) if i not in set(gate)]
assert len(complement) == 74
ALL = list(range(134))
EXEC = {
    "GPT": {"base": "gpt"},
    "QWEN": {"base": "qwen"},
    "GPT41MINI": {"base": "gpt", "model": "gpt-4.1-mini-2025-04-14"},
    "LLAMA8B": {"base": "qwen", "model": "llama-3.1-8b", "chat_template_kwargs": None},
    "QWEN14B": {"base": "qwen", "model": "qwen3-14b"},
}
PLAN = [("P2A", ["GPT", "QWEN"], complement), ("P2B", ["GPT"], ALL), ("P2C", ["GPT41MINI", "LLAMA8B", "QWEN14B"], ALL)]
n = 0
for part, executors, ids in PLAN:
    for ex in executors:
        spec = EXEC[ex]
        for cond in ("none", "true", "placebo", "irrelevant"):
            cfg = yaml.safe_load((SRC / f"{cond}_{spec['base']}.yaml").read_text())
            cfg = copy.deepcopy(cfg)
            cfg["run_id"] = f"{part}_{cond.upper()}_{ex}"
            cfg["milestone"] = f"phase2-2026-10-10-{part}"
            cfg["task_ids"] = ids
            for k in ("model", "chat_template_kwargs"):
                if k in spec:
                    if spec[k] is None:
                        cfg.pop(k, None)
                    else:
                        cfg[k] = spec[k]
            (OUT / f"{part}_{cond}_{ex}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
            n += 1
print("wrote", n, "configs to", OUT)
