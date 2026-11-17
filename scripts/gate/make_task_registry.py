#!/usr/bin/env python3
"""Frozen task registries for the WebShop release-check panels: one JSON line per task with task_id, task_uid and
task_type, from the seeded instruction map (independent of any run). Identities use the adapter's own function, and
the script checks them against the identities computed from logged r2 observations (sessions 0-499).
Writes configs/gate_ws_2026_11/registry_{T,D}.jsonl. Never overwrites."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from skillrc.gate.ingest.skillrc_jsonl import task_uid  # noqa: E402

INSTR = ROOT / "data/summaries/webshop_2026-10-10/instructions_seed0_1k.json"
OUT = ROOT / "configs/gate_ws_2026_11"
PANELS = {"T": range(0, 500), "D": range(520, 1020)}
R2 = ROOT / "results/R2031_P_L_NONE_Q8.episodes.jsonl"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-json", help="JSON file with a list under --key; writes one custom registry")
    ap.add_argument("--key", default="pilot_sessions")
    ap.add_argument("--out")
    a = ap.parse_args()
    panels = PANELS
    if a.sessions_json:
        panels = {"custom": [int(t) for t in json.loads(Path(a.sessions_json).read_text())[a.key]]}
    instr = json.loads(INSTR.read_text())["instructions"]
    tpl = None
    rows = [json.loads(x) for x in R2.read_text().splitlines() if x.strip()]
    r0 = rows[0]
    tpl = r0["task_text"].replace(instr[str(r0["task_id"])]["instruction"], "{INSTR}")
    mism = sum(task_uid(r["task_id"], r["task_text"]) != task_uid(r["task_id"], tpl.replace("{INSTR}", instr[str(r["task_id"])]["instruction"]))
               for r in rows)
    if mism:
        sys.exit(f"{mism} registry identities differ from logged identities")
    for name, ids in panels.items():
        p = Path(a.out) if a.sessions_json else OUT / f"registry_{name}.jsonl"
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
        with p.open("w") as f:
            for t in ids:
                f.write(json.dumps({"task_id": str(t), "task_uid": task_uid(t, tpl.replace("{INSTR}", instr[str(t)]["instruction"])),
                                    "task_type": instr[str(t)]["category"]}) + "\n")
        print(p, len(ids))
    print(f"checked {len(rows)} logged identities: 0 mismatches")


if __name__ == "__main__":
    main()
