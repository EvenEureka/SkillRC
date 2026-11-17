#!/usr/bin/env python3
"""Prerequisites of every job in the pre-registered chain, in one testable place.

SGE's -hold_jid waits for the previous job to END, not to SUCCEED, so each job checks success markers itself:

  block 1           -
  block 2           BLOCK_1_DONE                                   (8B, results/gate_ws_2026_11)
  32B pilot         BLOCK_1_DONE, BLOCK_2_DONE
  32B T1            PILOT_PASSED                                   (32B, results/gate_ws32_2026_11)
  32B T2            PILOT_PASSED, PHASE_T1_DONE
  32B D1            PHASE_T2_DONE, D_TRIGGERED
  32B D2            PHASE_T2_DONE, D_TRIGGERED, PHASE_D1_DONE

A marker counts only if it is valid: completion markers (`BLOCK_*`, `PHASE_*`) must be JSON whose recorded episode
hashes equal the current files; `PILOT_PASSED` must name both pilot arms with their current episode hashes;
`D_TRIGGERED` must record a primary component BLOCK. A malformed or mismatched marker blocks.

Exit codes: 0 run the job; 10 the job's runs are already complete (valid marker): only finalize; 3 blocked.
usage: prereq.py block <1|2>   |   prereq.py phase <pilot|T1|T2|D1|D2>   [--root DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

R8, R32 = "results/gate_ws_2026_11", "results/gate_ws32_2026_11"
NEEDS = {
    ("block", "1"): [],
    ("block", "2"): [(R8, "BLOCK_1_DONE")],
    ("phase", "pilot"): [(R8, "BLOCK_1_DONE"), (R8, "BLOCK_2_DONE")],
    ("phase", "T1"): [(R32, "PILOT_PASSED")],
    ("phase", "T2"): [(R32, "PILOT_PASSED"), (R32, "PHASE_T1_DONE")],
    ("phase", "D1"): [(R32, "PHASE_T2_DONE"), (R32, "D_TRIGGERED")],
    ("phase", "D2"): [(R32, "PHASE_T2_DONE"), (R32, "D_TRIGGERED"), (R32, "PHASE_D1_DONE")],
}
OWN = {"block": (R8, "BLOCK_{}_DONE"), "phase": (R32, "PHASE_{}_DONE")}


def _sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def marker_problem(path: Path):
    """None if the marker is valid, else the reason."""
    import json
    if not path.exists():
        return "missing"
    try:
        m = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "unreadable"
    if not isinstance(m, dict):
        return "not a JSON object"
    d = path.parent
    try:
        if path.name.startswith(("BLOCK_", "PHASE_")):
            eh = m.get("episodes_sha256")
            if not isinstance(eh, dict) or not eh:
                return "no episode hashes"
            bad = [r for r, h in eh.items() if not (d / f"{r}.episodes.jsonl").exists() or _sha(d / f"{r}.episodes.jsonl") != h]
            return f"episode hashes differ for {bad}" if bad else None
        if path.name == "PILOT_PASSED":
            if set(m) != {"NONE", "TRUE"}:
                return "does not name both pilot arms"
            bad = [a for a, h in m.items() if _sha(d / f"GWS32_P_{a}_Q32.episodes.jsonl") != h]
            return f"pilot episode hashes differ for {bad}" if bad else None
        if path.name == "D_TRIGGERED":
            return None if m.get("primary_component") == "BLOCK" and m.get("state") not in ("INVALID", "DESCRIPTIVE") \
                else "does not record a valid primary BLOCK"
    except OSError as e:
        return f"cannot verify: {e}"
    return None


def check(kind: str, name: str, root: Path) -> int:
    if (kind, name) not in NEEDS:
        print(f"unknown {kind} {name}")
        return 3
    d, pat = OWN[kind]
    own = root / d / pat.format(name)
    if own.exists():
        prob = marker_problem(own)
        if prob:
            print(f"{kind} {name}: own completion marker invalid ({prob}); blocked")
            return 3
        print(f"{kind} {name}: runs already complete; finalize only")
        return 10
    probs = {f"{d}/{m}": marker_problem(root / d / m) for d, m in NEEDS[(kind, name)]}
    probs = {k: v for k, v in probs.items() if v}
    if probs:
        print(f"{kind} {name}: blocked by {probs}")
        return 3
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["block", "phase"])
    ap.add_argument("name")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    a = ap.parse_args()
    sys.exit(check(a.kind, a.name, Path(a.root)))
