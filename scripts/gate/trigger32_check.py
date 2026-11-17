#!/usr/bin/env python3
"""D-phase trigger for the Qwen3-32B extension (PREREG addendum A): runs the frozen 32B analysis spec and launches
panel D only if target C1_32_ship_pub is a valid release decision (not INVALID, not DESCRIPTIVE) whose PRIMARY
component is BLOCK. Writes results/gate_ws32_2026_11/trigger_decision.json and D_TRIGGERED (only if triggered).
Idempotent (decision_io.finalize): a saved decision must equal a fresh evaluation; a missing marker is restored."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from skillrc.gate.engine import run_gate  # noqa: E402
from skillrc.gate.spec import load_spec  # noqa: E402
from skillrc.gate.verdict import State  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_io import finalize  # noqa: E402

OUT = ROOT / "results/gate_ws32_2026_11"


def main():
    dec = OUT / "trigger_decision.json"
    for p in ("T1", "T2"):
        if not (OUT / f"PHASE_{p}_DONE").exists():
            sys.exit(f"phase {p} not complete")
    res = run_gate(load_spec(ROOT / "examples/webshop_release_check_32b/gate.yaml"))
    v = res.targets["C1_32_ship_pub"]
    primary = v.components[0].state if v.components else None
    triggered = v.state not in (State.INVALID, State.DESCRIPTIVE) and primary == State.BLOCK
    out = {"target": "C1_32_ship_pub", "state": v.state.value, "primary_component": primary.value if primary else None,
           "reasons": v.reasons, "primary": res.target_details["C1_32_ship_pub"].get("primary"),
           "triggered": triggered,
           "rule": "valid release decision (not INVALID/DESCRIPTIVE) and primary component BLOCK"}
    rc = finalize(dec, out, ["target", "state", "primary_component", "primary", "triggered"], OUT / "D_TRIGGERED",
                  json.dumps({k: out[k] for k in ("state", "primary_component")}) + "\n", triggered)
    print(json.dumps({k: out[k] for k in ("state", "primary_component", "triggered")}))
    return rc


if __name__ == "__main__":
    sys.exit(main())
