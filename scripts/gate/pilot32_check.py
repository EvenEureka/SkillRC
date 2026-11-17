#!/usr/bin/env python3
"""Qwen3-32B premise pilot decision (PREREG addendum A). Operational screen, not evidence for any claim.

Validity first: both pilot arms must be exactly the frozen pilot (task set = schedule.json's pilot sessions,
identities from registry_pilot.jsonl, seed 0, repetition 1, no duplicate or out-of-grid attempts), each manifest must
equal its frozen config, carry complete provenance, and the two arms may differ only in the memory keys.
Then the screens, reported separately:
  operational  every cell scored and <= 2% of cells with an infrastructure failure, per arm
  score        the NONE arm averages >= 0.30
Writes results/gate_ws32_2026_11/pilot_decision.json and, only if all pass, PILOT_PASSED (with the episode hashes).
Idempotent (decision_io.finalize): a saved decision must equal a fresh evaluation; a missing marker is restored."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from skillrc.config import RunConfig  # noqa: E402
from skillrc.gate.ingest.skillrc_jsonl import load_skillrc  # noqa: E402
from skillrc.gate.pairing import ArmData, PairedTable  # noqa: E402
from skillrc.gate.schema import fingerprint, fingerprint_diff, provenance_gaps  # noqa: E402
from skillrc.gate.spec import Grid  # noqa: E402
from skillrc.runner import _normalized  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_io import finalize  # noqa: E402

OUT = ROOT / "results/gate_ws32_2026_11"
CFG = ROOT / "configs/gate_ws32_2026_11"
SNAPSHOT = "9216db5781bf21249d130ec9da846c4624c16137"
THRESHOLD, INFRA_CAP = 0.30, 0.02
MEMORY_KEYS = {"memory", "retriever", "token_budget", "demo_pool", "rank_reference_pool"}
VOLATILE = {"run_id", "out_dir", "repetition", "block"}


def main(out_dir: Path = None, cfg_dir: Path = None):
    global OUT, CFG
    OUT, CFG = Path(out_dir or OUT), Path(cfg_dir or CFG)
    dec = OUT / "pilot_decision.json"
    pilot = [str(t) for t in json.loads((CFG / "schedule.json").read_text())["pilot_sessions"]]
    problems, arms, report = [], {}, {}
    for arm in ("NONE", "TRUE"):
        run = f"GWS32_P_{arm}_Q32"
        ep_path = OUT / f"{run}.episodes.jsonl"
        try:
            eps, man = load_skillrc([ep_path], arm)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{arm}: cannot load: {e}")
            continue
        if man is None:
            problems.append(f"{arm}: no manifest")
            continue
        frozen = _normalized(RunConfig.load(str(CFG / f"{run}.yaml")))
        if man.config != frozen:
            problems.append(f"{arm}: manifest config differs from the frozen config in "
                            f"{sorted(k for k in set(frozen) | set(man.config) if frozen.get(k) != man.config.get(k))}")
        if man.model != "qwen3-32b" or man.model_revision != SNAPSHOT:
            problems.append(f"{arm}: model {man.model} @ {man.model_revision}")
        gaps = provenance_gaps(man)
        if gaps:
            problems.append(f"{arm}: provenance gaps {gaps}")
        a = ArmData(arm, eps, man, Grid(task_ids=pilot, seeds=[0], repetitions=[1], registry="registry_pilot.jsonl"),
                    base_dir=CFG, max_missing_frac=1.0, max_infra_frac=1.0)
        problems += [f"{arm}: {r}" for r in a.invalid]
        vals = [e.score for e in a.scored.values()]
        report[arm] = {"expected_cells": len(a.expected), "scored_cells": len(a.scored), "missing_cells": len(a.missing),
                       "infra_cells": len(a.infra_cells), "infra_share": len(a.infra_cells) / max(1, len(a.expected)),
                       "mean_score": sum(vals) / max(1, len(vals)),
                       "episodes_sha256": hashlib.sha256(ep_path.read_bytes()).hexdigest()}
        arms[arm] = a
    if len(arms) == 2:
        pt = PairedTable(arms["TRUE"], arms["NONE"])
        problems += pt.invalid
        diff = fingerprint_diff(fingerprint(arms["TRUE"].manifest, VOLATILE | MEMORY_KEYS, MEMORY_KEYS),
                                fingerprint(arms["NONE"].manifest, VOLATILE | MEMORY_KEYS, MEMORY_KEYS))
        if diff:
            problems.append(f"pilot arms differ outside the memory keys: {diff}")
    valid = not problems and len(arms) == 2
    operational = valid and all(r["missing_cells"] == 0 and r["infra_share"] <= INFRA_CAP for r in report.values())
    score = valid and report["NONE"]["mean_score"] >= THRESHOLD
    out = {"valid": valid, "problems": problems, "operational_screen_passed": operational,
           "score_screen_passed": score, "passed": valid and operational and score, "arms": report,
           "rules": {"operational": f"every cell scored and infrastructure-failed cells <= {INFRA_CAP:.0%} per arm",
                     "score": f"NONE mean score >= {THRESHOLD}"},
           "note": "operational screen only; not evidence for any claim"}
    rc = finalize(dec, out, ["valid", "problems", "operational_screen_passed", "score_screen_passed", "passed", "arms"],
                  OUT / "PILOT_PASSED", json.dumps({k: v["episodes_sha256"] for k, v in report.items()}) + "\n",
                  out["passed"])
    print(json.dumps(out, indent=1))
    out["finalize_rc"] = rc
    return out


if __name__ == "__main__":
    sys.exit(main(*(Path(x) for x in sys.argv[1:3]))["finalize_rc"])
