#!/usr/bin/env python3
"""Descriptive extras for the WebShop release check, computed after the runs with the gate's own modules.
Not part of any verdict; the frozen analysis spec and result.json are untouched.

Writes examples/webshop_release_check/extras.json:
  behaviour   action-category and outcome shifts (paired, task-clustered bootstrap, spec seed and draws) for
              PUB vs no memory, and for the repaired memory and each comparator edit vs PUB (panel D)
  reruns      for every arm with two repetitions: tasks whose score / steps / prompt tokens are identical in both
  totals      attempts, scored cells, missing cells, infrastructure failures over all arm-runs

Usage: python scripts/gate/release_extras.py [spec]   (default examples/webshop_release_check/gate.yaml)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from skillrc.gate.behaviour import behaviour_shift  # noqa: E402
from skillrc.gate.engine import load_arms  # noqa: E402
from skillrc.gate.pairing import PairedTable  # noqa: E402
from skillrc.gate.spec import load_spec  # noqa: E402
from skillrc.gate.stats import PairedBootstrap  # noqa: E402

SHIFTS = [("D_TRUE", "D_NONE"), ("D_ABL", "D_TRUE"), ("D_COMP1", "D_TRUE"), ("D_COMP2", "D_TRUE"),
          ("D_COMP3", "D_TRUE"), ("D_ABL", "D_NONE")]
REPEATED = ["T_NONE", "T_TRUE", "D_NONE", "D_TRUE", "D_ABL"]
RUN_ARMS = ["T_NONE", "T_TRUE", "D_NONE", "D_TRUE", "D_ABL", "D_COMP1", "D_COMP2", "D_COMP3"]


def main(spec_path: str = "examples/webshop_release_check/gate.yaml") -> Path:
    spec = load_spec(ROOT / spec_path)
    arms = load_arms(spec)
    out = {"note": "descriptive, computed after the runs with skillrc.gate.behaviour; not part of any verdict; "
                   "paired 95% intervals, no multiplicity adjustment",
           "spec": spec_path, "behaviour": {}, "reruns": {}, "totals": {}}
    for c, b in SHIFTS:
        pair = PairedTable(arms[c], arms[b])
        boot = PairedBootstrap(pair.strata, spec.bootstrap_draws, spec.seed, spec.alpha, spec.min_distinct)
        sh = behaviour_shift(pair, boot)
        sh["score_diff"] = float(pair.diff("score", "average").mean())
        out["behaviour"][f"{c}-{b}"] = sh
    for a in REPEATED:
        by_rep = {}
        for ep in arms[a].episodes:
            if ep.infra_error is None:
                by_rep.setdefault(ep.repetition, {})[ep.task_uid] = (ep.score, ep.steps, ep.prompt_tokens)
        reps = sorted(by_rep)
        if len(reps) != 2:
            continue
        r1, r2 = by_rep[reps[0]], by_rep[reps[1]]
        shared = sorted(set(r1) & set(r2))
        out["reruns"][a] = {"tasks": len(shared),
                            "same_score": sum(r1[t][0] == r2[t][0] for t in shared),
                            "same_score_steps_tokens": sum(r1[t] == r2[t] for t in shared)}
    tot = {"arm_runs_attempts": 0, "scored_cells": 0, "missing_cells": 0, "infra_failed_cells": 0}
    for a in RUN_ARMS:
        ad = arms[a]
        tot["arm_runs_attempts"] += ad.n_attempts
        tot["scored_cells"] += len(ad.scored)
        tot["missing_cells"] += len(ad.missing)
        tot["infra_failed_cells"] += len(ad.infra_cells)
    out["totals"] = tot
    dst = ROOT / Path(spec_path).parent / "extras.json"
    dst.write_text(json.dumps(out, indent=1))
    print("wrote", dst)
    return dst


if __name__ == "__main__":
    main(*sys.argv[1:])
