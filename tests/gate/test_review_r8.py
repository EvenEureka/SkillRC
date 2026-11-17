"""Regression tests for the Codex round-8 findings (engineering/REVIEW_r8_codex.md)."""
import json

import numpy as np

from skillrc.gate.engine import run_gate
from skillrc.gate.pairing import ArmData
from skillrc.gate.schema import Episode, RunManifest, config_hash
from skillrc.gate.spec import Grid
from skillrc.gate.stats import Estimate
from skillrc.gate.verdict import PrimaryEvidence, State, decide

from conftest import episodes, spec
from test_engine import BASE_SCORE, MEM, TARGET, perturbed, two_arms


def test_equal_missing_provenance_is_descriptive_not_pass():
    arms = two_arms(perturbed(BASE_SCORE, 3))
    for a in arms.values():
        d = a.manifest.model_dump()
        d.update(model_revision=None, scorer_version=None, serving={}, env={})
        a.manifest = RunManifest.model_validate(d)
        a.descriptive.clear()
        a.descriptive.extend(ArmData(a.name, a.episodes, a.manifest, a.grid).descriptive)
    v = run_gate(spec(TARGET, ["BASE", "CAND"], MEM), arms).targets["C"]
    assert v.state == State.DESCRIPTIVE and any("provenance" in r for r in v.reasons)


def test_unknown_identity_attempts_resolve_from_the_registry(tmp_path):
    reg = tmp_path / "reg.jsonl"
    reg.write_text("".join(json.dumps({"task_id": str(i), "task_uid": f"toy:{i}:{i:016x}", "task_type": "x"}) + "\n"
                           for i in range(500)))
    eps = [e.model_copy(update={"task_type": "x"}) for e in episodes("A", 500, lambda i, r: 1.0, reps=(1, 2))
           if e.task_id != "7"]
    for att in (1, 2):   # task 7 rep 1: every attempt failed before the task text was seen
        eps.append(Episode(run_id="A", arm="A", task_id="7", task_uid="unknown:7", task_type="unknown", seed=0,
                           repetition=1, block=1, attempt=att, score=None, infra_error="interrupted"))
    a = ArmData("A", eps, None, Grid(task_range=[0, 500], repetitions=[1, 2], registry=str(reg)), base_dir=tmp_path)
    assert not a.invalid, a.invalid
    assert ("toy:7:0000000000000007", 0, 1) in a.missing and ("toy:7:0000000000000007", 0, 1) in a.infra_cells


def test_tolerated_flag_requires_the_optimistic_upper_bound_below_zero():
    pess = Estimate("p", -0.002, np.linspace(-0.04, -0.0001, 20_000), alpha=0.025)
    opt = Estimate("o", 0.001, np.linspace(-0.03, 0.0042, 20_000), alpha=0.025)
    v = decide("t", PrimaryEvidence("score", 0.05, pess, opt), [], any_missing=True)
    assert not any("tolerated" in f for f in v.flags)
    v2 = decide("t", PrimaryEvidence("score", 0.05, pess, pess), [])
    assert any("tolerated" in f for f in v2.flags)


def test_difference_of_differences_contrast():
    from test_engine import make, independent
    arms = {"B8": make("BASE", independent(0.6, 11)), "C8": make("C8", independent(0.4, 12)),
            "B32": make("BASE", independent(0.6, 13)), "C32": make("C32", independent(0.6, 14))}
    for k, a in arms.items():
        a.name = k
    con = {"DD": {"candidate": "C32", "baseline": "B32", "minus": {"candidate": "C8", "baseline": "B8"}},
           "A": {"candidate": "C32", "baseline": "B32"}, "B": {"candidate": "C8", "baseline": "B8"}}
    t = {"C": {**TARGET["C"], "baseline": "B8", "candidate": "C8"}}
    r = run_gate(spec(t, list(arms), MEM, con), arms)
    dd = r.contrasts["DD"]["estimate"]["estimate"]
    assert dd == __import__("pytest").approx(r.contrasts["A"]["estimate"]["estimate"] - r.contrasts["B"]["estimate"]["estimate"])
    assert r.contrasts["DD"]["estimate"]["lower"] > 0     # the 8B change hurts (-0.2), the 32B change does not


def test_difference_of_differences_sees_missing_cells_in_the_subtracted_pair():
    from test_engine import make, independent
    arms = {"B8": make("BASE", independent(0.6, 11)), "C8": make("C8", independent(0.4, 12), skip=[(3, 2)]),
            "B32": make("BASE", independent(0.6, 13)), "C32": make("C32", independent(0.6, 14))}
    for k, a in arms.items():
        a.name = k
    con = {"DD": {"candidate": "C32", "baseline": "B32", "minus": {"candidate": "C8", "baseline": "B8"}}}
    t = {"C": {**TARGET["C"], "baseline": "B32", "candidate": "C32"}}
    r = run_gate(spec(t, list(arms), MEM, con), arms)
    assert r.contrasts["DD"]["evaluable"] is False


def test_generic_category_uses_the_tool_name():
    from skillrc.gate.behaviour import generic_category
    assert generic_category('search_files({"query": "x"})') == "search_files"
    assert generic_category("search[shoes]") == "search"
    assert generic_category("answer") == "answer"
