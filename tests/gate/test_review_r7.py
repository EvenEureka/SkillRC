"""Regression tests for the Codex round-7 pre-launch findings (engineering/REVIEW_r7_codex_prelaunch.md)."""
import json
from pathlib import Path

import numpy as np
import pytest

from skillrc.gate.engine import load_arms, run_gate
from skillrc.gate.ingest.skillrc_jsonl import load_skillrc
from skillrc.gate.pairing import ArmData, PairedTable
from skillrc.gate.schema import RunManifest, config_hash
from skillrc.gate.spec import GateSpec, Grid
from skillrc.gate.verdict import GuardrailEvidence, PrimaryEvidence, State, decide

from conftest import arm, episodes, manifest, spec
from test_engine import BASE_SCORE, MEM, POOL, TARGET, make, perturbed, two_arms
from test_verdict import est, prim


def _with(man: RunManifest, **upd) -> RunManifest:
    d = man.model_dump()
    for k, v in upd.items():
        d[k] = {**d[k], **v} if isinstance(v, dict) else v
    return RunManifest.model_validate(d)


# --- finding 1: undeclared environment / serving / scorer / file differences must not PASS --------------------------
@pytest.mark.parametrize("field,value", [
    ("serving", {"engine_version": "0.25.0"}),
    ("env", {"webshop_server_seed": 1}),
    ("scorer_version", "other"),
    ("files_sha256", {"fewshot_file": "sha-changed"}),
])
def test_undeclared_execution_differences_are_invalid(field, value):
    arms = two_arms(perturbed(BASE_SCORE, 3))
    assert run_gate(spec(TARGET, ["BASE", "CAND"], MEM), arms).targets["C"].state == State.PASS
    arms["CAND"].manifest = _with(arms["CAND"].manifest, **{field: value})
    v = run_gate(spec(TARGET, ["BASE", "CAND"], MEM), arms).targets["C"]
    assert v.state == State.INVALID and any("undeclared" in r for r in v.reasons)


def test_execution_metadata_may_differ():
    arms = two_arms(perturbed(BASE_SCORE, 3))
    arms["CAND"].manifest = _with(arms["CAND"].manifest, serving={"gpu_uuid": "GPU-x", "chain": "B", "job_id": "9"},
                                  env={"webshop_base_url": "http://127.0.0.1:1"})
    assert run_gate(spec(TARGET, ["BASE", "CAND"], MEM), arms).targets["C"].state == State.PASS


def _write_run(tmp: Path, run: str, eps, man: RunManifest):
    (tmp / f"{run}.episodes.jsonl").write_text("".join(
        json.dumps({**e.model_dump(), "reward": e.score}) + "\n" for e in eps))
    (tmp / f"{run}.manifest.json").write_text(man.model_dump_json())


def test_within_arm_fingerprint_mismatch_fails_only_that_arm(tmp_path):
    m1 = manifest("R1")
    m2 = _with(m1, run_id="R2", serving={"engine_version": "other"})
    _write_run(tmp_path, "R1", episodes("A", 4, lambda i, r: 1.0, reps=(1,)), m1)
    _write_run(tmp_path, "R2", episodes("A", 4, lambda i, r: 1.0, reps=(2,)), m2)
    with pytest.raises(ValueError, match="differ"):
        load_skillrc([tmp_path / "R1.episodes.jsonl", tmp_path / "R2.episodes.jsonl"], "A")


# --- finding 6: a task missing from one arm is a missing cell, not a different task set -----------------------------
def test_wholly_missing_task_is_bounded_not_invalid():
    n = 300
    b = arm("BASE", episodes("BASE", n, BASE_SCORE, reps=(1, 2)), n, (1, 2), manifest("BASE"))
    c = arm("CAND", episodes("CAND", n, perturbed(BASE_SCORE, 3), reps=(1, 2), skip=[(7, 1)]), n, (1, 2),
            manifest("CAND", memory_pool=POOL))
    # task 7 repetition 1 missing; now remove task 9 entirely from the candidate
    c = arm("CAND", [e for e in c.episodes if e.task_id != "9"], n, (1, 2), manifest("CAND", memory_pool=POOL),
            max_missing_frac=0.01)
    t = PairedTable(c, b)
    assert not t.invalid and t.n == n and t.any_missing()
    v = run_gate(spec({"C": {**TARGET["C"], "guardrails": []}}, ["BASE", "CAND"], MEM),
                 {"BASE": b, "CAND": c}).targets["C"]
    assert v.state == State.INCONCLUSIVE


def test_registry_supplies_identity_and_type(tmp_path):
    reg = tmp_path / "reg.jsonl"
    reg.write_text("".join(json.dumps({"task_id": str(i), "task_uid": f"toy:{i}:{i:016x}", "task_type": "x"}) + "\n"
                           for i in range(5)))
    eps = [e for e in episodes("A", 5, lambda i, r: 1.0) if e.task_id != "2"]
    eps = [e.model_copy(update={"task_type": "x"}) for e in eps]
    a = ArmData("A", eps, None, Grid(task_range=[0, 5], registry=str(reg)), base_dir=tmp_path, max_missing_frac=0.5)
    assert ("toy:2:0000000000000002", 0, 1) in a.missing and not a.invalid
    bad = [e.model_copy(update={"task_uid": "toy:1:ffff"}) if e.task_id == "1" else e for e in eps]
    b = ArmData("B", bad, None, Grid(task_range=[0, 5], registry=str(reg)), base_dir=tmp_path, max_missing_frac=0.5)
    assert any("registry" in r for r in b.invalid)


# --- finding 8: a broken arm invalidates only the targets that use it; default target is explicit -------------------
def test_unloadable_arm_does_not_invalidate_other_targets(tmp_path):
    s = GateSpec.model_validate({"name": "x", "bootstrap_draws": 2000,
                                 "arms": {"A": {"logs": ["nope/*.jsonl"]}, "B": {"logs": ["nope2/*.jsonl"]}},
                                 "targets": {"T": {"baseline": "A", "candidate": "B", "default": True,
                                                   "primary": {"metric": "score", "margin": 0.05}}}})
    s._base = tmp_path
    arms = load_arms(s)
    assert any("failed to load" in r for r in arms["A"].invalid)
    arms2 = two_arms(perturbed(BASE_SCORE, 3))
    arms2["BROKEN"] = arms["A"]
    targets = {"C": TARGET["C"], "X": {**TARGET["C"], "candidate": "BROKEN", "default": False}}
    r = run_gate(spec(targets, ["BASE", "CAND", "BROKEN"], MEM), arms2)
    assert r.targets["C"].state == State.PASS and r.targets["X"].state == State.INVALID


# --- finding 10: flags, sensitivity, completed-cell guardrails, mean-over-baselines contrast ------------------------
def test_tolerated_flag_survives_a_cost_block():
    g = GuardrailEvidence("tokens", 0.30, est(0.6, 0.1, name="tokens"))
    v = decide("t", prim(-0.01, 0.0053, margin=0.02), [g])
    assert v.state == State.BLOCK and any("tolerated" in f for f in v.flags)


def test_margin_sensitivity_and_completed_cell_guardrails_are_reported():
    r = run_gate(spec(TARGET, ["BASE", "CAND"], MEM), two_arms(perturbed(BASE_SCORE, 3), skip=[(5, 1)]))
    d = r.target_details["C"]
    assert [x["margin"] for x in d["primary"]["margin_sensitivity"]] == [0.0, 0.02, 0.05]
    assert all(g["estimate"] is not None and "completed attempts only" in g["note"] for g in d["guardrails"])


def test_mean_over_baselines_contrast():
    arms = {"BASE": make("BASE", BASE_SCORE), "CAND": make("CAND", perturbed(BASE_SCORE, 3)),
            "CAND2": make("CAND2", perturbed(BASE_SCORE, 4))}
    con = {"M": {"candidate": "BASE", "baseline": ["CAND", "CAND2"], "role": "diagnostic"}}
    r = run_gate(spec(TARGET, ["BASE", "CAND", "CAND2"], MEM, con), arms)
    m = r.contrasts["M"]
    a = run_gate(spec(TARGET, ["BASE", "CAND", "CAND2"], MEM,
                      {"A": {"candidate": "BASE", "baseline": "CAND"}, "B": {"candidate": "BASE", "baseline": "CAND2"}}), arms)
    assert m["estimate"]["estimate"] == pytest.approx(
        (a.contrasts["A"]["estimate"]["estimate"] + a.contrasts["B"]["estimate"]["estimate"]) / 2)
