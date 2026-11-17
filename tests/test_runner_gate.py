"""Runner integrity for release checks, on the mock environment (no model, no GPU):
resume by cell, attempts after infrastructure failures, manifest, and that the logs are release-eligible."""
import json
import os

import pytest

from skillrc.gate.ingest.skillrc_jsonl import load_skillrc
from skillrc.gate.pairing import ArmData
from skillrc.gate.schema import RunManifest
from skillrc.gate.spec import Grid
from skillrc.runner import main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "configs", "mock_smoke.yaml")


def _run(tmp, *over):
    return main(["--config", CFG, "--override", f"out_dir={tmp}", "run_id=gate_mock", *over])


def _records(tmp, run="gate_mock"):
    with open(os.path.join(tmp, f"{run}.episodes.jsonl")) as f:
        return [json.loads(x) for x in f if x.strip()]


def test_records_carry_provenance_and_manifest(tmp_path):
    _run(tmp_path, "repetition=2", "block=2")
    recs = _records(tmp_path)
    assert len(recs) == 3
    for r in recs:
        assert r["repetition"] == 2 and r["block"] == 2 and r["attempt"] == 1
        assert r["infra_error"] is None and r["latency_s"] >= 0 and r["n_calls"] >= 1
        assert r["task_uid"].startswith(("alfworld:", "webshop:")) or ":" in r["task_uid"]
    man = RunManifest.model_validate_json((tmp_path / "gate_mock.manifest.json").read_text())
    assert man.config["repetition"] == 2 and man.ended_at is not None


def test_resume_skips_scored_cells_and_refuses_changed_config(tmp_path):
    _run(tmp_path)
    _run(tmp_path)                                  # second launch: nothing left to do
    assert len(_records(tmp_path)) == 3
    with pytest.raises(SystemExit):
        _run(tmp_path, "max_steps=9")               # different config -> refuse to append


def test_infrastructure_failure_is_rerun_and_logged(tmp_path, monkeypatch):
    import skillrc.envs.mock_env as mock_env

    real = mock_env.mock_react_policy
    state = {"fails": 0}

    def flaky(messages):
        if state["fails"] < 1:                      # first LLM call of the run fails hard
            state["fails"] += 1
            raise ConnectionError("endpoint down")
        return real(messages)

    monkeypatch.setattr(mock_env, "mock_react_policy", flaky)
    # provenance a real job supplies (the gate treats logs without it as descriptive)
    monkeypatch.setenv("SKILLRC_SERVING_JSON", '{"engine": "mock", "engine_version": "0"}')
    _run(tmp_path, "model_revision=mock-rev")
    recs = _records(tmp_path)
    assert len(recs) == 4                           # 3 cells + 1 failed attempt
    failed = [r for r in recs if r["infra_error"]]
    assert len(failed) == 1 and failed[0]["attempt"] == 1
    rerun = [r for r in recs if r["task_id"] == failed[0]["task_id"] and not r["infra_error"]]
    assert len(rerun) == 1 and rerun[0]["attempt"] == 2

    eps, man = load_skillrc([tmp_path / "gate_mock.episodes.jsonl"], "A")
    assert man is not None
    arm = ArmData("A", eps, man, Grid(task_ids=[str(r["task_id"]) for r in recs if not r["infra_error"]]),
                  max_infra_frac=0.5)
    assert not arm.invalid and not arm.descriptive and not arm.missing
    assert len(arm.infra_cells) == 1


def test_rerun_limit_leaves_a_missing_cell(tmp_path, monkeypatch):
    import skillrc.envs.mock_env as mock_env

    def always_down(messages):
        raise ConnectionError("endpoint down")

    monkeypatch.setattr(mock_env, "mock_react_policy", always_down)
    _run(tmp_path, "max_infra_reruns=1", "max_consecutive_infra=100")
    recs = _records(tmp_path)
    assert len(recs) == 6 and all(r["infra_error"] for r in recs)   # 3 cells x (1 + 1 rerun)


def test_outage_stops_the_run_early_and_resume_recovers(tmp_path, monkeypatch):
    import skillrc.envs.mock_env as mock_env

    real = mock_env.mock_react_policy
    state = {"down": True}

    def outage(messages):
        if state["down"]:
            raise ConnectionError("endpoint down")
        return real(messages)

    monkeypatch.setattr(mock_env, "mock_react_policy", outage)
    with pytest.raises(SystemExit) as e:
        _run(tmp_path)                               # default max_consecutive_infra = 2
    assert e.value.code == 75
    recs = _records(tmp_path)
    assert len(recs) == 2 and all(r["infra_error"] for r in recs)   # cell 1: attempts 1-2, reruns left
    state["down"] = False                            # servers restarted
    _run(tmp_path)
    recs = _records(tmp_path)
    scored = [r for r in recs if not r["infra_error"]]
    assert len(scored) == 3 and scored[0]["attempt"] == 3          # the interrupted cell used its last rerun
