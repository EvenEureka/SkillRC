"""Chain control (Codex round 9 part 2): prerequisites by success markers, idempotent markers and decisions."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/gate/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


prereq = _load("prereq")
marker = _load("phase_marker")


def _touch(root, rel):
    """A VALID marker: completion markers record an episode file and its hash."""
    import hashlib
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.name.startswith(("BLOCK_", "PHASE_")):
        ep = p.parent / f"RUN_{p.name}.episodes.jsonl"
        ep.write_text('{"x": 1}\n')
        p.write_text(json.dumps({"episodes_sha256": {ep.name.replace(".episodes.jsonl", ""): hashlib.sha256(ep.read_bytes()).hexdigest()}}))
    elif p.name == "PILOT_PASSED":
        h = {}
        for a in ("NONE", "TRUE"):
            ep = p.parent / f"GWS32_P_{a}_Q32.episodes.jsonl"
            ep.write_text('{"x": 1}\n')
            h[a] = hashlib.sha256(ep.read_bytes()).hexdigest()
        p.write_text(json.dumps(h))
    elif p.name == "D_TRIGGERED":
        p.write_text(json.dumps({"state": "BLOCK", "primary_component": "BLOCK"}))


def test_a_failed_predecessor_blocks_the_next_job(tmp_path):
    assert prereq.check("block", "1", tmp_path) == 0
    assert prereq.check("block", "2", tmp_path) == 3            # block 1 ended without success
    _touch(tmp_path, "results/gate_ws_2026_11/BLOCK_1_DONE")
    assert prereq.check("block", "2", tmp_path) == 0
    assert prereq.check("phase", "pilot", tmp_path) == 3        # block 2 not done
    _touch(tmp_path, "results/gate_ws_2026_11/BLOCK_2_DONE")
    assert prereq.check("phase", "pilot", tmp_path) == 0
    assert prereq.check("phase", "T1", tmp_path) == 3           # pilot not passed
    _touch(tmp_path, "results/gate_ws32_2026_11/PILOT_PASSED")
    assert prereq.check("phase", "T2", tmp_path) == 3           # T1 not done
    _touch(tmp_path, "results/gate_ws32_2026_11/PHASE_T1_DONE")
    _touch(tmp_path, "results/gate_ws32_2026_11/PHASE_T2_DONE")
    assert prereq.check("phase", "D1", tmp_path) == 3           # not triggered
    _touch(tmp_path, "results/gate_ws32_2026_11/D_TRIGGERED")
    assert prereq.check("phase", "D1", tmp_path) == 0
    assert prereq.check("phase", "D2", tmp_path) == 3


def test_a_completed_job_only_finalizes(tmp_path):
    _touch(tmp_path, "results/gate_ws_2026_11/BLOCK_1_DONE")
    assert prereq.check("block", "1", tmp_path) == 10
    _touch(tmp_path, "results/gate_ws_2026_11/BLOCK_2_DONE")
    _touch(tmp_path, "results/gate_ws32_2026_11/PHASE_pilot_DONE")
    assert prereq.check("phase", "pilot", tmp_path) == 10


def test_marker_is_idempotent_and_detects_changed_evidence(tmp_path):
    (tmp_path / "R1.episodes.jsonl").write_text('{"a": 1}\n')
    chain = tmp_path / "chain.txt"
    chain.write_text("configs/x/R1.yaml\n")
    assert marker.main([str(tmp_path), "PHASE_X_DONE", "1", str(chain)]) == 0
    assert marker.main([str(tmp_path), "PHASE_X_DONE", "2", str(chain)]) == 0     # rerun after a kill: unchanged
    (tmp_path / "R1.episodes.jsonl").write_text('{"a": 2}\n')
    assert marker.main([str(tmp_path), "PHASE_X_DONE", "3", str(chain)]) == 3


def test_malformed_or_mismatched_markers_block(tmp_path):
    _touch(tmp_path, "results/gate_ws_2026_11/BLOCK_1_DONE")
    m = tmp_path / "results/gate_ws_2026_11/BLOCK_1_DONE"
    (tmp_path / "results/gate_ws_2026_11/RUN_BLOCK_1_DONE.episodes.jsonl").write_text('{"x": 2}\n')  # evidence changed
    assert prereq.check("block", "2", tmp_path) == 3
    assert prereq.check("block", "1", tmp_path) == 3            # own marker invalid: neither skip nor rerun silently
    m.write_text("")                                             # truncated marker
    assert prereq.check("block", "2", tmp_path) == 3
    t = tmp_path / "results/gate_ws32_2026_11/D_TRIGGERED"
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_text(json.dumps({"state": "PASS", "primary_component": "PASS"}))
    assert prereq.marker_problem(t)


def test_interrupted_publication_leaves_no_partial_file(tmp_path, monkeypatch):
    dio = _load("decision_io")
    target = tmp_path / "trigger_decision.json"
    real = dio.os.replace

    def boom(src, dst):
        raise KeyboardInterrupt("killed during publication")

    monkeypatch.setattr(dio.os, "replace", boom)
    try:
        dio.finalize(target, {"a": 1}, ["a"], tmp_path / "MARK", "x", True)
    except KeyboardInterrupt:
        pass
    assert not target.exists()                                  # never partial: old state (none) kept
    monkeypatch.setattr(dio.os, "replace", real)
    assert dio.finalize(target, {"a": 1}, ["a"], tmp_path / "MARK", "x", True) == 0
    assert json.loads(target.read_text()) == {"a": 1} and (tmp_path / "MARK").exists()
