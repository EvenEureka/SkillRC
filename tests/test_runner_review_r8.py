"""Runner: a truncated final record is recovered on resume; interior corruption stops the resume (round-8 #4)."""
import json
import os

import pytest

from skillrc.runner import main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "configs", "mock_smoke.yaml")


def _run(tmp, *over):
    return main(["--config", CFG, "--override", f"out_dir={tmp}", "run_id=r8", *over])


def test_truncated_final_episode_record_is_recovered(tmp_path):
    _run(tmp_path, "max_tasks=2")
    ep = tmp_path / "r8.episodes.jsonl"
    lines = ep.read_text().splitlines()
    ep.write_text("\n".join(lines[:1]) + "\n" + lines[1][: len(lines[1]) // 2])     # died mid-append of record 2
    _run(tmp_path, "max_tasks=2")
    recs = [json.loads(x) for x in ep.read_text().splitlines() if x.strip()]
    assert any(p.name.startswith("r8.episodes.jsonl.partial.") for p in tmp_path.iterdir())
    inter = [r for r in recs if r["infra_error"] and "interrupted" in r["infra_error"]]
    scored = [r for r in recs if not r["infra_error"]]
    assert len(inter) == 1 and len(scored) == 2 and {r["task_id"] for r in scored} == {inter[0]["task_id"], recs[0]["task_id"]}


def test_interior_corruption_stops_the_resume(tmp_path):
    _run(tmp_path, "max_tasks=2")
    ep = tmp_path / "r8.episodes.jsonl"
    lines = ep.read_text().splitlines()
    ep.write_text("{broken\n" + "\n".join(lines) + "\n")
    with pytest.raises(SystemExit, match="corrupt"):
        _run(tmp_path, "max_tasks=2")


@pytest.mark.parametrize("which", ["episodes", "attempts"])
def test_cut_multibyte_character_is_recovered(tmp_path, which):
    _run(tmp_path, "max_tasks=2")
    f = tmp_path / f"r8.{which}.jsonl"
    f.write_bytes(f.read_bytes() + '{"task_id": "x", "note": "é'.encode()[:-1])   # cut inside the 2-byte é
    _run(tmp_path, "max_tasks=2")
    for x in f.read_text().splitlines():
        if x.strip():
            json.loads(x)
    assert any(p.name.startswith(f"r8.{which}.jsonl.partial.") for p in tmp_path.iterdir())


@pytest.mark.parametrize("which", ["episodes", "attempts"])
def test_complete_record_without_newline_gets_its_delimiter_back(tmp_path, which):
    _run(tmp_path, "max_tasks=2")
    ep = tmp_path / "r8.episodes.jsonl"
    first = ep.read_text().splitlines()[0]
    ep.write_text(first + ("" if which == "episodes" else "\n"))     # cell 2's record gone: it must rerun
    if which == "attempts":
        f = tmp_path / "r8.attempts.jsonl"
        f.write_bytes(f.read_bytes().rstrip(b"\n"))                   # journal's last record without newline
    _run(tmp_path, "max_tasks=2")
    for name in ("episodes", "attempts"):
        lines = [x for x in (tmp_path / f"r8.{name}.jsonl").read_text().splitlines() if x.strip()]
        assert all(json.loads(x) for x in lines), name
    recs = [json.loads(x) for x in ep.read_text().splitlines() if x.strip()]
    assert sum(1 for r in recs if not r["infra_error"]) == 2
