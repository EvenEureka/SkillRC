import json
from pathlib import Path

import yaml

from skillrc.gate.cli import main
from skillrc.gate.ingest.skillrc_jsonl import record_to_episode, task_uid

from conftest import episodes, manifest

WS_TEXT = ("[Current page: Search]\nWebShop [SEP] Instruction: [SEP] Find me double sided pillows, "
           "and price lower than 30.00 dollars [SEP] Search")


def test_webshop_uid_ignores_page_label_and_whitespace():
    a = task_uid(0, WS_TEXT)
    b = task_uid(0, WS_TEXT.replace("[Current page: Search]\n", "").replace("  ", " "))
    assert a == b and a.startswith("webshop:0:")
    assert task_uid(0, WS_TEXT.replace("30.00", "40.00")) != a       # different goal -> different identity


def test_legacy_record_maps_error_to_infrastructure_failure():
    r = {"task_id": 3, "task_type": "garden", "success": False, "reward": 0.0, "steps": 2, "prompt_tokens": 10,
         "completion_tokens": 2, "seed": 0, "memory_tokens": 524, "error": "ConnectionError", "task_text": WS_TEXT,
         "trajectory": [{"step": 1, "action": "search[x]", "obs": "o", "reward": 0.0, "prompt_tokens": 5,
                         "completion_tokens": 1}]}
    ep = record_to_episode(r, "A", "RUN")
    assert ep.infra_error == "ConnectionError" and ep.score is None and not ep.scored
    assert ep.repetition is None and ep.trajectory[0].action == "search[x]"
    ok = record_to_episode({**r, "error": "", "reward": 0.5}, "A", "RUN")
    assert ok.scored and ok.score == 0.5


def _write_arm(tmp: Path, name: str, eps, man):
    p = tmp / f"{name}.episodes.jsonl"
    p.write_text("".join(e.model_dump_json() + "\n" for e in eps))
    (tmp / f"{name}.manifest.json").write_text(man.model_dump_json())
    return p


def test_cli_report_end_to_end(tmp_path):
    n = 120
    base = episodes("BASE", n, lambda i, r: float((i * 7 + r) % 5 < 3), reps=(1, 2),
                    tokens=lambda i, r: 1000 + i % 9, latency=lambda i, r: 5 + (i % 7) * 0.5)
    cand = episodes("CAND", n, lambda i, r: float((i * 7 + r) % 5 < 1), reps=(1, 2),
                    tokens=lambda i, r: 1100 + i % 9, latency=lambda i, r: 5.5 + (i % 7) * 0.5)
    _write_arm(tmp_path, "BASE", base, manifest("BASE"))
    _write_arm(tmp_path, "CAND", cand, manifest("CAND", memory_pool="m.jsonl"))
    grid = {"task_range": [0, n], "repetitions": [1, 2]}
    spec = {
        "name": "cli-e2e", "bootstrap_draws": 2000,
        "arms": {a: {"format": "episodes", "logs": [f"{a}.episodes.jsonl"], "manifest": f"{a}.manifest.json",
                     "grid": grid} for a in ("BASE", "CAND")},
        "interventions": {"memory": {"config_patch": {"memory_pool": "m.jsonl"},
                                     "files_sha256": {"memory_pool": "sha-m.jsonl"}}},
        "targets": {"ship": {"baseline": "BASE", "candidate": "CAND", "interventions": ["memory"], "default": True,
                             "primary": {"metric": "score", "margin": 0.05},
                             "guardrails": [{"metric": "prompt_tokens", "statistic": "relative_mean",
                                             "tolerance": 0.3}]}},
    }
    sp = tmp_path / "gate.yaml"
    sp.write_text(yaml.safe_dump(spec))
    out = tmp_path / "out"
    code = main(["report", str(sp), "--out", str(out)])
    assert code == 2                                            # BLOCK: score drops by ~0.4
    res = json.loads((out / "result.json").read_text())
    assert res["targets"]["ship"]["state"] == "BLOCK"
    assert "BLOCK" in (out / "summary.md").read_text()
    html = (out / "report.html").read_text()
    assert "<svg" in html and "Paired replays" in html


def test_cli_schema_and_validate(tmp_path, capsys):
    assert main(["schema", "--out", str(tmp_path / "s.json")]) == 0
    s = json.loads((tmp_path / "s.json").read_text())
    assert {"Episode", "RunManifest", "GateSpec"} <= set(s)
    p = _write_arm(tmp_path, "A", episodes("A", 3, lambda i, r: 1.0), manifest("A"))
    assert main(["validate", str(p), "--format", "episodes", "--manifest", str(tmp_path / "A.manifest.json")]) == 0
