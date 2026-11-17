"""The 32B pilot decision validates the frozen experiment before applying thresholds (Codex round 9 #3)."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from skillrc.config import RunConfig
from skillrc.gate.ingest.skillrc_jsonl import task_uid
from skillrc.runner import build_manifest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "configs/gate_ws32_2026_11"
spec = importlib.util.spec_from_file_location("pilot32", ROOT / "scripts/gate/pilot32_check.py")
pilot32 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot32)

pytestmark = pytest.mark.skipif(not (SRC / "GWS32_P_NONE_Q32.yaml").exists(), reason="32B configs not generated")


def _setup(tmp, tasks, seed=0, score=0.5):
    cfg, out = tmp / "cfg", tmp / "out"
    cfg.mkdir(), out.mkdir()
    shutil.copy(SRC / "schedule.json", cfg)
    pilot = json.loads((SRC / "schedule.json").read_text())["pilot_sessions"]
    (cfg / "registry_pilot.jsonl").write_text("".join(
        json.dumps({"task_id": str(t), "task_uid": task_uid(t, f"WebShop [SEP] Instruction: [SEP] goal {t} [SEP] Search"),
                    "task_type": "fashion"}) + "\n" for t in pilot))
    for arm in ("NONE", "TRUE"):
        run = f"GWS32_P_{arm}_Q32"
        shutil.copy(SRC / f"{run}.yaml", cfg)
        c = RunConfig.load(str(cfg / f"{run}.yaml"))
        import os
        os.environ["SKILLRC_SERVING_JSON"] = json.dumps({"engine": "vllm", "engine_version": "0.24.0"})
        os.environ["SKILLRC_ENV_JSON"] = json.dumps({"webshop_server_seed": 0, "instructions_sha256": "x",
                                                     "server_script_sha256": "y"})
        man = build_manifest(c)
        (out / f"{run}.manifest.json").write_text(json.dumps(man))
        with (out / f"{run}.episodes.jsonl").open("w") as f:
            for t in tasks:
                f.write(json.dumps({"task_id": t, "task_type": "fashion", "success": False, "reward": score,
                                    "steps": 5, "prompt_tokens": 100, "completion_tokens": 10, "seed": seed,
                                    "memory_tokens": 0, "error": "", "repetition": 1, "block": 1, "attempt": 1,
                                    "latency_s": 1.0, "n_calls": 5, "n_retries": 0, "infra_error": None,
                                    "env": "webshop", "task_text": f"WebShop [SEP] Instruction: [SEP] goal {t} [SEP] Search"}) + "\n")
    return out, cfg, pilot


def test_wrong_tasks_and_seed_cannot_pass(tmp_path):
    out, cfg, pilot = _setup(tmp_path, list(range(100)), seed=999)        # Codex's counterexample
    d = pilot32.main(out, cfg)
    assert d["valid"] is False and d["passed"] is False and not (out / "PILOT_PASSED").exists()


def test_the_frozen_pilot_passes_and_low_score_fails_the_score_screen(tmp_path):
    pilot = json.loads((SRC / "schedule.json").read_text())["pilot_sessions"]
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    out, cfg, _ = _setup(tmp_path / "a", pilot)
    assert pilot32.main(out, cfg)["passed"] is True
    out2, cfg2, _ = _setup(tmp_path / "b", pilot, score=0.1)
    d = pilot32.main(out2, cfg2)
    assert d["valid"] and d["operational_screen_passed"] and not d["score_screen_passed"] and not d["passed"]


def test_pilot_finalization_is_idempotent(tmp_path):
    pilot = json.loads((SRC / "schedule.json").read_text())["pilot_sessions"]
    out, cfg, _ = _setup(tmp_path, pilot)
    assert pilot32.main(out, cfg)["finalize_rc"] == 0 and (out / "PILOT_PASSED").exists()
    (out / "PILOT_PASSED").unlink()                       # killed after the decision, before the marker
    assert pilot32.main(out, cfg)["finalize_rc"] == 0 and (out / "PILOT_PASSED").exists()
    d = json.loads((out / "pilot_decision.json").read_text())
    d["passed"] = False                                   # a saved decision that no longer matches the evidence
    (out / "pilot_decision.json").write_text(json.dumps(d))
    assert pilot32.main(out, cfg)["finalize_rc"] == 3
