"""Zero-API tests for passport calibration and runtime enforcement."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skillrc.config import RunConfig  # noqa: E402
from skillrc.memory import build_memory  # noqa: E402
from skillrc.passport import build_global_passport, sha256_file  # noqa: E402


def fake_rows(values: dict[str, list[int]]) -> tuple[dict, dict]:
    baseline, payload = {}, {}
    for pair_index, (pair, outcomes) in enumerate(values.items()):
        seed = pair_index
        for task_id, outcome in enumerate(outcomes):
            baseline[(seed, task_id)] = {"success": False, "reward": 0.0}
            payload[(seed, task_id)] = {"success": bool(outcome), "reward": float(outcome)}
    return baseline, payload


class PassportTests(unittest.TestCase):
    def test_global_passport_requires_every_pair(self) -> None:
        baseline, payload = fake_rows({"a": [1, 1, 1, 1], "b": [1, 1, 1, 1]})
        card = build_global_passport(
            baseline, payload, {"a": (0,), "b": (1,)}, range(4), "success", 1.2816,
            {"provider": "mock", "model": "mock"}, {"kind": "expel"},
        )
        self.assertTrue(card["decision"]["deploy"])

        baseline, payload = fake_rows({"a": [1, 1, 1, 1], "b": [0, 0, 0, 0]})
        card = build_global_passport(
            baseline, payload, {"a": (0,), "b": (1,)}, range(4), "success", 1.2816,
            {"provider": "mock", "model": "mock"}, {"kind": "expel"},
        )
        self.assertFalse(card["decision"]["deploy"])
        self.assertEqual(card["decision"]["limiting_pair"], "b")

    def test_runtime_enforces_card_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            pool = tmp_path / "pool.jsonl"
            pool.write_text(
                json.dumps({"id": "s1", "text": "Always inspect before acting."}) + "\n"
            )
            base_card = {
                "schema_version": "memory-passport-v1",
                "scope": "global",
                "executor": {"provider": "mock", "model": "mock-model"},
                "payload": {"kind": "expel", "sha256": sha256_file(pool)},
                "decision": {
                    "status": "deployable", "deploy": True, "limiting_lower_bound": 0.1,
                },
            }
            card_path = tmp_path / "passport.json"
            card_path.write_text(json.dumps(base_card))
            cfg = RunConfig(
                provider="mock", model="mock-model", memory="passport",
                passport_file=str(card_path), demo_pool=str(pool), token_budget=100,
            )
            memory = build_memory(cfg)
            self.assertIn("inspect", memory.retrieve("inspect the room"))
            self.assertEqual(memory.size()["deployed"], 1)

            base_card["decision"] = {
                "status": "quarantined", "deploy": False, "limiting_lower_bound": -0.1,
            }
            card_path.write_text(json.dumps(base_card))
            self.assertEqual(build_memory(cfg).retrieve("inspect the room"), "")

            cfg.model = "different-model"
            with self.assertRaisesRegex(ValueError, "identity check failed"):
                build_memory(cfg)

            cfg.model = "mock-model"
            pool.write_text(json.dumps({"id": "s1", "text": "Payload changed."}) + "\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                build_memory(cfg)


if __name__ == "__main__":
    unittest.main()
