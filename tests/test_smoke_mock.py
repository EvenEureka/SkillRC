"""Zero-cost end-to-end smoke test of the M0 harness on MockEnv.

Runnable directly:  python tests/test_smoke_mock.py
Or under pytest.    Proves: prompt-build -> LLM -> action-parse -> env-step ->
metrics -> token accounting all work, with no API cost and no external deps.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from skillrc.runner import main  # noqa: E402


def test_mock_end_to_end():
    cfg = os.path.join(ROOT, "configs", "mock_smoke.yaml")
    summary = main(["--config", cfg])
    assert summary["n"] == 3, summary
    assert summary["success_rate"] == 1.0, f"expected perfect mock success, got {summary}"
    assert summary["memory"]["kind"] == "none"
    assert summary["total_cost_usd"] == 0.0
    print("\nSMOKE OK: success_rate =", summary["success_rate"])


def test_mock_confound_path():
    """Pool-memory path (memory=random_k over the focal 33-insight pool) runs end-to-end."""
    summary = main([
        "--config", os.path.join(ROOT, "configs", "mock_smoke.yaml"),
        "--override", "run_id=smoke_pool", "memory=random_k", "memory_k=5",
        "demo_pool=data/pools/expel_insights.jsonl", "seeds=[0,1,2]",
    ])
    assert summary["n"] == 9, summary  # 3 mock tasks x 3 seeds
    assert summary["success_rate"] == 1.0, summary
    assert summary["memory"]["kind"] == "random_k"
    assert summary["memory"]["n_items"] >= 1, "sample pool should load"
    print("SMOKE OK (pool memory): pool items =", summary["memory"]["n_items"])


if __name__ == "__main__":
    test_mock_end_to_end()
    test_mock_confound_path()
    print("\nALL SMOKE TESTS PASSED")
