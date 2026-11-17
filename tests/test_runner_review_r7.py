"""Runner and LLM-client regressions for the Codex round-7 findings 2 and 5."""
import json
import os

import pytest

from skillrc.llm import LLMCallFailed, LLMClient
from skillrc.runner import main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "configs", "mock_smoke.yaml")


def _run(tmp, *over):
    return main(["--config", CFG, "--override", f"out_dir={tmp}", "run_id=r7", *over])


def _records(tmp):
    with open(os.path.join(tmp, "r7.episodes.jsonl")) as f:
        return [json.loads(x) for x in f if x.strip()]


def test_resume_refuses_changed_file_content_at_the_same_path(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({"text": "[put] rule one", "task_type": "put"}) + "\n")
    _run(tmp_path, "memory=random_k", f"demo_pool={pool}", "memory_k=1")
    pool.write_text(json.dumps({"text": "[put] rule two", "task_type": "put"}) + "\n")   # same path, new bytes
    with pytest.raises(SystemExit, match="files_sha256"):
        _run(tmp_path, "memory=random_k", f"demo_pool={pool}", "memory_k=1")


def test_manifest_is_append_only_across_launches(tmp_path):
    _run(tmp_path)
    m1 = json.loads((tmp_path / "r7.manifest.json").read_text())
    _run(tmp_path)
    m2 = json.loads((tmp_path / "r7.manifest.json").read_text())
    assert m2["started_at"] == m1["started_at"] and m2["config_sha256"] == m1["config_sha256"]
    assert len(m2["segments"]) == 2 and m2["segments"][0] == m1["segments"][0]


def test_interrupted_attempt_is_recorded_on_resume(tmp_path):
    _run(tmp_path, "max_tasks=1")
    # simulate a process that died after starting attempt 1 of task 1 (start record, no episode record)
    with open(tmp_path / "r7.attempts.jsonl", "a") as f:
        f.write(json.dumps({"task_id": "1", "seed": 0, "repetition": 1, "block": None, "attempt": 1,
                            "env": "mock", "at": "t"}) + "\n")
    _run(tmp_path, "max_tasks=1")       # same config; task 1 is outside max_tasks but its attempt is still accounted
    recs = _records(tmp_path)
    inter = [r for r in recs if r["infra_error"] and "interrupted" in r["infra_error"]]
    assert len(inter) == 1 and inter[0]["task_id"] == "1" and inter[0]["attempt"] == 1


class _Boom(Exception):
    pass


class _FakeCompletions:
    def __init__(self, fail_times):
        self.fail_times, self.calls = fail_times, 0

    def create(self, **kw):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise _Boom("transient")

        class U:
            prompt_tokens, completion_tokens, prompt_tokens_details = 10, 2, None

        class C:
            class message:
                content = "Action: look"

        class R:
            choices, usage = [C()], U()
        return R()


def _client(fail_times, max_retries, monkeypatch):
    monkeypatch.setattr("skillrc.llm.time.sleep", lambda s: None)
    c = LLMClient(model="m", provider="mock", max_retries=max_retries)
    comp = _FakeCompletions(fail_times)
    c._client = type("X", (), {"chat": type("Y", (), {"completions": comp})()})()
    return c, comp


def test_retry_accounting_is_exact(monkeypatch):
    c, comp = _client(2, 5, monkeypatch)
    r = c._complete_openai([{"role": "user", "content": "hi"}])
    assert comp.calls == 3 and r.retries == 2 and c.usage.retries == 2 and c.usage.calls == 1


def test_retries_are_one_try_plus_max_retries(monkeypatch):
    c, comp = _client(99, 2, monkeypatch)
    with pytest.raises(LLMCallFailed):
        c._complete_openai([{"role": "user", "content": "hi"}])
    assert comp.calls == 3 and c.usage.failed_calls == 1 and c.usage.retries == 2


def test_sdk_retries_are_disabled():
    pytest.importorskip("openai")
    c = LLMClient(model="m", provider="openai", base_url="http://127.0.0.1:9/v1", api_key="EMPTY")
    assert c._client.max_retries == 0
