"""SkillRC runner logs: `<run_id>.episodes.jsonl` (+ optional `<run_id>.manifest.json`).

Two generations are accepted:

* legacy records (runs before the gate existed): task_id, task_type, success, reward, steps, tokens, seed,
  memory_tokens, error, task_text, trajectory. They have no repetition, attempt or latency, so they are read in
  DESCRIPTIVE mode only.
* current records (runner with attempts log): the legacy fields plus repetition, block, attempt, latency_s,
  n_calls, n_retries, infra_error, agent_error.

Task identity: `<env>:<task_id>:<sha256(task text)[:16]>`, where the task text is the WebShop instruction, or the
full ALFWorld task text with any `[Current page: ...]` label removed. Hashing the text catches id remapping (an
ALFWorld re-download once mapped the same ids to different games).
"""
from __future__ import annotations

import json
import re
import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..schema import Episode, RunManifest, Step, fingerprint, fingerprint_diff, text_hash

_LABEL = re.compile(r"^\s*\[Current page:[^\]]*\]\s*\n?")
_WS_INSTR = re.compile(r"Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]", re.S)


def detect_env(task_text: str) -> str:
    return "webshop" if "WebShop [SEP]" in task_text or "Instruction: [SEP]" in task_text else "alfworld"


def task_text_key(task_text: str, env: Optional[str] = None) -> Tuple[str, str]:
    """(env, canonical text used for the identity hash)."""
    text = _LABEL.sub("", task_text or "")
    env = env or detect_env(text)
    if env == "webshop":
        m = _WS_INSTR.search(text)
        if m:
            return env, " ".join(m.group(1).split())
    return env, " ".join(text.split())


def task_uid(task_id: Any, task_text: str, env: Optional[str] = None) -> str:
    env, key = task_text_key(task_text, env)
    return f"{env}:{task_id}:{text_hash(key)}"


def score_of(r: Dict[str, Any], env: str) -> Optional[float]:
    """Score in [0, 1]: WebShop's final purchase reward; success for every other environment
    (ALFWorld and the mock environment accumulate step rewards that are not a score)."""
    if r.get("score") is not None:
        return float(r["score"])
    if env == "webshop" and r.get("reward") is not None:
        return float(r["reward"])
    if r.get("success") is not None:
        return float(bool(r["success"]))
    return None


def record_to_episode(r: Dict[str, Any], arm: str, run_id: str) -> Episode:
    infra = r.get("infra_error")
    if infra is None and r.get("error"):
        infra = str(r["error"])              # legacy: any recorded crash is an infrastructure failure
    env = r.get("env") or task_text_key(r.get("task_text", ""))[0]
    score = score_of(r, env)
    traj = None
    if r.get("trajectory") is not None:
        traj = [Step(step=int(s.get("step", i + 1)), action=str(s.get("action", "")),
                     obs=str(s.get("obs", "")), reward=s.get("reward"))
                for i, s in enumerate(r["trajectory"])]
    return Episode(
        run_id=str(r.get("run_id", run_id)),
        arm=arm,
        task_id=str(r["task_id"]),
        task_uid=r.get("task_uid") or (task_uid(r["task_id"], r["task_text"], r.get("env")) if r.get("task_text")
                                       else f"unknown:{r['task_id']}"),
        task_type=str(r.get("task_type") or "all"),
        seed=int(r.get("seed", 0)),
        repetition=r.get("repetition"),
        block=r.get("block"),
        attempt=int(r.get("attempt", 1)),
        score=None if infra else score,
        success=None if r.get("success") is None else bool(r["success"]),
        steps=r.get("steps"),
        prompt_tokens=r.get("prompt_tokens"),
        completion_tokens=r.get("completion_tokens"),
        memory_tokens=r.get("memory_tokens"),
        n_calls=r.get("n_calls"),
        n_retries=r.get("n_retries"),
        latency_s=r.get("latency_s"),
        infra_error=infra,
        agent_error=r.get("agent_error"),
        trajectory=traj,
    )


# an arm may be split across runs by repetition, block or task subset (coverage is checked by the grid)
WITHIN_ARM_VOLATILE = {"run_id", "out_dir", "repetition", "block", "task_ids"}


def _fill_unknown_identities(episodes: List[Episode]) -> List[Episode]:
    """An attempt that failed before the task text was seen gets the identity of the same task_id from the
    other attempts of the arm (all attempts of a cell share one task)."""
    known: Dict[str, Tuple[str, str]] = {}
    for e in episodes:
        if not e.task_uid.startswith("unknown:"):
            known.setdefault(e.task_id, (e.task_uid, e.task_type))
    out = []
    for e in episodes:
        if e.task_uid.startswith("unknown:") and e.task_id in known:
            uid, tt = known[e.task_id]
            e = e.model_copy(update={"task_uid": uid, "task_type": tt if e.task_type in ("unknown", "all") else e.task_type})
        out.append(e)
    return out


def load_skillrc(paths: List[Path], arm: str,
                 manifest_path: Optional[Path] = None) -> Tuple[List[Episode], Optional[RunManifest]]:
    episodes: List[Episode] = []
    manifests: List[Optional[RunManifest]] = []
    for p in paths:
        p = Path(p)
        run_id = p.name.replace(".gz", "").replace(".episodes.jsonl", "").replace(".jsonl", "")
        with (gzip.open(p, "rt") if p.suffix == ".gz" else p.open()) as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    episodes.append(record_to_episode(json.loads(line), arm, run_id))
                except Exception as e:  # surface the exact line
                    raise ValueError(f"{p}:{ln}: {e}") from e
        side = p.with_name(run_id + ".manifest.json")
        manifests.append(RunManifest.model_validate_json(side.read_text()) if side.exists() else None)
    episodes = _fill_unknown_identities(episodes)
    if manifest_path is not None:
        return episodes, RunManifest.model_validate_json(Path(manifest_path).read_text())
    if not manifests or any(m is None for m in manifests):
        if any(m is not None for m in manifests):
            raise ValueError(f"arm {arm}: some runs have a manifest and some do not")
        return episodes, None
    f0 = fingerprint(manifests[0], WITHIN_ARM_VOLATILE)
    for m in manifests[1:]:
        diff = fingerprint_diff(f0, fingerprint(m, WITHIN_ARM_VOLATILE))
        if diff:
            raise ValueError(f"arm {arm}: runs {manifests[0].run_id} and {m.run_id} differ in {diff}")
    return episodes, manifests[0]
