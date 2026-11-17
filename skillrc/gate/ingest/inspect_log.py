"""Inspect AI eval logs (`.eval` or `.json`) -> Episodes + RunManifest.

Mapping (one Inspect sample = one scored attempt of one cell):
  task_id      sample.id                              repetition   sample.epoch
  task_uid     inspect:<task>:<sample id>:<sha(input)> task_type    sample.metadata[task_type_key]
  score        the named score's value (C/I/P/N, bool or number; dict values via value_key) in [0, 1]
  tokens       summed sample.model_usage               latency_s    sample.total_time (wall time of the sample)
  n_calls      number of model events                  trajectory   assistant tool calls -> tool results
  infra_error  sample.error; each entry of sample.error_retries becomes an earlier failed attempt
Manifest: task, task args (flattened to `task_args.<name>`), solver, model, model args, generate config, epochs and
dataset from `log.eval`. Provenance is producer-time only: a task argument `<name>_sha256` (computed and checked by
the task when it ran) becomes `files_sha256["task_args.<name>"]`; the logged system message of the samples
(identical across a run, else an error) becomes `files_sha256["system_message"]`. Nothing is hashed at ingestion.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..schema import Episode, RunManifest, Step, config_hash, fingerprint, fingerprint_diff, text_hash

_LETTER = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}
WITHIN_ARM_VOLATILE = {"run_id", "eval_id", "created"}


def _to_float(v: Any, value_key: Optional[str]) -> Optional[float]:
    if isinstance(v, dict):
        if value_key is None:
            if len(v) != 1:
                raise ValueError(f"score value is a dict with keys {sorted(v)}; set options.value_key")
            v = next(iter(v.values()))
        else:
            v = v.get(value_key)
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        if v in _LETTER:
            return _LETTER[v]
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for c in content or []:
        t = getattr(c, "text", None)
        if t:
            parts.append(t)
    return " ".join(parts)


def _trajectory(messages) -> List[Step]:
    steps: List[Step] = []
    pending: Dict[str, int] = {}
    for m in messages:
        role = getattr(m, "role", "")
        if role == "assistant":
            calls = getattr(m, "tool_calls", None) or []
            if calls:
                for tc in calls:
                    args = json.dumps(tc.arguments, ensure_ascii=False, sort_keys=True)
                    steps.append(Step(step=len(steps) + 1, action=f"{tc.function}({args[:300]})", obs=""))
                    pending[tc.id] = len(steps) - 1
            else:
                steps.append(Step(step=len(steps) + 1, action="answer", obs=_text(m.content)[:1500]))
        elif role == "tool":
            i = pending.get(getattr(m, "tool_call_id", None))
            if i is not None:
                steps[i] = steps[i].model_copy(update={"obs": _text(m.content)[:1500]})
    return steps


def _system_message(sample) -> Optional[str]:
    for m in sample.messages or []:
        if getattr(m, "role", "") == "system":
            return _text(m.content)
    return None


def _manifest(log, path: Path) -> RunManifest:
    s = log.eval
    gen = s.model_generate_config.model_dump(exclude_none=True) if s.model_generate_config else {}
    config = {
        "task": s.task, "task_version": s.task_version,
        "solver": s.solver, "solver_args": s.solver_args or {},
        "model": s.model, "model_args": s.model_args or {}, "generate_config": gen,
        "epochs": getattr(s.config, "epochs", None),
        "dataset": {"name": s.dataset.name, "samples": s.dataset.samples,
                    "sample_ids": [str(x) for x in (s.dataset.sample_ids or [])]},
    }
    for k, v in sorted((s.task_args or {}).items()):
        config[f"task_args.{k}"] = v
    config = json.loads(json.dumps(config, default=str, sort_keys=True))
    files = {}
    for k, v in (s.task_args or {}).items():
        if k.endswith("_sha256") and v:
            files[f"task_args.{k[:-len('_sha256')]}"] = str(v)
    systems = {_system_message(x) for x in (log.samples or [])}
    systems.discard(None)
    if len(systems) > 1:
        raise ValueError(f"{path}: samples of one run have different system messages")
    if systems:
        files["system_message"] = text_hash(next(iter(systems)), 64)
    pk = s.packages or {}
    meta = s.metadata or {}            # producer-time provenance passed to inspect eval(metadata=...)
    serving = {"model_base_url": s.model_base_url, **(meta.get("serving") or {})}
    return RunManifest(run_id=s.run_id or s.eval_id or path.stem, config=config, config_sha256=config_hash(config),
                       model=s.model, model_revision=meta.get("model_revision"),
                       serving=serving,
                       env={"inspect_ai": pk.get("inspect_ai"), "packages": pk,
                            "revision": s.revision.model_dump() if s.revision else None},
                       files_sha256=files, scorer_version=None, started_at=str(s.created), ended_at=None, host=None)


def load_inspect(paths: List[Path], arm: str, manifest_path: Optional[Path] = None,
                 score_name: Optional[str] = None, value_key: Optional[str] = None,
                 task_type_key: Optional[str] = None) -> Tuple[List[Episode], Optional[RunManifest]]:
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as e:  # pragma: no cover
        raise ValueError("the inspect adapter needs `pip install skillrc[inspect]`") from e
    episodes: List[Episode] = []
    manifests: List[RunManifest] = []
    for p in paths:
        p = Path(p)
        log = read_eval_log(str(p))
        spec = log.eval
        manifests.append(_manifest(log, p))
        for s in log.samples or []:
            text = s.input if isinstance(s.input, str) else " ".join(_text(m.content) for m in s.input)
            uid = f"inspect:{spec.task}:{s.id}:{text_hash(text)}"
            ttype = str((s.metadata or {}).get(task_type_key, "all")) if task_type_key else "all"
            common = dict(run_id=spec.run_id or p.stem, arm=arm, task_id=str(s.id), task_uid=uid,
                          task_type=ttype, seed=0, repetition=int(s.epoch))
            retries = list(s.error_retries or [])
            for k, err in enumerate(retries, 1):
                episodes.append(Episode(**common, attempt=k, score=None,
                                        infra_error=str(getattr(err, "message", err))[:500]))
            usage = list((s.model_usage or {}).values())
            pt = sum(int(u.input_tokens or 0) for u in usage)
            ct = sum(int(u.output_tokens or 0) for u in usage)
            score = None
            if s.error is None:
                scores = s.scores or {}
                if not scores:
                    raise ValueError(f"{p}: sample {s.id} epoch {s.epoch} has no score and no error")
                name = score_name or (next(iter(scores)) if len(scores) == 1 else None)
                if name is None or name not in scores:
                    raise ValueError(f"{p}: choose options.score_name among {sorted(scores)}")
                score = _to_float(scores[name].value, value_key)
                if score is None or not 0.0 <= score <= 1.0:
                    raise ValueError(f"{p}: sample {s.id}: score {scores[name].value!r} is not in [0, 1]")
            traj = _trajectory(s.messages or [])
            n_model = sum(1 for e in (s.events or []) if getattr(e, "event", "") == "model")
            episodes.append(Episode(
                **common, attempt=len(retries) + 1, score=score,
                success=None if score is None else score == 1.0, steps=len(traj),
                prompt_tokens=pt, completion_tokens=ct, n_calls=n_model, n_retries=len(retries),
                latency_s=s.total_time, infra_error=(str(s.error.message)[:500] if s.error else None),
                trajectory=traj))
    if manifest_path is not None:
        return episodes, RunManifest.model_validate_json(Path(manifest_path).read_text())
    f0 = fingerprint(manifests[0], WITHIN_ARM_VOLATILE)
    for m in manifests[1:]:
        diff = fingerprint_diff(f0, fingerprint(m, WITHIN_ARM_VOLATILE))
        if diff:
            raise ValueError(f"arm {arm}: Inspect logs {manifests[0].run_id} and {m.run_id} differ in {diff}")
    return episodes, manifests[0]
