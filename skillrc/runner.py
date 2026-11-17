"""Config-driven run entry point.

Usage:
  python -m skillrc.runner --config configs/mock_smoke.yaml
  python -m skillrc.runner --config configs/w1_2026_10_13/eval/W1A1013_NONE_Q8.yaml --max-tasks 3 --seeds 0
  python -m skillrc.runner --config configs/w1_2026_10_13/eval/W1A1013_NONE_Q8.yaml --dry-run   # force mock, no API

Episodes stream to results/<run_id>.episodes.jsonl as they finish (crash-safe);
a single crash is recorded as a failed episode and the run continues.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from dataclasses import asdict
from typing import List

from . import __version__
from .config import RunConfig
from .llm import LLMClient
from .memory import build_memory
from .metrics import episode_to_dict, summarize, write_summary
from .react import EpisodeResult, ReActAgent


def build_env(cfg: RunConfig):
    if cfg.env == "mock":
        from .envs.mock_env import MockEnv
        return MockEnv()
    if cfg.env == "alfworld":
        from .envs.alfworld_env import ALFWorldEnv
        return ALFWorldEnv(split=getattr(cfg, "alfworld_split", "eval_out_of_distribution"), max_steps=cfg.max_steps)
    if cfg.env == "webshop":
        from .envs.webshop_env import WebShopEnv
        return WebShopEnv(
            n_tasks=cfg.webshop_n_tasks,
            base_url=cfg.webshop_base_url,
            show_page_label=cfg.webshop_page_label,
            timeout=cfg.request_timeout,
        )
    raise ValueError(f"unknown env {cfg.env!r}")


def build_llm(cfg: RunConfig) -> LLMClient:
    mock_policy = None
    if cfg.provider == "mock":
        from .envs.mock_env import mock_react_policy
        mock_policy = mock_react_policy
    extra_body = None
    if getattr(cfg, "chat_template_kwargs", None):
        extra_body = {"chat_template_kwargs": cfg.chat_template_kwargs}
    return LLMClient(model=cfg.model, provider=cfg.provider, temperature=cfg.temperature,
                     max_completion_tokens=cfg.max_completion_tokens, base_url=cfg.base_url,
                     request_timeout=cfg.request_timeout, max_retries=cfg.max_retries,
                     mock_policy=mock_policy, api_key=getattr(cfg, "api_key", None),
                     extra_body=extra_body)


def _load_fewshot(path):
    """Returns (fewshot_list, fewshot_prompts_dict). .json => dict (react_alfworld),
    .jsonl => list of exemplar strings (generic)."""
    if not path:
        return [], {}
    if path.endswith(".json"):
        with open(path) as f:
            return [], json.load(f)
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)["text"])
    return out, {}


def _coerce(key: str, v: str):
    """Coerce an override string by the field's DECLARED type (not the current
    value's runtime type), so None/list-defaulted fields parse correctly."""
    if key not in RunConfig.__dataclass_fields__:
        raise ValueError(f"unknown override key: {key}")
    t = str(RunConfig.__dataclass_fields__[key].type)
    if v.lower() in ("none", "null"):
        return None
    if "bool" in t:
        return v.lower() in ("1", "true", "yes")
    if "List" in t or "list" in t:
        inner = v.strip().lstrip("[").rstrip("]")
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        out = []
        for p in parts:
            try:
                out.append(int(p))
            except ValueError:
                out.append(p)
        return out
    if "float" in t:
        return float(v)
    if "int" in t:
        return int(v)
    return v


# ---------------------------------------------------------------------------------------------------------------
# Release-check provenance: run manifest, resume by cell, attempts with infrastructure failures.
# A cell is (task_id, seed, repetition). Its scored attempt is the first one without an infrastructure failure.
# ---------------------------------------------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Connection details change between launches (ports) and secrets must never be written: they are kept out of the
# hashed config. Endpoints are recorded under `serving` / `env`; api_key is not recorded anywhere.
CONNECTION_KEYS = ("base_url", "webshop_base_url", "api_key")
EXIT_OUTAGE = 75     # EX_TEMPFAIL: consecutive infrastructure failures; resume after restarting the servers


def _normalized(cfg: RunConfig) -> dict:
    d = json.loads(json.dumps(cfg.to_dict(), default=str, sort_keys=True))
    return {k: v for k, v in d.items() if k not in CONNECTION_KEYS}


def build_manifest(cfg: RunConfig) -> dict:
    """Manifest dict matching skillrc.gate.schema.RunManifest. Serving/env details that the harness cannot see
    (vLLM version, GPU, server seed) are passed by the job script as JSON in SKILLRC_SERVING_JSON / SKILLRC_ENV_JSON."""
    from .gate.schema import config_hash, file_sha256

    conf = _normalized(cfg)
    files = {k: file_sha256(v) for k, v in conf.items() if isinstance(v, str) and v and os.path.isfile(v)}
    serving = {"provider": cfg.provider, "base_url": cfg.base_url, "temperature": cfg.temperature,
               "max_completion_tokens": cfg.max_completion_tokens, "request_timeout": cfg.request_timeout,
               "max_retries": cfg.max_retries}
    serving.update(json.loads(os.environ.get("SKILLRC_SERVING_JSON", "{}") or "{}"))
    env_info = {"env": cfg.env, "webshop_base_url": getattr(cfg, "webshop_base_url", None)}
    env_info.update(json.loads(os.environ.get("SKILLRC_ENV_JSON", "{}") or "{}"))
    return {"schema_version": 1, "run_id": cfg.run_id, "config": conf, "config_sha256": config_hash(conf),
            "model": cfg.model, "model_revision": cfg.model_revision, "serving": serving, "env": env_info,
            "memory": {"kind": cfg.memory, "token_budget": cfg.token_budget},
            "files_sha256": files, "scorer_version": f"skillrc-{__version__}",
            "started_at": _now(), "ended_at": None, "host": socket.gethostname(),
            "segments": [_segment(serving, env_info)]}


def _segment(serving: dict, env_info: dict) -> dict:
    from .gate.schema import EXECUTION_ENV_KEYS, EXECUTION_SERVING_KEYS
    return {"started_at": _now(), "ended_at": None, "host": socket.gethostname(), "job_id": os.environ.get("JOB_ID"),
            "serving": {k: v for k, v in serving.items() if k in EXECUTION_SERVING_KEYS},
            "env": {k: v for k, v in env_info.items() if k in EXECUTION_ENV_KEYS}, "stopped": None}


def write_manifest(path: str, manifest: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _attempts_path(ep_path: str) -> str:
    return ep_path.replace(".episodes.jsonl", ".attempts.jsonl")


def _read_jsonl_recovering(path: str) -> list:
    """Parse a JSONL file written by appends, recovering the two states an interrupted append can leave:
    * a truncated final record (incomplete JSON or a cut multi-byte character): moved to `<path>.partial.<time>`
      and the file cut back to its last complete record;
    * a complete final record without its newline: the newline is restored, so the next append starts a new line.
    Any corrupt record that is not the last one stops the resume: it cannot come from an interrupted append."""
    with open(path, "rb") as f:
        raw = f.read()
    lines = raw.split(b"\n")
    rows, good_bytes = [], 0
    for i, ln in enumerate(lines):
        last = i == len(lines) - 1
        if not ln.strip():
            good_bytes += len(ln) + (0 if last else 1)
            continue
        try:
            rows.append(json.loads(ln.decode("utf-8")))
            good_bytes += len(ln) + (0 if last else 1)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if any(x.strip() for x in lines[i + 1:]):
                raise SystemExit(f"{path}: corrupt record at line {i + 1} that is not the last; refusing to resume")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            with open(f"{path}.partial.{stamp}", "wb") as g:
                g.write(ln)
            with open(path, "r+b") as g:
                g.truncate(good_bytes)
            print(f"[resume] {path}: truncated final record moved aside ({len(ln)} bytes)", flush=True)
            return rows
    if raw and not raw.endswith(b"\n"):
        with open(path, "ab") as g:
            g.write(b"\n")
        print(f"[resume] {path}: restored the newline after the final record", flush=True)
    return rows


def load_existing(ep_path: str, man_path: str, manifest: dict, resume: bool) -> dict:
    """Resume state from an existing run written under the same fingerprint.

    Returns the cells already scored, the attempts used per cell, and the manifest to write: the original identity
    with this launch appended to `segments` (the original record is never replaced). An attempt that was started
    (attempts log) but never completed (no episode record) is recorded now as an interrupted infrastructure failure,
    so it counts against the cell's reruns."""
    from .gate.schema import RunManifest, fingerprint, fingerprint_diff

    out = {"done": set(), "attempts": Counter(), "manifest": manifest}
    if not os.path.exists(ep_path):
        return out
    if not resume:
        raise SystemExit(f"{ep_path} exists and resume=false; refusing to overwrite")
    if not os.path.exists(man_path):
        raise SystemExit(f"{ep_path} exists without a manifest (legacy run); refusing to append to it")
    with open(man_path) as f:
        old = json.load(f)
    diff = fingerprint_diff(fingerprint(RunManifest.model_validate(old)), fingerprint(RunManifest.model_validate(manifest)))
    if diff:
        raise SystemExit(f"refusing to resume {ep_path}: this launch differs from the original run in {diff}")
    old.setdefault("segments", []).append(manifest["segments"][-1])
    old["ended_at"] = None
    out["manifest"] = old
    completed = set()
    for r in _read_jsonl_recovering(ep_path):
        if True:
            key = (str(r["task_id"]), int(r.get("seed", 0)), int(r.get("repetition") or 1))
            completed.add((key, int(r.get("attempt", 1))))
            out["attempts"][key] = max(out["attempts"][key], int(r.get("attempt", 1)))
            if not r.get("infra_error"):
                out["done"].add(key)
    ap = _attempts_path(ep_path)
    if os.path.exists(ap):
        interrupted = []
        for a in _read_jsonl_recovering(ap):   # a truncated start record means the attempt never began
            key = (str(a["task_id"]), int(a["seed"]), int(a["repetition"]))
            if (key, int(a["attempt"])) not in completed and key not in out["done"]:
                interrupted.append((key, a))
        with open(ep_path, "a") as ep_f:
            for key, a in interrupted:
                rec = {"task_id": a["task_id"], "task_type": "unknown", "success": False, "reward": 0.0, "steps": 0,
                       "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "seed": a["seed"],
                       "memory_tokens": 0, "error": "interrupted", "task_text": "", "repetition": a["repetition"],
                       "block": a.get("block"), "attempt": a["attempt"], "latency_s": 0.0, "episode_wall_s": None,
                       "n_calls": 0, "n_retries": 0, "agent_error": None, "env": a.get("env"), "task_uid": None,
                       "infra_error": "interrupted: the process stopped during this attempt"}
                ep_f.write(json.dumps(rec) + "\n")
                out["attempts"][key] = max(out["attempts"][key], int(a["attempt"]))
        if interrupted:
            print(f"[resume] recorded {len(interrupted)} interrupted attempt(s) as infrastructure failures", flush=True)
    return out


def log_attempt_start(ep_path: str, tid, seed, cfg: RunConfig, attempt: int) -> None:
    with open(_attempts_path(ep_path), "a") as f:
        f.write(json.dumps({"task_id": str(tid), "seed": int(seed), "repetition": int(cfg.repetition),
                            "block": cfg.block, "attempt": attempt, "env": cfg.env, "at": _now()}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_attempt(agent, env, memory, llm, tid, seed, cfg: RunConfig, attempt: int):
    """One attempt of one cell. Any exception is an infrastructure failure; tokens and time used so far are kept."""
    from .gate.ingest.skillrc_jsonl import task_uid

    before = llm.usage.snapshot()
    t0 = time.perf_counter()
    infra = None
    try:
        r = agent.run_episode(env, tid, seed=seed)
        memory.update(r.task_id, r.trajectory, r.success, r.reward, task_type=r.task_type)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        infra = f"{type(e).__name__}: {e}"
        obs = getattr(agent, "last_obs", None)
        r = EpisodeResult(task_id=tid, task_type=getattr(obs, "task_type", None) or "unknown", success=False,
                          reward=0.0, steps=0, prompt_tokens=0, completion_tokens=0, cost_usd=0.0, seed=seed,
                          error=infra, task_text=getattr(obs, "text", "") or "")
    after = llm.usage.snapshot()
    payload = episode_to_dict(r)
    payload.update({
        "repetition": cfg.repetition, "block": cfg.block, "attempt": attempt,
        "latency_s": round(after["latency_s"] - before["latency_s"], 4),
        "episode_wall_s": round(time.perf_counter() - t0, 4),
        "n_calls": int(after["calls"] - before["calls"] + after["failed_calls"] - before["failed_calls"]),
        "n_retries": int(after["retries"] - before["retries"]),
        "infra_error": infra, "agent_error": None, "env": cfg.env,
        "task_uid": task_uid(tid, payload["task_text"], cfg.env) if payload.get("task_text") else None,
    })
    if infra:  # keep what the failed attempt consumed
        payload["prompt_tokens"] = int(after["prompt_tokens"] - before["prompt_tokens"])
        payload["completion_tokens"] = int(after["completion_tokens"] - before["completion_tokens"])
    if cfg.save_trajectories:
        payload["trajectory"] = [asdict(step) for step in r.trajectory]
    return r, payload


def main(argv: List[str] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="key=value config overrides")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override cfg.seeds")
    ap.add_argument("--dry-run", action="store_true", help="force provider=mock (no API cost)")
    args = ap.parse_args(argv)

    cfg = RunConfig.load(args.config)
    for kv in args.override:
        k, v = kv.split("=", 1)
        setattr(cfg, k, _coerce(k, v))
    if args.max_tasks is not None:
        cfg.max_tasks = args.max_tasks
    if args.seeds is not None:
        cfg.seeds = args.seeds
    if args.dry_run:
        cfg.provider = "mock"

    print(f"[config] {cfg.run_id} | env={cfg.env} provider={cfg.provider} "
          f"model={cfg.model} memory={cfg.memory} K={cfg.memory_k} "
          f"budget={cfg.token_budget} style={cfg.prompt_style} seeds={cfg.seeds}", flush=True)

    env = build_env(cfg)
    llm = build_llm(cfg)
    memory = build_memory(cfg)
    fewshot_list, fewshot_prompts = _load_fewshot(cfg.fewshot_file)
    agent = ReActAgent(llm, memory=memory, max_steps=cfg.max_steps,
                       n_fewshot=cfg.n_fewshot, system_prompt_key=cfg.system_prompt_key,
                       fewshot_examples=fewshot_list, prompt_style=cfg.prompt_style,
                       fewshot_prompts=fewshot_prompts, inject_admissible=cfg.inject_admissible)

    ids = list(cfg.task_ids) if cfg.task_ids else env.task_ids()
    if cfg.max_tasks:
        ids = ids[:cfg.max_tasks]
    # Online curators care about STREAM ORDER; optionally shuffle it (the real seed
    # axis for a self-evolving method under a temp-0 executor). Env still reset()s by
    # absolute game index, so shuffling only reorders when the curator learns.
    stream_ids = list(ids)
    if getattr(cfg, "stream_shuffle_seed", None) is not None:
        import random as _r
        _r.Random(cfg.stream_shuffle_seed).shuffle(stream_ids)

    os.makedirs(cfg.out_dir, exist_ok=True)
    ep_path = os.path.join(cfg.out_dir, f"{cfg.run_id}.episodes.jsonl")
    man_path = os.path.join(cfg.out_dir, f"{cfg.run_id}.manifest.json")
    manifest = build_manifest(cfg)
    existing = load_existing(ep_path, man_path, manifest, cfg.resume)
    manifest = existing["manifest"]
    write_manifest(man_path, manifest)
    done, attempts = existing["done"], existing["attempts"]
    if done:
        print(f"[resume] {len(done)} cells already scored in {ep_path}; skipping them", flush=True)
    all_results = []
    streak = 0
    with open(ep_path, "a") as ep_f:
        for seed in cfg.seeds:
            # rewind sequential envs (e.g. ALFWorld) so every seed pass is aligned
            if hasattr(env, "rewind"):
                env.rewind()
            # reset online-curator state so each seed pass is an independent stream
            if hasattr(memory, "reset_state"):
                memory.reset_state()
            for tid in stream_ids:
                key = (str(tid), int(seed), int(cfg.repetition))
                if key in done:
                    continue
                while attempts[key] < 1 + cfg.max_infra_reruns:
                    attempts[key] += 1
                    log_attempt_start(ep_path, tid, seed, cfg, attempts[key])
                    r, payload = run_attempt(agent, env, memory, llm, tid, seed, cfg, attempts[key])
                    all_results.append(r)
                    ep_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    ep_f.flush()
                    print(f"[seed {seed} rep {cfg.repetition} try {attempts[key]}] {r.task_id} "
                          f"succ={int(r.success)} reward={r.reward:.2f} steps={r.steps} "
                          f"ptok={payload['prompt_tokens']} memtok={r.memory_tokens} "
                          f"llm={payload['latency_s']:.1f}s" + (f" INFRA={r.error}" if r.error else ""),
                          flush=True)
                    if not payload["infra_error"]:
                        done.add(key)
                        streak = 0
                        break
                    streak += 1
                    if streak >= cfg.max_consecutive_infra:
                        manifest["segments"][-1].update(ended_at=_now(), stopped="outage")
                        write_manifest(man_path, manifest)
                        print(f"[abort] {streak} failed attempts in a row: stopping so the job can restart the "
                              f"servers and resume (exit {EXIT_OUTAGE})", flush=True)
                        raise SystemExit(EXIT_OUTAGE)
    manifest["ended_at"] = _now()
    manifest["segments"][-1].update(ended_at=manifest["ended_at"], stopped="complete")
    write_manifest(man_path, manifest)

    summary = summarize(all_results)
    summary["memory"] = {"kind": memory.name, **memory.size()}
    # self-evolving curator: record learned utilities + stream event log (for the
    # longitudinal / degradation curves). No-op for static/pool memories.
    if hasattr(memory, "state_summary"):
        state_key = "passport" if memory.name == "passport" else "evolve"
        summary[state_key] = memory.state_summary()
        try:
            ev_path = os.path.join(cfg.out_dir, f"{cfg.run_id}.events.jsonl")
            with open(ev_path, "w") as ev_f:
                for e in getattr(memory, "event_log", []):
                    ev_f.write(json.dumps(e, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            pass
    sm_path = write_summary(cfg.out_dir, cfg.run_id, cfg.to_dict(), summary, llm.usage)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[usage] calls={llm.usage.calls} prompt_tok={llm.usage.prompt_tokens} "
          f"compl_tok={llm.usage.completion_tokens} cached={llm.usage.cached_tokens} "
          f"cost=${llm.usage.cost_usd:.4f}")
    print(f"[saved] {ep_path}\n        {sm_path}")
    env.close()
    return summary


if __name__ == "__main__":
    main()
