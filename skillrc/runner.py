"""Config-driven run entry point.

Usage:
  python -m skillrc.runner --config configs/m0_sanity_mock.yaml
  python -m skillrc.runner --config configs/m0_alfworld.yaml --max-tasks 3 --seeds 0
  python -m skillrc.runner --config configs/m0_alfworld.yaml --dry-run   # force mock, no API

Episodes stream to results/<run_id>.episodes.jsonl as they finish (crash-safe);
a single crash is recorded as a failed episode and the run continues.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import asdict
from typing import List

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
        return ALFWorldEnv(max_steps=cfg.max_steps)
    if cfg.env == "webshop":
        from .envs.webshop_env import WebShopEnv
        return WebShopEnv(
            n_tasks=cfg.webshop_n_tasks,
            base_url=cfg.webshop_base_url,
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
    all_results = []
    with open(ep_path, "w") as ep_f:
        for seed in cfg.seeds:
            # rewind sequential envs (e.g. ALFWorld) so every seed pass is aligned
            if hasattr(env, "rewind"):
                env.rewind()
            # reset online-curator state so each seed pass is an independent stream
            if hasattr(memory, "reset_state"):
                memory.reset_state()
            for tid in stream_ids:
                try:
                    r = agent.run_episode(env, tid, seed=seed)
                    # self-evolution hook: online curators learn from the just-finished
                    # episode (which skills it injected are stashed on the store). No-op
                    # for static/pool memories (base MemoryStore.update).
                    memory.update(r.task_id, r.trajectory, r.success, r.reward,
                                  task_type=r.task_type)
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    r = EpisodeResult(task_id=tid, task_type="error", success=False,
                                      reward=0.0, steps=0, prompt_tokens=0,
                                      completion_tokens=0, cost_usd=0.0, seed=seed,
                                      error=f"{type(e).__name__}: {e}")
                all_results.append(r)
                payload = episode_to_dict(r)
                if cfg.save_trajectories:
                    payload["trajectory"] = [asdict(step) for step in r.trajectory]
                ep_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                ep_f.flush()
                print(f"[seed {seed}] {r.task_id} succ={int(r.success)} reward={r.reward:.2f} "
                      f"steps={r.steps} ptok={r.prompt_tokens} memtok={r.memory_tokens} "
                      f"cost=${r.cost_usd:.4f}" + (f" ERR={r.error}" if r.error else ""),
                      flush=True)

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
