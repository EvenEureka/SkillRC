#!/usr/bin/env python3
"""Replay selected Paper O cases on local Qwen while preserving full trajectories."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from skillrc.config import RunConfig
from skillrc.memory import build_memory
from skillrc.react import ReActAgent
from skillrc.runner import _load_fewshot, build_env, build_llm


TASK_IDS = [0, 5, 7, 30, 45, 55, 86, 113, 133]
SEEDS = [0, 1, 5]


def make_config(config_path: str, condition: str, base_url: str) -> RunConfig:
    cfg = RunConfig.load(config_path)
    cfg.run_id = f"PAPERO_Q_replay_{condition}"
    cfg.base_url = base_url
    cfg.task_ids = TASK_IDS
    cfg.seeds = SEEDS
    if condition == "nomem":
        cfg.memory = "none"
        cfg.demo_pool = None
        cfg.token_budget = None
    elif condition == "expel_bm25":
        cfg.memory = "expel"
        cfg.demo_pool = "data/pools/expel_insights.jsonl"
        cfg.token_budget = 2048
        cfg.memory_k = 20
        cfg.retriever = "bm25"
    else:
        raise ValueError(condition)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", default="results/PAPERO_Q_mechanism_replay.jsonl")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    records = []
    for condition in ["nomem", "expel_bm25"]:
        cfg = make_config("configs/qwen_nomem.yaml", condition, args.base_url)
        env = build_env(cfg)
        llm = build_llm(cfg)
        memory = build_memory(cfg)
        fewshot_list, fewshot_prompts = _load_fewshot(cfg.fewshot_file)
        agent = ReActAgent(
            llm,
            memory=memory,
            max_steps=cfg.max_steps,
            n_fewshot=cfg.n_fewshot,
            system_prompt_key=cfg.system_prompt_key,
            fewshot_examples=fewshot_list,
            prompt_style=cfg.prompt_style,
            fewshot_prompts=fewshot_prompts,
            inject_admissible=cfg.inject_admissible,
        )
        for seed in SEEDS:
            if hasattr(env, "rewind"):
                env.rewind()
            if hasattr(memory, "reset_state"):
                memory.reset_state()
            for task_id in TASK_IDS:
                result = agent.run_episode(env, task_id, seed=seed)
                records.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "task_id": str(result.task_id),
                        "task_type": result.task_type,
                        "success": bool(result.success),
                        "reward": result.reward,
                        "steps": result.steps,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "memory_tokens": result.memory_tokens,
                        "task_text": result.task_text,
                        "trajectory": [asdict(step) for step in result.trajectory],
                    }
                )
                print(
                    f"[{condition} seed={seed} task={task_id}] success={int(result.success)} "
                    f"steps={result.steps} trajectory_records={len(result.trajectory)}",
                    flush=True,
                )
        env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "episodes": len(records), "expected": 54}))


if __name__ == "__main__":
    main()
