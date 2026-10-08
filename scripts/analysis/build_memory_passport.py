#!/usr/bin/env python3
"""Build an auditable global memory-passport card from paired episode logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

from skillrc.passport import (  # noqa: E402
    build_global_passport,
    load_episode_runs,
    sha256_file,
    stratified_task_sample,
)


def parse_pair(value: str) -> tuple[str, tuple[int, ...]]:
    try:
        name, raw_seeds = value.split("=", 1)
        seeds = tuple(int(seed) for seed in raw_seeds.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("pair must look like 02=0 or 12=1,2") from error
    if not name or not seeds:
        raise argparse.ArgumentTypeError("pair name and seeds must be non-empty")
    return name, seeds


def load_run_config(run_id: str) -> dict:
    path = RESULTS / f"{run_id}.summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())["config"]


def resolve_pool(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    rooted = ROOT / candidate
    return rooted if rooted.is_file() else candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", action="append", required=True)
    parser.add_argument("--payload-run", action="append", required=True)
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--metric", choices=("success", "reward"), default="success")
    parser.add_argument("--z-value", type=float, default=1.2816)
    parser.add_argument("--stratified-per-type", type=int)
    parser.add_argument("--selection-seed", type=int, default=20260713)
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.stratified_per_type and args.task_ids:
        parser.error("use either --stratified-per-type or --task-ids")

    baseline = load_episode_runs(RESULTS, args.baseline_run)
    payload = load_episode_runs(RESULTS, args.payload_run)
    baseline_cfg = load_run_config(args.baseline_run[0])
    payload_cfg = load_run_config(args.payload_run[0])
    executor_fields = (
        "provider", "model", "prompt_style", "n_fewshot", "system_prompt_key",
        "fewshot_file", "chat_template_kwargs", "inject_admissible", "max_steps",
        "max_completion_tokens",
    )
    executor = {field: payload_cfg.get(field) for field in executor_fields}
    fewshot_path = resolve_pool(payload_cfg.get("fewshot_file"))
    executor["fewshot_sha256"] = (
        sha256_file(fewshot_path) if fewshot_path and fewshot_path.is_file() else None
    )
    mismatched = [
        field for field in executor_fields if baseline_cfg.get(field) != payload_cfg.get(field)
    ]
    if mismatched:
        raise ValueError(f"baseline and payload executor stacks differ: {mismatched}")
    if args.stratified_per_type:
        task_ids, chosen_by_type = stratified_task_sample(
            baseline, args.stratified_per_type, args.selection_seed
        )
        selection = {
            "kind": "metadata_only_stratified",
            "per_type": args.stratified_per_type,
            "seed": args.selection_seed,
            "by_type": chosen_by_type,
        }
    elif args.task_ids:
        task_ids = args.task_ids
        selection = {"kind": "explicit"}
    else:
        task_ids = sorted({task_id for _, task_id in baseline} & {task_id for _, task_id in payload})
        selection = {"kind": "all_common_tasks"}

    pool_path = resolve_pool(payload_cfg.get("demo_pool"))
    payload_spec = {
        "kind": payload_cfg["memory"],
        "demo_pool": payload_cfg.get("demo_pool"),
        "sha256": sha256_file(pool_path) if pool_path and pool_path.is_file() else None,
        "retriever": payload_cfg.get("retriever"),
        "token_budget": payload_cfg.get("token_budget"),
        "memory_k": payload_cfg.get("memory_k"),
        "run_ids": args.payload_run,
    }
    card = build_global_passport(
        baseline=baseline,
        payload=payload,
        pair_seeds=dict(args.pair),
        task_ids=task_ids,
        metric=args.metric,
        z_value=args.z_value,
        executor=executor,
        payload_spec=payload_spec,
        selection=selection,
    )
    card["calibration"]["baseline_run_ids"] = args.baseline_run
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "status": card["decision"]["status"],
        "limiting_lower_bound": card["decision"]["limiting_lower_bound"],
        "n_tasks": card["calibration"]["n_tasks"],
    }, indent=2))


if __name__ == "__main__":
    main()
