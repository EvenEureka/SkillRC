#!/usr/bin/env python3
"""Validate Paper O executor-moderation inputs against raw episode files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT.parent / "ARR" / "PaperO_assets" / "executor_moderation_integrity.json"

RUNS = {
    "gpt-4o-mini": {
        "none": "R008_nomem_s3",
        "expel_bm25": "R012_expel_bm25",
        "expel_dense": "R012_expel_dense",
        "skillos_bm25": "R012_skillos_bm25",
        "skillos_dense": "R012_skillos_dense",
    },
    "qwen3-8b": {
        "none": "Q_nomem_s3",
        "expel_bm25": "Q_expel_bm25_s3",
        "expel_dense": "Q_expel_s3",
        "skillos_bm25": "Q_skillos_bm25_s3",
        "skillos_dense": "Q_skillos_s3",
    },
}

COMMON_CONFIG_FIELDS = [
    "env",
    "max_tasks",
    "max_steps",
    "temperature",
    "max_completion_tokens",
    "n_fewshot",
    "system_prompt_key",
    "prompt_style",
    "inject_admissible",
    "fewshot_file",
    "memory_k",
]


def load_run(run_id: str) -> tuple[list[dict], dict]:
    episode_path = RESULTS / f"{run_id}.episodes.jsonl"
    summary_path = RESULTS / f"{run_id}.summary.json"
    with episode_path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    with summary_path.open() as handle:
        summary = json.load(handle)
    return rows, summary


def episode_key(row: dict) -> tuple[int, str]:
    return int(row["seed"]), str(row["task_id"])


def check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def main() -> None:
    loaded = {}
    checks = []
    metrics = {}

    for executor, conditions in RUNS.items():
        for condition, run_id in conditions.items():
            rows, summary = load_run(run_id)
            keys = [episode_key(row) for row in rows]
            successes = np.array([int(row["success"]) for row in rows], dtype=float)
            errors = [row for row in rows if row.get("error")]
            loaded[(executor, condition)] = {
                "rows": rows,
                "by_key": {episode_key(row): row for row in rows},
                "summary": summary,
                "keys": set(keys),
            }
            metrics[run_id] = {
                "n": len(rows),
                "unique_keys": len(set(keys)),
                "n_errors": len(errors),
                "success_rate_recomputed": float(successes.mean()),
                "success_rate_summary": float(summary["summary"]["success_rate"]),
            }
            checks.extend(
                [
                    check(f"{run_id}:complete", len(rows) == 402, f"n={len(rows)}"),
                    check(
                        f"{run_id}:unique_episode_keys",
                        len(set(keys)) == len(keys),
                        f"unique={len(set(keys))}, total={len(keys)}",
                    ),
                    check(f"{run_id}:zero_errors", not errors, f"errors={len(errors)}"),
                    check(
                        f"{run_id}:summary_success_matches_raw",
                        np.isclose(successes.mean(), summary["summary"]["success_rate"]),
                        f"raw={successes.mean():.12f}, summary={summary['summary']['success_rate']:.12f}",
                    ),
                ]
            )

    reference_keys = loaded[("gpt-4o-mini", "none")]["keys"]
    for (executor, condition), run in loaded.items():
        checks.append(
            check(
                f"{executor}/{condition}:paired_key_set",
                run["keys"] == reference_keys,
                f"missing={len(reference_keys - run['keys'])}, extra={len(run['keys'] - reference_keys)}",
            )
        )

    for executor, conditions in RUNS.items():
        configs = {
            condition: loaded[(executor, condition)]["summary"]["config"]
            for condition in conditions
        }
        for field in COMMON_CONFIG_FIELDS:
            values = {condition: config.get(field) for condition, config in configs.items()}
            checks.append(
                check(
                    f"{executor}:common_config:{field}",
                    len({json.dumps(value, sort_keys=True) for value in values.values()}) == 1,
                    json.dumps(values, sort_keys=True),
                )
            )

    configuration_aliases = []
    for condition in ["expel_bm25", "expel_dense", "skillos_bm25", "skillos_dense"]:
        gpt_config = loaded[("gpt-4o-mini", condition)]["summary"]["config"]
        qwen_config = loaded[("qwen3-8b", condition)]["summary"]["config"]
        memory_matches = gpt_config.get("memory") == qwen_config.get("memory")
        static_evolve_alias = (
            condition.startswith("skillos_")
            and gpt_config.get("memory") == "skillos"
            and qwen_config.get("memory") == "evolve"
            and qwen_config.get("curator") == "static"
            and gpt_config.get("demo_pool") == qwen_config.get("demo_pool")
        )
        if static_evolve_alias:
            configuration_aliases.append(
                {
                    "condition": condition,
                    "gpt_memory": "skillos",
                    "qwen_memory": "evolve",
                    "qwen_curator": "static",
                    "reason": (
                        "SelfEvolvingMemory with curator=static uses the same _rank and "
                        "budget-fill behavior as SkillOSMemory and its update does not alter retrieval."
                    ),
                }
            )
        checks.append(
            check(
                f"cross_executor/{condition}:memory_semantics",
                memory_matches or static_evolve_alias,
                (
                    f"gpt={gpt_config.get('memory')!r}, qwen={qwen_config.get('memory')!r}, "
                    f"qwen_curator={qwen_config.get('curator')!r}, alias={static_evolve_alias}"
                ),
            )
        )
        for field in ["memory_k", "token_budget", "demo_pool", "retriever", "embed_model"]:
            checks.append(
                check(
                    f"cross_executor/{condition}:memory_config:{field}",
                    gpt_config.get(field) == qwen_config.get(field),
                    f"gpt={gpt_config.get(field)!r}, qwen={qwen_config.get(field)!r}",
                )
            )

    failed = [item for item in checks if not item["passed"]]
    payload = {
        "material_id": "PAPERO-EXECMOD-INTEGRITY-2026-07-12-01",
        "verification_status": "ANALYZED",
        "all_checks_passed": not failed,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "scope_note": (
            "Validates local raw-data completeness, pairing, summaries, and configured controls. "
            "It does not re-run external GPT generations or local Qwen inference."
        ),
        "configuration_aliases": configuration_aliases,
        "metrics": metrics,
        "checks": checks,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: payload[key] for key in ["all_checks_passed", "n_checks", "n_failed"]}))
    print(OUT)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
