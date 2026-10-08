#!/usr/bin/env python3
"""Evaluate the predeclared 20-task WebShop feasibility gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.episodes).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    summary = json.loads(Path(args.summary).read_text())["summary"]
    actions = [
        step["action"]
        for row in rows
        for step in row.get("trajectory", [])
    ]
    env_actions = [action for action in actions if not action.lower().startswith("think:")]
    valid_syntax = [
        action.startswith("search[") or action.startswith("click[")
        for action in env_actions
    ]
    episodes_with_search = sum(
        any(step["action"].startswith("search[") for step in row.get("trajectory", []))
        for row in rows
    )
    episodes_with_buy = sum(
        any(step["action"].lower() == "click[buy now]" for step in row.get("trajectory", []))
        for row in rows
    )
    invalid_observations = sum(
        any(
            marker in step.get("obs", "").lower()
            for marker in ["invalid action", "not a valid", "nothing happens"]
        )
        for row in rows
        for step in row.get("trajectory", [])
    )
    syntax_rate = float(np.mean(valid_syntax)) if valid_syntax else 0.0
    mean_reward = float(summary.get("reward", {}).get("mean", 0.0))
    successes = sum(int(row["success"]) for row in rows)

    checks = {
        "complete_20": len(rows) == 20,
        "zero_errors": int(summary.get("n_error", -1)) == 0,
        "valid_action_syntax_ge_95pct": syntax_rate >= 0.95,
        "search_coverage_ge_15": episodes_with_search >= 15,
        "purchase_attempts_ge_5": episodes_with_buy >= 5,
        "nondegenerate_reward_or_success": successes >= 2 or mean_reward >= 0.20,
    }
    passed = all(checks.values())
    payload = {
        "material_id": "PAPERO-WEBSHOP-GATE-2026-07-12-01",
        "verification_status": "ANALYZED",
        "decision": "proceed_minimal_cross_environment_comparison" if passed else "stop_webshop_and_report_gate_failure",
        "passed": passed,
        "predeclared_checks": checks,
        "metrics": {
            "episodes": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows) if rows else 0.0,
            "mean_reward": mean_reward,
            "action_syntax_rate": syntax_rate,
            "episodes_with_search": episodes_with_search,
            "episodes_with_buy_now": episodes_with_buy,
            "invalid_observations": invalid_observations,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
