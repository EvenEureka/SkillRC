"""Unified metrics + result serialization.

Every system is scored the same way: success rate (primary), env score/reward,
steps, prompt-tokens (cost proxy), realized injected-memory tokens (the budget-
match audit — must be ~equal across M1 confounds), and skill-library size.
Episodes are streamed to JSONL as they complete (crash-safe); a summary JSON is
written at the end for the analysis / auto-review skills.
"""
from __future__ import annotations

import json
import os
import statistics
from dataclasses import asdict
from typing import Any, Dict, List


def _ms(xs: List[float]) -> Dict[str, float]:
    return {"mean": statistics.fmean(xs),
            "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0}


def episode_to_dict(r) -> Dict[str, Any]:
    return {k: v for k, v in asdict(r).items() if k != "trajectory"}


def summarize(results, group: str = "task_type") -> Dict[str, Any]:
    done = [r for r in results if not getattr(r, "error", "")]
    n = len(results)
    if not done:
        return {"n": n, "n_ok": 0, "n_error": n}
    succ = [1.0 if r.success else 0.0 for r in done]
    out: Dict[str, Any] = {
        "n": n,
        "n_ok": len(done),
        "n_error": n - len(done),
        "success_rate": statistics.fmean(succ),
        "reward": _ms([r.reward for r in done]),
        "steps": _ms([float(r.steps) for r in done]),
        "prompt_tokens": _ms([float(r.prompt_tokens) for r in done]),
        "completion_tokens": _ms([float(r.completion_tokens) for r in done]),
        "memory_tokens": _ms([float(getattr(r, "memory_tokens", 0)) for r in done]),
        "total_cost_usd": sum(r.cost_usd for r in done),
    }
    groups: Dict[Any, List[float]] = {}
    for r in done:
        groups.setdefault(getattr(r, group, "unknown"), []).append(
            1.0 if r.success else 0.0)
    out["by_" + group] = {str(k): {"n": len(v), "success_rate": statistics.fmean(v)}
                          for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}
    # per-seed success (so seeds can be treated as replicates in analysis)
    per_seed: Dict[Any, List[float]] = {}
    for r in done:
        per_seed.setdefault(getattr(r, "seed", 0), []).append(1.0 if r.success else 0.0)
    out["by_seed"] = {str(k): statistics.fmean(v) for k, v in sorted(per_seed.items())}
    return out


def write_summary(out_dir: str, run_id: str, config_dict: Dict[str, Any],
                  summary: Dict[str, Any], usage) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run_id}.summary.json")
    with open(path, "w") as f:
        json.dump({"run_id": run_id, "config": config_dict, "summary": summary,
                   "usage": asdict(usage)}, f, indent=2, ensure_ascii=False)
    return path
