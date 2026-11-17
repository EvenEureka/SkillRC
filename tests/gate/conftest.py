"""Synthetic arms for gate tests."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pytest

from skillrc.gate.pairing import ArmData
from skillrc.gate.schema import Episode, RunManifest, Step, config_hash
from skillrc.gate.spec import GateSpec, Grid

TYPES = ("fashion", "garden", "grocery")


def uid(i: int) -> str:
    return f"toy:{i}:{i:016x}"


def episodes(arm: str, n: int, score: Callable[[int, int], float], reps: Iterable[int] = (1,),
             block: Callable[[int, int], int] = lambda i, r: r, tokens: Callable[[int, int], int] = lambda i, r: 1000,
             latency: Callable[[int, int], Optional[float]] = lambda i, r: 10.0,
             actions: Callable[[int, int], list] = lambda i, r: ["search[x]", "click[buy now]"],
             skip: Iterable[tuple] = (), infra: Iterable[tuple] = (), legacy: bool = False):
    skip, infra = set(skip), set(infra)
    out = []
    for i in range(n):
        for r in reps:
            if (i, r) in skip:
                continue
            common = dict(run_id=f"{arm}_r{r}", arm=arm, task_id=str(i), task_uid=uid(i),
                          task_type=TYPES[i % len(TYPES)], seed=0,
                          repetition=None if legacy else r, block=None if legacy else block(i, r))
            att = 1
            if (i, r) in infra:
                out.append(Episode(**common, attempt=att, infra_error="server 502", score=None))
                att += 1
            s = float(score(i, r))
            acts = actions(i, r)
            out.append(Episode(**common, attempt=att, score=s, success=s == 1.0, steps=len(acts),
                               prompt_tokens=int(tokens(i, r)), completion_tokens=50, latency_s=latency(i, r),
                               n_calls=len(acts), n_retries=0,
                               trajectory=[Step(step=k + 1, action=a, obs="") for k, a in enumerate(acts)]))
    return out


def manifest(run_id: str, **cfg) -> RunManifest:
    config = {"env": "toy", "model": "toy-8b", "memory_pool": None, **cfg}
    files = {k: f"sha-{v}" for k, v in config.items() if isinstance(v, str) and v.endswith(".jsonl")}
    return RunManifest(run_id=run_id, config=config, config_sha256=config_hash(config), model="toy-8b",
                       model_revision="rev-1", scorer_version="skillrc-test",
                       serving={"engine": "toy", "engine_version": "1"}, files_sha256=files)


def arm(name: str, eps, n: int, reps=(1,), man: Optional[RunManifest] = None, grid: bool = True,
        max_missing_frac: float = 0.005, max_infra_frac: float = 0.02) -> ArmData:
    g = Grid(task_range=[0, n], repetitions=list(reps)) if grid else None
    return ArmData(name, eps, man, g, max_missing_frac=max_missing_frac, max_infra_frac=max_infra_frac)


def spec(targets: Dict, arms: Iterable[str], interventions: Optional[Dict] = None, contrasts=None, claims=None,
         draws: int = 4000, **kw) -> GateSpec:
    return GateSpec.model_validate({
        "name": "toy", "bootstrap_draws": draws, "seed": 7,
        "arms": {a: {"logs": ["-"]} for a in arms},
        "interventions": interventions or {},
        "targets": targets, "contrasts": contrasts or {}, "claims": claims or {}, **kw})


@pytest.fixture
def rng():
    return np.random.default_rng(0)
