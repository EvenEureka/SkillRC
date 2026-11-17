"""Gate spec: arms, declared interventions, release targets, diagnostic contrasts, joint claims, expected grid."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

METRICS = ("score", "success", "steps", "prompt_tokens", "completion_tokens", "total_tokens",
           "memory_tokens", "latency_s", "n_calls")


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Grid(_M):
    """Expected cells of an arm: tasks x seeds x repetitions. With a `registry` (JSONL of task_id, task_uid,
    task_type, frozen before the runs), task identities and strata come from it, not from the episodes."""

    registry: Optional[str] = None
    task_ids: Optional[List[str]] = None
    task_range: Optional[List[int]] = Field(default=None, description="[start, stop) of integer task ids")
    task_ids_file: Optional[str] = None
    seeds: List[int] = Field(default_factory=lambda: [0])
    repetitions: List[int] = Field(default_factory=lambda: [1])

    def resolve_task_ids(self, base: Path) -> List[str]:
        if self.task_ids is not None:
            return [str(t) for t in self.task_ids]
        if self.task_range is not None:
            a, b = self.task_range
            return [str(t) for t in range(a, b)]
        if self.task_ids_file is not None:
            p = (base / self.task_ids_file)
            return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        raise ValueError("grid needs task_ids, task_range or task_ids_file")


class ArmSpec(_M):
    logs: List[str]
    format: Literal["skillrc", "inspect", "episodes"] = "skillrc"
    manifest: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict,
                                    description="adapter options, e.g. inspect: score_name, value_key, task_type_key")
    grid: Optional[Grid] = None
    label: Optional[str] = None


class Intervention(_M):
    """A declared change: config fields it sets, and the sha256 of any file it adds."""

    description: str = ""
    config_patch: Dict[str, Any] = Field(default_factory=dict)
    files_sha256: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="manifest file key -> expected producer-time sha256 (null: declared to change, pinned elsewhere)")


class Primary(_M):
    metric: Literal[METRICS] = "score"  # type: ignore[valid-type]
    margin: float = Field(ge=0)


class Guardrail(_M):
    metric: Literal[METRICS]  # type: ignore[valid-type]
    statistic: Literal["relative_mean", "relative_p50", "relative_p95"]
    tolerance: float

    @property
    def name(self) -> str:
        return f"{self.metric}:{self.statistic}"


class Target(_M):
    description: str = ""
    baseline: str
    candidate: str
    interventions: List[str] = Field(default_factory=list)
    primary: Primary
    guardrails: List[Guardrail] = Field(default_factory=list)
    default: bool = False
    pairing: Literal["average", "same_block"] = "average"


class Contrast(_M):
    description: str = ""
    candidate: str
    baseline: Union[str, List[str]] = Field(description="an arm, or a list of arms: per-task mean of the differences")
    metric: Literal[METRICS] = "score"  # type: ignore[valid-type]
    role: Literal["diagnostic", "descriptive"] = "descriptive"
    claim: Literal["greater", "less", "none"] = "none"
    threshold: float = 0.0
    pairing: Literal["average", "same_block"] = "average"
    minus: Optional["MinusContrast"] = Field(default=None, description="difference of differences: subtract this "
                                             "contrast per task, e.g. the same change measured on another reader")


class MinusContrast(_M):
    candidate: str
    baseline: str


class JointClaim(_M):
    description: str = ""
    all_of: List[str]


class ReportOptions(_M):
    replays_worse: int = 3
    replays_random: int = 2
    replay_seed: int = 20261116


class GateSpec(_M):
    version: Literal[1] = 1
    name: str
    description: str = ""
    base_dir: Optional[str] = Field(default=None, description="paths are relative to this (itself relative to the spec file)")
    seed: int = 20261116
    bootstrap_draws: int = Field(default=20_000, ge=1000)
    alpha: float = Field(default=0.025, gt=0, lt=0.5)
    min_distinct: int = 20
    max_missing_frac: float = 0.005
    max_infra_frac: float = 0.02
    volatile_config_keys: List[str] = Field(default_factory=lambda: ["run_id", "out_dir", "repetition", "block"])
    arms: Dict[str, ArmSpec]
    interventions: Dict[str, Intervention] = Field(default_factory=dict)
    targets: Dict[str, Target]
    contrasts: Dict[str, Contrast] = Field(default_factory=dict)
    claims: Dict[str, JointClaim] = Field(default_factory=dict)
    report: ReportOptions = Field(default_factory=ReportOptions)

    _base: Path = Path(".")

    @model_validator(mode="after")
    def _refs(self):
        for name, t in self.targets.items():
            for a in (t.baseline, t.candidate):
                if a not in self.arms:
                    raise ValueError(f"target {name}: unknown arm {a!r}")
            for i in t.interventions:
                if i not in self.interventions:
                    raise ValueError(f"target {name}: unknown intervention {i!r}")
        for name, c in self.contrasts.items():
            extra = [c.minus.candidate, c.minus.baseline] if c.minus else []
            for a in [c.candidate] + ([c.baseline] if isinstance(c.baseline, str) else list(c.baseline)) + extra:
                if a not in self.arms:
                    raise ValueError(f"contrast {name}: unknown arm {a!r}")
        for name, j in self.claims.items():
            for c in j.all_of:
                if c not in self.contrasts:
                    raise ValueError(f"claim {name}: unknown contrast {c!r}")
                if self.contrasts[c].claim == "none":
                    raise ValueError(f"claim {name}: contrast {c!r} has no claim direction")
        return self

    @property
    def root(self) -> Path:
        """Directory that relative paths in the spec resolve against."""
        return self._base

    def default_target(self) -> Optional[str]:
        """The target used when --target is omitted: exactly one target must say `default: true`."""
        defaults = [n for n, t in self.targets.items() if t.default]
        return defaults[0] if len(defaults) == 1 else None


def load_spec(path: str | Path) -> GateSpec:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    spec = GateSpec.model_validate(data)
    spec._base = (path.parent / spec.base_dir).resolve() if spec.base_dir else path.parent.resolve()
    return spec


Contrast.model_rebuild()
