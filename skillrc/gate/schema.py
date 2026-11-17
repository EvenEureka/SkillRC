"""Episode schema v1 and run manifest.

One `Episode` is one attempt of one cell (arm x task_uid x repetition). Every adapter converts its source format
into these models; everything downstream reads only these models.

Release-eligible logs carry a `RunManifest` and explicit `repetition` / `attempt` fields. Logs without them are
accepted, but the verdict for any target that uses them is DESCRIPTIVE (see verdict.py).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    action: str
    obs: str = ""
    reward: Optional[float] = None


class Episode(BaseModel):
    """One attempt of one cell."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    arm: str
    task_id: str
    task_uid: str = Field(description="Stable identity: env + split + task id + hash of the task text.")
    task_type: str = "all"
    seed: int = 0
    repetition: Optional[int] = Field(default=None, ge=1, description="None = legacy log (descriptive only).")
    block: Optional[int] = Field(default=None, ge=1, description="Execution window the attempt ran in.")
    attempt: int = Field(default=1, ge=1)
    score: Optional[float] = None
    success: Optional[bool] = None
    steps: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    memory_tokens: Optional[int] = None
    n_calls: Optional[int] = None
    n_retries: Optional[int] = None
    latency_s: Optional[float] = Field(default=None, ge=0)
    infra_error: Optional[str] = Field(default=None, description="Infrastructure failure; the attempt is not scored.")
    agent_error: Optional[str] = Field(default=None, description="Agent-side failure; the attempt is scored as is.")
    trajectory: Optional[List[Step]] = None

    @field_validator("score")
    @classmethod
    def _score_range(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def _scored_needs_score(self):
        if self.infra_error is None and self.score is None:
            raise ValueError("an attempt without infra_error must have a score")
        return self

    @property
    def scored(self) -> bool:
        return self.infra_error is None


class RunManifest(BaseModel):
    """Provenance for one run (one arm, possibly several repetitions/blocks)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    config: Dict[str, Any] = Field(description="Normalized run configuration (not only its hash).")
    config_sha256: str
    model: str
    model_revision: Optional[str] = None
    serving: Dict[str, Any] = Field(default_factory=dict)
    env: Dict[str, Any] = Field(default_factory=dict)
    memory: Dict[str, Any] = Field(default_factory=dict)
    files_sha256: Dict[str, str] = Field(default_factory=dict,
                                         description="config key whose value is a file path -> sha256 of that file, "
                                                     "recorded by the producer at execution time")
    scorer_version: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    host: Optional[str] = None
    segments: List[Dict[str, Any]] = Field(default_factory=list,
                                           description="one record per launch of the run (append-only execution metadata)")

    @model_validator(mode="after")
    def _hash_matches(self):
        if config_hash(self.config) != self.config_sha256:
            raise ValueError("config_sha256 does not match the normalized config")
        return self


# Execution metadata: may differ between launches, chains and arms without changing what was measured.
EXECUTION_SERVING_KEYS = {"base_url", "model_base_url", "gpu_index", "gpu_uuid", "chain", "block", "job_id", "port"}
EXECUTION_ENV_KEYS = {"webshop_base_url", "host"}


def fingerprint(m: RunManifest, ignore_config=(), ignore_files=()) -> Dict[str, Any]:
    """What was measured: config, every file hash, model, scorer, and the scientific part of serving and env."""
    ic, ifl = set(ignore_config), set(ignore_files)
    return {
        "config": {k: v for k, v in m.config.items() if k not in ic},
        "files_sha256": {k: v for k, v in m.files_sha256.items() if k not in ifl},
        "model": None if "model" in ic else m.model,
        "model_revision": None if "model" in ic or "model_revision" in ic else m.model_revision,
        "scorer_version": m.scorer_version,
        "serving": {k: v for k, v in m.serving.items() if k not in EXECUTION_SERVING_KEYS},
        "env": {k: v for k, v in m.env.items() if k not in EXECUTION_ENV_KEYS},
    }


def fingerprint_diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    out = []
    for sec in sorted(set(a) | set(b)):
        x, y = a.get(sec), b.get(sec)
        if isinstance(x, dict) and isinstance(y, dict):
            out += [f"{sec}.{k}" for k in sorted(set(x) | set(y)) if x.get(k) != y.get(k)]
        elif x != y:
            out.append(sec)
    return out


def provenance_gaps(m: "RunManifest") -> List[str]:
    """Fields a manifest must carry for a release verdict. Equal missing information is not evidence of equal
    experiments, so a manifest with gaps makes its targets DESCRIPTIVE."""
    gaps = []
    if not m.model:
        gaps.append("model")
    if not m.model_revision:
        gaps.append("model_revision")
    if not m.serving.get("engine_version"):
        gaps.append("serving.engine_version")
    if not m.config:
        gaps.append("config")
    is_inspect = "inspect_ai" in m.env
    if is_inspect:
        if not m.env.get("inspect_ai"):
            gaps.append("env.inspect_ai")
    else:
        if not m.scorer_version:
            gaps.append("scorer_version")
        if m.config.get("env") == "webshop":
            for k in ("webshop_server_seed", "instructions_sha256", "server_script_sha256"):
                if m.env.get(k) is None:
                    gaps.append(f"env.{k}")
    return gaps


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def config_hash(config: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def text_hash(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_schemas() -> Dict[str, Any]:
    """JSON Schemas of the public models (for `skillrc-gate schema`)."""
    return {"Episode": Episode.model_json_schema(), "RunManifest": RunManifest.model_json_schema()}
