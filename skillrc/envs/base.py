"""Minimal environment interface shared by the mock, ALFWorld, and WebShop adapters.

Reconstructed during code recovery from the call sites in ``react.py`` and the three
adapters; it carries no behaviour beyond the fields those call sites read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Observation:
    text: str
    task_id: Any = None
    task_type: Optional[str] = None
    admissible_actions: Optional[List[str]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    obs_text: str
    reward: float = 0.0
    done: bool = False
    success: bool = False
    info: Optional[Dict[str, Any]] = None


class BaseEnv:
    name = "base"

    def task_ids(self) -> List[Any]:
        raise NotImplementedError

    def reset(self, task_id: Any) -> Observation:
        raise NotImplementedError

    def step(self, action: str) -> StepResult:
        raise NotImplementedError

    def close(self) -> None:
        pass
