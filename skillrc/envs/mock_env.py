"""Deterministic toy environment + oracle mock policy.

Purpose: exercise the FULL harness (prompt build -> LLM call -> action parse ->
env step -> metrics -> token accounting) with zero API cost and zero external
deps, so M0 plumbing can be verified before spending a cent or installing
ALFWorld/WebShop. Each task has a fixed gold action sequence; the observation
carries a clearly-labelled [MOCK-HINT] that the oracle mock policy follows, so a
correct harness yields 100% success deterministically. Nothing here is used with
a real LLM.
"""
from __future__ import annotations

import re
from typing import Any, List

from .base import BaseEnv, Observation, StepResult

_TASKS = {
    "mock-000": {
        "type": "pick_and_place",
        "goal": "put a clean mug in the cabinet",
        "gold": ["go to sinkbasin 1", "clean mug 1 with sinkbasin 1",
                 "go to cabinet 1", "put mug 1 in cabinet 1"],
        "distractors": ["go to fridge 1", "open drawer 1", "look"],
    },
    "mock-001": {
        "type": "examine_in_light",
        "goal": "examine the book under the desklamp",
        "gold": ["go to desk 1", "take book 1 from desk 1", "use desklamp 1"],
        "distractors": ["go to bed 1", "open safe 1", "look"],
    },
    "mock-002": {
        "type": "webshop_buy",
        "goal": "buy a blue cotton t-shirt under $20",
        "gold": ["search[blue cotton t-shirt]", "click[b09xyz]", "click[buy now]"],
        "distractors": ["click[back to search]", "click[next >]"],
    },
}


class MockEnv(BaseEnv):
    name = "mock"

    def __init__(self, **kw):
        self._ids = list(_TASKS.keys())
        self._task_id = None
        self._cur = None
        self._i = 0

    def task_ids(self) -> List[Any]:
        return list(self._ids)

    def reset(self, task_id: Any) -> Observation:
        self._task_id = task_id
        self._cur = _TASKS[task_id]
        self._i = 0
        return self._obs(first=True)

    def _gold(self):
        g = self._cur["gold"]
        return g[self._i] if self._i < len(g) else None

    def _obs(self, first: bool = False) -> Observation:
        gold = self._gold()
        adm = ([gold] if gold else []) + list(self._cur["distractors"])
        header = f"Task: {self._cur['goal']}\n" if first else ""
        hint = f"[MOCK-HINT] next correct action: {gold}\n" if gold else ""
        text = f"{header}Step {self._i}. {hint}Admissible actions: {adm}"
        return Observation(text=text, task_id=self._task_id,
                           task_type=self._cur["type"], admissible_actions=adm)

    def step(self, action: str) -> StepResult:
        gold = self._gold()
        action = (action or "").strip()
        if gold is not None and action == gold:
            self._i += 1
            if self._i >= len(self._cur["gold"]):
                return StepResult(obs_text="You succeeded.", reward=1.0,
                                  done=True, success=True)
            return StepResult(obs_text=self._obs().text, reward=0.1,
                              done=False, success=False)
        return StepResult(obs_text=f"Nothing happens. (invalid action: {action!r})",
                          reward=0.0, done=False, success=False)


_HINT_RE = re.compile(r"\[MOCK-HINT\] next correct action:\s*(.+)")


def mock_react_policy(messages) -> str:
    """Oracle policy that supports both generic Thought/Action and ReAct-ALFWorld.

    The M1 smoke path reuses ALFWorld-style prompting with the mock env, so we
    detect that prompt shape and emit a bare action line instead of a
    Thought/Action pair.
    """
    content = "\n".join(m.get("content", "") for m in messages)
    hints = _HINT_RE.findall(content)
    action = hints[-1].strip() if hints else "look"
    if content.rstrip().endswith(">"):
        return action
    return f"Thought: follow the mock hint.\nAction: {action}"
