"""ALFWorld adapter (verified against the installed alfworld API).

Uses the version-robust factory `get_environment('AlfredTWEnv')` and the official
`configs/alfworld_base_config.yaml` (fetched during install), expanding
`$ALFWORLD_DATA`. The TextWorld batch env is sequential: `reset()` advances to the
next game, so `reset(task_id)` ignores task_id and we record the real gamefile in
`meta` for traceability. Default split `eval_out_of_distribution` == 134 unseen.

NOTE on seeds: with a temp-0 executor both the env and the LLM are deterministic,
so repeated seeds reproduce identical runs. Seeds only add variance if they perturb
few-shot selection or task order — handle that at the experiment-design level.
"""
from __future__ import annotations

import os
import re
from typing import Any, List

from .base import BaseEnv, Observation, StepResult

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# gamefile-path prefix -> ReAct prompt family key
_TYPE_MAP = [
    ("pick_two_obj", "puttwo"),
    ("look_at_obj", "examine"),
    ("pick_heat_then_place", "heat"),
    ("pick_cool_then_place", "cool"),
    ("pick_clean_then_place", "clean"),
    ("pick_and_place", "put"),
]


def _infer_task_type(gamefile: str) -> str:
    base = (gamefile or "").lower()
    for prefix, key in _TYPE_MAP:
        if prefix in base:
            return key
    return "unknown"


# This alfworld/TextWorld version's placement verb is "move X to Y" (verified via
# admissible_commands), NOT the ReAct exemplars' "put X in/on Y". Map put-phrasings
# to the valid grammar, uniformly for ALL systems so the fix cannot bias comparisons.
# (Belt-and-suspenders alongside inject_admissible, which shows the model the exact
# valid actions each step.)
_PUT_RE = re.compile(r"^put\s+(.+?)\s+(?:in/on|in|on)\s+(.+)$", re.IGNORECASE)


def _normalize_action(action: str) -> str:
    a = (action or "").strip()
    m = _PUT_RE.match(a)
    if m:
        return f"move {m.group(1)} to {m.group(2)}"
    return a


def _expandvars(obj):
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expandvars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expandvars(v) for v in obj]
    return obj


class ALFWorldEnv(BaseEnv):
    name = "alfworld"

    def __init__(self, split: str = "eval_out_of_distribution", max_steps: int = 30,
                 config_path: str = None, data_dir: str = None):
        try:
            import yaml
            from alfworld.agents.environment import get_environment
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "ALFWorld not installed. Run scripts/install_alfworld.sge on a node. "
                f"(import error: {e})")
        os.environ.setdefault(
            "ALFWORLD_DATA", data_dir or os.path.join(_REPO_ROOT, "data", "alfworld"))
        cfg_path = config_path or os.path.join(
            _REPO_ROOT, "configs", "alfworld_base_config.yaml")
        with open(cfg_path) as f:
            cfg = _expandvars(yaml.safe_load(f))
        self.split = split
        self.max_steps = max_steps
        env_type = cfg.get("env", {}).get("type", "AlfredTWEnv")
        self._builder = get_environment(env_type)(cfg, train_eval=split)
        self._n = int(getattr(self._builder, "num_games", 0))
        self._tw = self._builder.init_env(batch_size=1)
        self._consumed = 0          # #games drawn from the current iterator
        self._last_info = None

    def task_ids(self) -> List[Any]:
        return list(range(self._n))

    @staticmethod
    def _first(x):
        return x[0] if isinstance(x, (list, tuple)) else x

    def rewind(self) -> None:
        """Fresh iterator from game 0 (ALFWorld iterates gamefiles in fixed order),
        so reset(i) returns the same game across seeds/systems."""
        self._tw = self._builder.init_env(batch_size=1)
        self._consumed = 0

    def reset(self, task_id: Any) -> Observation:
        # Deterministically position the sequential iterator at game index `target`.
        target = int(task_id)
        if target < self._consumed:
            self.rewind()
        obs, info = None, None
        while self._consumed <= target:
            obs, info = self._tw.reset()
            self._consumed += 1
        self._last_info = info
        text = str(self._first(obs))
        adm = self._first(info.get("admissible_commands")) if info else None
        gamefile = self._first(info.get("extra.gamefile")) if info else None
        return Observation(text=text, task_id=target, task_type=_infer_task_type(gamefile),
                           admissible_actions=adm, meta={"gamefile": gamefile})

    def step(self, action: str) -> StepResult:
        action = _normalize_action(action)
        obs, scores, dones, info = self._tw.step([action])
        self._last_info = info
        text = str(self._first(obs))
        done = bool(self._first(dones))
        won = bool(self._first(info.get("won", [False]))) if info else False
        score = self._first(scores)
        reward = 1.0 if won else (float(score) if score is not None else 0.0)
        adm = self._first(info.get("admissible_commands")) if info else None
        return StepResult(obs_text=text, reward=reward, done=done, success=won,
                          info={"admissible_actions": adm})

    def close(self) -> None:
        try:
            self._tw.close()
        except Exception:
            pass
