"""WebShop adapter — HTTP mode (default).

Talks to scripts/webshop_server.py (running in the webshop conda env) over HTTP, so
the harness stays in skillrc (openai/pydantic-v2) and never imports WebShop's old
pydantic-v1 deps. Start the server first:
  ${WEBSHOP_PY} scripts/webshop_server.py --port 3000
Plan uses the fixed 100-item subset (sessions 0..99); reward is WebShop's 0..1 score,
success = score >= 1.0. UNVERIFIED until the server is up (validate at R002).
"""
from __future__ import annotations

from typing import Any, List

from .base import BaseEnv, Observation, StepResult


def _flatten_actions(actions) -> List[str]:
    if not actions:
        return []
    out = []
    if actions.get("has_search_bar"):
        out.append("search[<query>]")
    for c in actions.get("clickables", []) or []:
        if c and c != "search":
            out.append(f"click[{c}]")
    return out


class WebShopEnv(BaseEnv):
    name = "webshop"

    def __init__(self, n_tasks: int = 100, mode: str = "http",
                 base_url: str = "http://127.0.0.1:3000", timeout: float = 120.0, **kw):
        self.n_tasks = n_tasks
        self.mode = mode
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if mode == "http":
            import requests  # noqa: F401
            self._rq = requests
            try:
                self._rq.get(self.base_url + "/health", timeout=10)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"WebShop server not reachable at {self.base_url}. Start it with "
                    f"scripts/webshop_server.py in the webshop env. ({e})")
        elif mode == "in_process":
            from web_agent_site.envs import WebAgentTextEnv  # type: ignore
            self._env = WebAgentTextEnv(observation_mode="text", **kw)
        else:
            raise ValueError(f"unknown WebShop mode {mode!r}")

    def task_ids(self) -> List[Any]:
        return list(range(self.n_tasks))

    def reset(self, task_id: Any) -> Observation:
        if self.mode == "http":
            r = self._rq.post(self.base_url + "/reset", json={"session": int(task_id)},
                              timeout=self.timeout).json()
            return Observation(text=r["obs"], task_id=task_id, task_type="webshop_buy",
                               admissible_actions=_flatten_actions(r.get("actions")))
        obs = self._env.reset(session=task_id)
        obs = obs[0] if isinstance(obs, tuple) else obs
        return Observation(text=str(obs), task_id=task_id, task_type="webshop_buy")

    def step(self, action: str) -> StepResult:
        if self.mode == "http":
            r = self._rq.post(self.base_url + "/step", json={"action": action},
                              timeout=self.timeout).json()
            reward = float(r.get("reward", 0.0))
            done = bool(r.get("done"))
            return StepResult(obs_text=r["obs"], reward=reward, done=done,
                              success=bool(done and reward >= 1.0),
                              info={"admissible_actions": _flatten_actions(r.get("actions"))})
        obs, reward, done, info = self._env.step(action)
        obs = obs[0] if isinstance(obs, tuple) else obs
        reward = float(reward or 0.0)
        return StepResult(obs_text=str(obs), reward=reward, done=bool(done),
                          success=bool(done and reward >= 1.0))
