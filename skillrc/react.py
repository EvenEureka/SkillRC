"""ReAct agent loop. Two prompt styles, held constant across memory systems:

- "thought_action" (generic / mock / WebShop): model emits 'Thought:'/'Action:'.
- "react_alfworld" (faithful ReAct-ALFWorld): completion-style transcript with
  '> ' turns; a line starting with 'think:' is a reasoning step that does NOT
  consume an env action (env replies 'OK.'), matching Yao et al. 2-shot exemplars
  selected by task type. This is what makes no-mem ReAct non-degenerate on ALFWorld.

Prompt layout is cache-friendly: instruction + few-shot + retrieved memory sit in
the system message (stable within an episode); the growing transcript is the user
message.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SYSTEM_PROMPTS = {
    "default": (
        "You are an agent solving an interactive text task using the ReAct framework.\n"
        "At each step: first write one short line starting with 'Thought:' with your "
        "reasoning, then output exactly one line starting with 'Action:' containing a "
        "single admissible action. Output nothing after the Action line."
    ),
}

_ACTION_RE = re.compile(r"Action:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

# task-type key -> ReAct prompt family (keys in alfworld_react_fewshot.json)
_ALFWORLD_TYPES = ("put", "clean", "heat", "cool", "puttwo", "examine")


def parse_action(text: str) -> Optional[str]:
    matches = _ACTION_RE.findall(text or "")
    if matches:
        return matches[-1].strip().strip("`").strip()
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else None


def parse_react_line(text: str) -> str:
    """First meaningful line of a ReAct-ALFWorld completion, '>' stripped."""
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith(">"):
            s = s[1:].strip()
        if s:
            return s
    return "look"


@dataclass
class StepRecord:
    step: int
    action: str
    obs: str
    reward: float
    prompt_tokens: int
    completion_tokens: int


@dataclass
class EpisodeResult:
    task_id: Any
    task_type: str
    success: bool
    reward: float
    steps: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    seed: int = 0
    memory_tokens: int = 0        # realized injected-memory tokens (budget audit)
    error: str = ""
    task_text: str = ""           # initial task/observation (for pool serialization)
    trajectory: List[StepRecord] = field(default_factory=list)


class ReActAgent:
    def __init__(self, llm, memory=None, max_steps: int = 30, n_fewshot: int = 2,
                 system_prompt_key: str = "default", fewshot_examples=None,
                 prompt_style: str = "thought_action", fewshot_prompts: dict = None,
                 inject_admissible: bool = False):
        self.llm = llm
        self.memory = memory
        self.max_steps = max_steps
        self.n_fewshot = n_fewshot
        self.style = prompt_style
        self.system = SYSTEM_PROMPTS.get(system_prompt_key, SYSTEM_PROMPTS["default"])
        self.fewshot = (fewshot_examples or [])[:n_fewshot]     # list[str] (generic)
        self.fewshot_prompts = fewshot_prompts or {}            # dict (react_alfworld)
        self.inject_admissible = inject_admissible

    # ---- generic Thought/Action path (mock, WebShop) ----
    def _messages(self, task_obs: str, history_text: str, memory_ctx: str,
                  admissible_actions=None):
        sys = self.system
        if self.fewshot:
            sys += "\n\n# Examples\n" + "\n\n".join(self.fewshot)
        if memory_ctx:
            sys += "\n\n# Retrieved experience / skills\n" + memory_ctx
        admissible = ""
        if self.inject_admissible and admissible_actions:
            admissible = "\n\n# Valid actions right now\n" + "; ".join(admissible_actions)
        user = (f"{task_obs}{admissible}\n\n# Interaction so far\n{history_text}\n\n"
                "What is your next Thought and Action?")
        return [{"role": "system", "content": sys},
                {"role": "user", "content": user}]

    def _run_generic(self, env, obs, seed):
        mem_ctx = self.memory.retrieve(obs.text, seed=seed) if self.memory else ""
        mem_tokens = getattr(self.memory, "last_tokens", 0) if self.memory else 0
        history, traj = [], []
        total_reward, success, pt, ct = 0.0, False, 0, 0
        cost0 = self.llm.usage.cost_usd
        cur_adm = list(obs.admissible_actions or [])
        step = 0
        for step in range(1, self.max_steps + 1):
            history_text = "\n".join(history) if history else "(none yet)"
            resp = self.llm.complete(
                self._messages(obs.text, history_text, mem_ctx, cur_adm)
            )
            action = parse_action(resp.text) or "look"
            sr = env.step(action)
            cur_adm = list((sr.info or {}).get("admissible_actions") or cur_adm)
            total_reward += sr.reward
            pt += resp.prompt_tokens
            ct += resp.completion_tokens
            history.append(f"Action: {action}\nObservation: {sr.obs_text}")
            traj.append(StepRecord(step, action, sr.obs_text, sr.reward,
                                   resp.prompt_tokens, resp.completion_tokens))
            if sr.done:
                success = sr.success
                break
        return self._result(obs, success, total_reward, step, pt, ct, cost0, seed,
                            mem_tokens, traj)

    # ---- faithful ReAct-ALFWorld path ----
    def _select_examples(self, task_type: str, seed: int = 0) -> str:
        """Seeded selection of n_fewshot exemplars from the {0,1,2} available per
        type. Seeding gives every system a real seed axis under a temp-0 executor
        (findings [3]/[8]); the exemplar COUNT stays fixed = n_fewshot."""
        t = task_type if task_type in _ALFWORLD_TYPES else "put"
        avail = [f"react_{t}_{i}" for i in range(3) if f"react_{t}_{i}" in self.fewshot_prompts]
        if not avail:
            return ""
        order = list(avail)
        random.Random(seed).shuffle(order)
        keys = sorted(order[:self.n_fewshot])  # sorted => stable prompt prefix for caching
        return "\n\n".join(self.fewshot_prompts[k] for k in keys)

    def _run_react_alfworld(self, env, obs, seed):
        examples = self._select_examples(obs.task_type, seed)
        instruction = (
            "Interact with a household to solve a task. Output only the next line: "
            "either an action, or a line beginning with 'think:' to reason (a 'think:' "
            "line is not an environment action). Here are two examples.\n\n"
            + examples + "\n\nHere is the task.")
        mem_ctx = self.memory.retrieve(obs.text, seed=seed) if self.memory else ""
        mem_tokens = getattr(self.memory, "last_tokens", 0) if self.memory else 0
        if mem_ctx:
            instruction += "\n\n# Retrieved experience / skills\n" + mem_ctx

        transcript = obs.text
        cur_adm = list(obs.admissible_actions or [])
        traj = []
        total_reward, success, pt, ct = 0.0, False, 0, 0
        cost0 = self.llm.usage.cost_usd
        env_steps = 0
        max_calls = 2 * self.max_steps  # bound thinks + actions
        for _ in range(max_calls):
            adm_block = ""
            if self.inject_admissible and cur_adm:
                adm_block = "\nValid actions right now: " + "; ".join(cur_adm) + "\n"
            messages = [{"role": "system", "content": instruction},
                        {"role": "user", "content": transcript + adm_block + "\n>"}]
            resp = self.llm.complete(messages)
            pt += resp.prompt_tokens
            ct += resp.completion_tokens
            line = parse_react_line(resp.text)
            if line.lower().startswith("think:"):
                transcript += f"\n> {line}\nOK."
                traj.append(StepRecord(env_steps, line, "OK.", 0.0,
                                       resp.prompt_tokens, resp.completion_tokens))
                continue
            env_steps += 1
            sr = env.step(line)
            total_reward += sr.reward
            cur_adm = list((sr.info or {}).get("admissible_actions") or cur_adm)
            transcript += f"\n> {line}\n{sr.obs_text}"
            traj.append(StepRecord(env_steps, line, sr.obs_text, sr.reward,
                                   resp.prompt_tokens, resp.completion_tokens))
            if sr.done:
                success = sr.success
                break
            if env_steps >= self.max_steps:
                break
        return self._result(obs, success, total_reward, env_steps, pt, ct, cost0, seed,
                            mem_tokens, traj)

    def _result(self, obs, success, reward, steps, pt, ct, cost0, seed, mem_tokens, traj):
        return EpisodeResult(
            task_id=obs.task_id, task_type=obs.task_type, success=success,
            reward=reward, steps=steps, prompt_tokens=pt, completion_tokens=ct,
            cost_usd=self.llm.usage.cost_usd - cost0, seed=seed,
            memory_tokens=mem_tokens, task_text=obs.text, trajectory=traj)

    def run_episode(self, env, task_id, seed: int = 0) -> EpisodeResult:
        obs = env.reset(task_id)
        if self.style == "react_alfworld":
            return self._run_react_alfworld(env, obs, seed)
        return self._run_generic(env, obs, seed)
