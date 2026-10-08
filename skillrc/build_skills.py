"""Distill the raw M1 pool into method-specific REPRESENTATIONS (plan B2 factor 2),
for the M2 reproductions of ExpeL and SkillOS under a unified retrieval+budget:

  - ExpeL   : natural-language INSIGHTS (rules) per task type -> expel_insights.jsonl
  - SkillOS : Markdown SKILL per successful trajectory        -> skillos_skills.jsonl

Source = the same valid_seen success pool as the raw_traj confound (disjoint from the
unseen eval), and downstream retrieval + token budget B are IDENTICAL to raw_traj. So
the only thing that differs vs raw_traj is the representation — this is what isolates
ExpeL/SkillOS's contribution from "just more retrieved context". These are unified-
protocol representation reproductions, not full re-implementations (the plan cuts the
methods' training/curators).

Usage:
  python -m skillrc.build_skills --pool results/pools/alfworld_pool.jsonl \
      --expel-out results/pools/expel_insights.jsonl \
      --skill-out results/pools/skillos_skills.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os

from .config import RunConfig
from .llm import LLMClient

INSIGHT_PROMPT = (
    "Below are successful agent trajectories for ALFWorld '{task_type}' tasks.\n"
    "Extract 3 to 6 concise, GENERAL insights (rules/heuristics) that help solve this "
    "task type. Each insight: one imperative line, no specific object numbers or room "
    "layouts. Output ONLY a bullet list, each line starting with '- '.\n\n"
    "Trajectories:\n{trajs}"
)
SKILL_PROMPT = (
    "Below is a successful agent trajectory in ALFWorld.\n"
    "Distill it into ONE reusable skill in Markdown with exactly:\n"
    "'## Skill: <short name>', a 'When to use:' line, and a 'Steps:' numbered list "
    "(generalized — describe the procedure, not the specific object numbers). "
    "Keep it under 110 words.\n\nTrajectory:\n{traj}"
)


def _bullets(text: str):
    out = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if s.startswith(("-", "*", "•")):
            out.append(s.lstrip("-*• ").strip())
    return [b for b in out if b]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m0_alfworld.yaml")
    ap.add_argument("--pool", default="results/pools/alfworld_pool.jsonl")
    ap.add_argument("--expel-out", default="results/pools/expel_insights.jsonl")
    ap.add_argument("--skill-out", default="results/pools/skillos_skills.jsonl")
    ap.add_argument("--max-per-type", type=int, default=6,
                    help="trajectories per type fed to the insight prompt")
    args = ap.parse_args(argv)

    cfg = RunConfig.load(args.config)
    llm = LLMClient(model=cfg.model, provider=cfg.provider, temperature=cfg.temperature,
                    max_completion_tokens=512, max_retries=cfg.max_retries)

    pool = [json.loads(l) for l in open(args.pool) if l.strip()]
    by_type = collections.defaultdict(list)
    for r in pool:
        by_type[r["task_type"]].append(r["text"])
    print(f"[distill] pool={len(pool)} types={dict((k,len(v)) for k,v in by_type.items())}",
          flush=True)

    # --- ExpeL: NL insights per task type ---
    os.makedirs(os.path.dirname(args.expel_out), exist_ok=True)
    n_ins = 0
    with open(args.expel_out, "w") as f:
        for t, trajs in sorted(by_type.items()):
            joined = "\n\n---\n\n".join(trajs[:args.max_per_type])
            resp = llm.complete([{"role": "user",
                                  "content": INSIGHT_PROMPT.format(task_type=t, trajs=joined)}])
            for b in _bullets(resp.text):
                f.write(json.dumps({"text": f"[{t}] {b}", "task_type": t},
                                   ensure_ascii=False) + "\n")
                n_ins += 1
            print(f"[expel] {t}: +{len(_bullets(resp.text))} insights", flush=True)

    # --- SkillOS: Markdown skill per successful trajectory ---
    n_sk = 0
    with open(args.skill_out, "w") as f:
        for r in pool:
            resp = llm.complete([{"role": "user",
                                  "content": SKILL_PROMPT.format(traj=r["text"])}])
            skill = resp.text.strip()
            if skill:
                f.write(json.dumps({"text": skill, "task_type": r["task_type"]},
                                   ensure_ascii=False) + "\n")
                n_sk += 1
    print(f"[distill] wrote {n_ins} insights -> {args.expel_out}", flush=True)
    print(f"[distill] wrote {n_sk} skills   -> {args.skill_out}", flush=True)
    print(f"[distill] cost=${llm.usage.cost_usd:.4f} calls={llm.usage.calls}", flush=True)
    print("SKILLRC_DISTILL_DONE", flush=True)


if __name__ == "__main__":
    main()
