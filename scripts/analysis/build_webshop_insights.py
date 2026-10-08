#!/usr/bin/env python3
"""Distill train-only WebShop trajectories into a compact ExpeL-style pool."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from skillrc.llm import LLMClient


PROMPT = """You are distilling reusable procedural insights for the WebShop text environment.
The examples below come only from training sessions disjoint from evaluation.
Extract 6 to 10 concise general rules that improve product search, attribute checking,
option selection, and purchase decisions. Do not mention product IDs, session IDs, or
specific products. Output only bullet lines beginning with '- '.

Training experiences:
{experiences}
"""


def objective(text: str) -> str:
    marker = "Instruction: [SEP]"
    if marker in text:
        return text.split(marker, 1)[1].split("[SEP]", 1)[0].strip()
    return text[:500]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.episodes).read_text().splitlines() if line.strip()]
    eligible = [row for row in rows if not row.get("error") and float(row.get("reward", 0.0)) > 0]
    eligible.sort(key=lambda row: (float(row["reward"]), int(row["success"])), reverse=True)
    selected = eligible[:12]
    if len(selected) < 4:
        raise ValueError(f"need at least four positive-reward train trajectories, found {len(selected)}")

    blocks = []
    for row in selected:
        actions = [
            step["action"]
            for step in row.get("trajectory", [])
            if not step["action"].lower().startswith("think:")
        ]
        blocks.append(
            f"Goal: {objective(row['task_text'])}\nReward: {row['reward']:.3f}\nActions: "
            + " -> ".join(actions)
        )
    client = LLMClient(
        model="qwen3-8b",
        provider="openai",
        temperature=0.0,
        max_completion_tokens=512,
        base_url=args.base_url,
        api_key="EMPTY",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    response = client.complete([{"role": "user", "content": PROMPT.format(experiences="\n\n---\n\n".join(blocks))}])
    insights = []
    for line in response.text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if cleaned and cleaned != line.strip() or line.lstrip().startswith(("-", "*")):
            insights.append(cleaned)
    insights = [item for item in insights if len(item.split()) >= 4]
    if len(insights) < 4:
        raise ValueError(f"distillation returned only {len(insights)} usable insights: {response.text}")
    insights = insights[:10]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for insight in insights:
            handle.write(json.dumps({"text": f"[webshop_buy] {insight}", "task_type": "webshop_buy"}) + "\n")
    metadata = {
        "source": str(args.episodes),
        "source_sessions": [int(row["task_id"]) for row in selected],
        "evaluation_sessions": "0-99",
        "disjoint": all(int(row["task_id"]) >= 500 for row in selected),
        "positive_reward_train_trajectories": len(eligible),
        "selected_trajectories": len(selected),
        "n_insights": len(insights),
        "external_api": False,
    }
    Path(args.metadata).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
