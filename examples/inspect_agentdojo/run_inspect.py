"""Run the two AgentDojo arms through Inspect AI against an OpenAI-compatible endpoint (vLLM), one after the other.

Usage (inside the GPU job, after the server is up):
    LOCAL_BASE_URL=http://127.0.0.1:8000/v1 LOCAL_API_KEY=EMPTY \
    python run_inspect.py --model openai-api/local/qwen3-8b --epochs 2
Writes logs/BASE/*.eval, logs/TIPS/*.eval and task_ids.txt (the expected grid, from the dataset definition).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from inspect_ai import eval as inspect_eval

from agentdojo_tips import agentdojo_tips, tips_digest

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-connections", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N samples only (smoke test)")
    ap.add_argument("--arms", default="BASE,TIPS")
    a = ap.parse_args()

    ids = [str(s.id) for s in agentdojo_tips().dataset][: a.limit] if a.limit else \
        [str(s.id) for s in agentdojo_tips().dataset]
    (HERE / "task_ids.txt").write_text("\n".join(ids) + "\n")
    print(f"{len(ids)} samples x {a.epochs} epochs per arm")

    # producer-time provenance, set by run.sge: pinned model snapshot and the serving stack
    provenance = {"model_revision": os.environ["MODEL_REVISION"], "serving": json.loads(os.environ["SERVING_JSON"])}
    tips = str(HERE / "tips.md")
    for arm in a.arms.split(","):
        t = (agentdojo_tips(tips_file=tips, tips_sha256=tips_digest(Path(tips).read_text())) if arm == "TIPS"
             else agentdojo_tips())
        logs = inspect_eval(
            t, model=a.model, epochs=a.epochs, limit=a.limit, log_dir=str(HERE / "logs" / arm), log_format="eval",
            max_connections=a.max_connections, temperature=0.0, max_tokens=1024, display="plain",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}, fail_on_error=False,
            retry_on_error=2, metadata=provenance,
        )
        for lg in logs:
            print(arm, lg.status, lg.location, lg.results.scores[0].metrics if lg.results and lg.results.scores else "")


if __name__ == "__main__":
    main()
