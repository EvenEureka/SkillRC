#!/usr/bin/env python3
"""Audit true/placebo footprint with the GPT and native Qwen tokenizers."""
from __future__ import annotations

import json
from pathlib import Path

import tiktoken
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]


def load_context(path: Path) -> str:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return "\n\n".join(row["text"] for row in rows)


def main() -> None:
    original = load_context(ROOT / "data" / "pools" / "expel_insights.jsonl")
    placebo = load_context(ROOT / "data" / "pools" / "expel_insights_order_placebo.jsonl")
    gpt = tiktoken.encoding_for_model("gpt-4o-mini")
    qwen = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    audit = {
        "gpt_4o_mini": {
            "original": len(gpt.encode(original)),
            "placebo": len(gpt.encode(placebo)),
        },
        "qwen3_8b_native": {
            "original": len(qwen.encode(original, add_special_tokens=False)),
            "placebo": len(qwen.encode(placebo, add_special_tokens=False)),
        },
    }
    if any(values["original"] != values["placebo"] for values in audit.values()):
        raise AssertionError(audit)
    out = ROOT / "results" / "placebo" / "order_placebo_tokenizer_audit.json"
    out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
