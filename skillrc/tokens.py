"""Token counting + budget trimming.

Uses tiktoken when available; falls back to a chars/4 heuristic so the harness
still imports on machines without it. All budget control (matched token budget B
in M1/M3) flows through here so it is measured consistently for every system.
"""
from __future__ import annotations

import functools

try:
    import tiktoken  # type: ignore
    _HAS_TIKTOKEN = True
except Exception:  # pragma: no cover
    _HAS_TIKTOKEN = False


@functools.lru_cache(maxsize=16)
def _enc(model: str):
    if not _HAS_TIKTOKEN:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        # Newer models (gpt-4.1/gpt-5*) use o200k_base.
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = _enc(model)
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text))


def count_message_tokens(messages, model: str = "gpt-4o-mini") -> int:
    """Approximate chat-format token count (content + ~4 tokens/message overhead)."""
    total = 0
    for m in messages:
        total += count_tokens(m.get("content", ""), model) + 4
    return total + 2


def trim_to_budget(text: str, budget_tokens: int, model: str = "gpt-4o-mini",
                   keep: str = "tail") -> str:
    """Trim text to at most `budget_tokens`. keep='tail' keeps the end (recent
    context), keep='head' keeps the beginning (earliest experiences)."""
    if budget_tokens is None or budget_tokens <= 0:
        return text
    enc = _enc(model)
    if enc is None:
        approx = budget_tokens * 4
        if len(text) <= approx:
            return text
        return text[-approx:] if keep == "tail" else text[:approx]
    toks = enc.encode(text)
    if len(toks) <= budget_tokens:
        return text
    kept = toks[-budget_tokens:] if keep == "tail" else toks[:budget_tokens]
    return enc.decode(kept)
