"""Frozen-executor LLM client (OpenAI-compatible) + usage/cost accounting.

The model is a config string; `provider='openai'` targets the real API OR any
OpenAI-compatible endpoint (set `base_url` for a local vLLM server serving
Qwen3-8B). `provider='mock'` runs offline with a supplied policy for zero-cost
plumbing tests. Cost is computed from the PRICES table below; edit it if OpenAI
changes prices.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# USD per 1,000,000 tokens: (input, cached_input, output).
# Verified against OpenAI pricing 2026-07; update if prices move. Local/vLLM
# models default to 0 cost (not in table).
PRICES: Dict[str, tuple] = {
    # legacy-but-cheap (still API-callable)
    "gpt-4o-mini":  (0.15, 0.075, 0.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1":      (2.00, 0.50, 8.00),
    "gpt-5-nano":   (0.05, 0.005, 0.40),
    "gpt-5-mini":   (0.25, 0.025, 2.00),
    "gpt-5":        (1.25, 0.125, 10.00),
    # current front-page family (2026-07)
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4":      (2.50, 0.25, 15.00),
    "gpt-5.5":      (5.00, 0.50, 30.00),
}


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    raw: Any = None


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, r: LLMResponse) -> None:
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.cached_tokens += r.cached_tokens
        self.cost_usd += r.cost_usd


def price_of(model: str, prompt_tokens: int, completion_tokens: int,
             cached_tokens: int = 0) -> float:
    key = model if model in PRICES else None
    if key is None:
        base = re.sub(r"-20\d{2}.*$", "", model)  # strip date suffix
        key = base if base in PRICES else None
    if key is None:
        return 0.0
    p_in, p_cached, p_out = PRICES[key]
    billed_in = max(0, prompt_tokens - cached_tokens)
    return (billed_in * p_in + cached_tokens * p_cached + completion_tokens * p_out) / 1e6


def embed_texts(texts, model="text-embedding-3-small", api_key=None,
                base_url=None, batch=256):
    """Embed texts via an OpenAI-compatible embeddings endpoint (for dense retrieval)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url)
    out = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=model, input=texts[i:i + batch])
        out.extend(d.embedding for d in resp.data)
    return out


class LLMClient:
    def __init__(self, model: str, provider: str = "openai", temperature: float = 0.0,
                 max_completion_tokens: int = 256, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, request_timeout: float = 120.0,
                 max_retries: int = 5, mock_policy: Optional[Callable] = None,
                 extra_body: Optional[Dict[str, Any]] = None):
        self.model = model
        # Passed through to OpenAI-compatible servers, e.g.
        # {"chat_template_kwargs": {"enable_thinking": False}} for Qwen3 on vLLM.
        self.extra_body = extra_body
        self.provider = provider
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.usage = Usage()
        self._mock_policy = mock_policy
        self._client = None
        if provider == "openai":
            from openai import OpenAI
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key and not base_url:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. `export OPENAI_API_KEY=sk-...` "
                    "or set base_url for a local endpoint.")
            self._client = OpenAI(api_key=key or "EMPTY", base_url=base_url,
                                  timeout=request_timeout)

    def complete(self, messages: List[Dict[str, str]]) -> LLMResponse:
        if self.provider == "mock":
            return self._complete_mock(messages)
        return self._complete_openai(messages)

    def _complete_mock(self, messages) -> LLMResponse:
        from .tokens import count_message_tokens, count_tokens
        text = self._mock_policy(messages) if self._mock_policy else "Action: look"
        r = LLMResponse(text=text,
                        prompt_tokens=count_message_tokens(messages, self.model),
                        completion_tokens=count_tokens(text, self.model),
                        cost_usd=0.0)
        self.usage.add(r)
        return r

    def _complete_openai(self, messages) -> LLMResponse:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs: Dict[str, Any] = dict(model=self.model, messages=messages,
                                              max_completion_tokens=self.max_completion_tokens)
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                resp = self._client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                u = resp.usage
                pt = int(getattr(u, "prompt_tokens", 0) or 0)
                ct = int(getattr(u, "completion_tokens", 0) or 0)
                cached = 0
                details = getattr(u, "prompt_tokens_details", None)
                if details is not None:
                    cached = int(getattr(details, "cached_tokens", 0) or 0)
                r = LLMResponse(text=text, prompt_tokens=pt, completion_tokens=ct,
                                cached_tokens=cached,
                                cost_usd=price_of(self.model, pt, ct, cached), raw=resp)
                self.usage.add(r)
                return r
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                # Some reasoning models reject temperature != 1: drop it and retry.
                if ("temperature" in msg and
                        ("unsupported" in msg or "does not support" in msg
                         or "only the default" in msg)):
                    self.temperature = None
                time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")
