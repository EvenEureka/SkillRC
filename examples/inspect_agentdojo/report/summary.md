## skillrc-gate: AgentDojo: appending a tips block to the system prompt (Inspect AI, Qwen3-8B)

| target | verdict | primary Δ [95% interval] | margin | guardrails |
|---|---|---|---|---|
| add_tips ◀ | **INCONCLUSIVE** | +0.026 [-0.047, +0.099] | 0.05 | prompt_tokens:relative_mean +0.212; latency_s:relative_p95 -0.145 |

**add_tips** — Ship the tips block
- primary score: PASS (lower bound -0.0469 > -margin -0.0500)
- guardrail prompt_tokens:relative_mean: INCONCLUSIVE (upper bound +0.5920 not below tolerance +0.3000; violation not confirmed)
- guardrail latency_s:relative_p95: INCONCLUSIVE (upper bound +0.5349 not below tolerance +0.3000; violation not confirmed)

> Intervals describe task-sampling uncertainty conditional on the observed execution blocks. They do not quantify variability across future execution windows. Two blocks are insufficient to estimate that component reliably.
> Nominal error budget per target: one-sided 0.025 for the primary test and 0.025 for the guardrail family (Holm). This is not a finite-sample guarantee; percentile-bootstrap error rates can exceed nominal levels, notably for rare-event mixtures and small strata.
