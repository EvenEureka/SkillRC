## skillrc-gate: WebShop: distilled memory release check (Qwen3-8B)

| target | verdict | primary Δ [95% interval] | margin | guardrails |
|---|---|---|---|---|
| C1_ship_pub ◀ | **BLOCK** | -0.134 [-0.176, -0.092] | 0.05 | prompt_tokens:relative_mean +0.581; latency_s:relative_p95 +0.135 |
| C2_ship_repaired | **INCONCLUSIVE** | -0.074 [-0.119, -0.029] | 0.05 | prompt_tokens:relative_mean +0.385; latency_s:relative_p95 -0.010 |

**C1_ship_pub** — Ship the published distilled memory
- primary score: BLOCK (upper bound -0.0920 < -margin -0.0500)
- guardrail prompt_tokens:relative_mean: BLOCK (confirmed violation (Holm): estimate +0.5813, tolerance +0.3000)
- guardrail latency_s:relative_p95: PASS (upper bound +0.2250 < tolerance +0.3000)

**C2_ship_repaired** — Ship the repaired memory
- primary score: INCONCLUSIVE (interval [-0.1189, -0.0288] straddles -margin -0.0500)
- guardrail prompt_tokens:relative_mean: INCONCLUSIVE (upper bound +0.4807 not below tolerance +0.3000; violation not confirmed)
- guardrail latency_s:relative_p95: PASS (upper bound +0.0348 < tolerance +0.3000)

| claim | evaluable | holds |
|---|---|---|
| localization | True | False |

> Intervals describe task-sampling uncertainty conditional on the observed execution blocks. They do not quantify variability across future execution windows. Two blocks are insufficient to estimate that component reliably.
> Nominal error budget per target: one-sided 0.025 for the primary test and 0.025 for the guardrail family (Holm). This is not a finite-sample guarantee; percentile-bootstrap error rates can exceed nominal levels, notably for rare-event mixtures and small strata.
