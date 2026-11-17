## skillrc-gate: WebShop: page label + distilled memory (historical, descriptive)

| target | verdict | primary Δ [95% interval] | margin | guardrails |
|---|---|---|---|---|
| bundle_vs_old | **DESCRIPTIVE** | +0.132 [+0.089, +0.174] | 0.05 | prompt_tokens:relative_mean +0.125 |
| label_only | **DESCRIPTIVE** | +0.251 [+0.208, +0.295] | 0.05 | prompt_tokens:relative_mean -0.276 |
| memory_given_label ◀ | **DESCRIPTIVE** | -0.119 [-0.164, -0.075] | 0.05 | prompt_tokens:relative_mean +0.554 |
| memory_old_harness | **DESCRIPTIVE** | -0.129 [-0.167, -0.091] | 0.05 | prompt_tokens:relative_mean +0.424 |

**bundle_vs_old** — Ship page label + memory together, compared with the old release
- not a release decision: arm L_PUB_TRUE: no repetition ids (legacy log); arm L_PUB_TRUE: no run manifest; arm L_PUB_TRUE: no declared grid; arm NONE: no repetition ids (legacy log); arm NONE: no run manifest; arm NONE: no declared grid
- evidence, primary score: lower bound +0.0888 > -margin -0.0500
- evidence, guardrail prompt_tokens:relative_mean: upper bound +0.1865 < tolerance +0.3000

**label_only** — Ship the page label alone
- not a release decision: arm L_NONE: no repetition ids (legacy log); arm L_NONE: no run manifest; arm L_NONE: no declared grid; arm NONE: no repetition ids (legacy log); arm NONE: no run manifest; arm NONE: no declared grid
- evidence, primary score: lower bound +0.2081 > -margin -0.0500
- evidence, guardrail prompt_tokens:relative_mean: upper bound -0.2304 < tolerance +0.3000

**memory_given_label** — Add the memory on top of the page-label harness
- not a release decision: arm L_PUB_TRUE: no repetition ids (legacy log); arm L_PUB_TRUE: no run manifest; arm L_PUB_TRUE: no declared grid; arm L_NONE: no repetition ids (legacy log); arm L_NONE: no run manifest; arm L_NONE: no declared grid
- evidence, primary score: upper bound -0.0746 < -margin -0.0500
- evidence, guardrail prompt_tokens:relative_mean: confirmed violation (Holm): estimate +0.5536, tolerance +0.3000

**memory_old_harness** — Add the memory to the old harness
- not a release decision: arm PUB_TRUE: no repetition ids (legacy log); arm PUB_TRUE: no run manifest; arm PUB_TRUE: no declared grid; arm NONE: no repetition ids (legacy log); arm NONE: no run manifest; arm NONE: no declared grid
- evidence, primary score: upper bound -0.0914 < -margin -0.0500
- evidence, guardrail prompt_tokens:relative_mean: confirmed violation (Holm): estimate +0.4239, tolerance +0.3000

| claim | evaluable | holds |
|---|---|---|
| memory_content_hurts_all_pools | False | False |

> Intervals describe task-sampling uncertainty conditional on the observed execution blocks. They do not quantify variability across future execution windows. Two blocks are insufficient to estimate that component reliably.
> Nominal error budget per target: one-sided 0.025 for the primary test and 0.025 for the guardrail family (Holm). This is not a finite-sample guarantee; percentile-bootstrap error rates can exceed nominal levels, notably for rare-event mixtures and small strata.
