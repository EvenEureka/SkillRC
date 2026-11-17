# `skillrc-gate` design v2 (2026-11-16, after Codex round 2)

Builds on GATE_DESIGN_v1.md; only the changed parts and the frozen parameters are written here. Reviews:
REVIEW_r1_codex.md, REVIEW_r2_codex.md. Internal; not committed. Nothing implemented or run yet.

## A. Design fixes

### A1. Hypothesis families, sidedness, precedence (round-2 item 2)

- All intervals: 95% two-sided, task-cluster bootstrap (20,000 draws), stratified by task type; one-sided tests
  at α = 0.025 read off the same interval's bound.
- The spec nominates **release-required contrasts**. Only these feed the verdict. Every other contrast is
  `confirmatory-diagnostic` (pre-registered, reported with its own test, never vetoes) or `descriptive`.
- For each release-required contrast:
  - Primary metric (higher is better), margin δ: PASS-component if lower bound > −δ; BLOCK-component if upper
    bound < −δ; else INCONCLUSIVE-component. "Detectable but tolerated regression" flag only here, when PASS and
    upper bound < 0.
  - Guardrails (increasing-cost metrics, tolerance τ_g each): family = all guardrails of that contrast.
    PASS needs every guardrail's unadjusted upper bound ≤ τ_g (intersection–union, no adjustment needed).
    Confirmed violation needs a Holm-adjusted one-sided p < 0.025 for "Δ_g > τ_g" (p from the bootstrap
    distribution); only a confirmed violation can BLOCK.
- Contrast verdict precedence: INVALID > BLOCK > INCONCLUSIVE > PASS (the worst component wins).
- Overall verdict = worst contrast verdict among release-required contrasts. Exit codes: 0 PASS, 1 INCONCLUSIVE,
  2 BLOCK, 3 INVALID, 4 DESCRIPTIVE (see A4). A CI job passes only on exit code 0.

### A2. What the intervals mean (round-2 item 3)

- Repetitions are averaged within task per arm; the paired arms stay together; tasks are the resampling unit.
- Episode-latency p50/p95 are computed on raw episodes (all repetitions) while resampling whole tasks.
- Per-block contrasts Δ₁, Δ₂ and Δ₂ − Δ₁ are always shown. The report prints, verbatim:
  "Intervals describe task-sampling uncertainty conditional on the two observed execution blocks. They do not
  quantify variability across future execution windows. Two blocks are insufficient to estimate that component
  reliably."
- Calibration claims come from simulation tests (known-effect synthetic paired data: coverage and verdict error
  rates). The real A/A pair (NONE r1 vs r2) is a diagnostic only.
- MDE statements always give α, sidedness, power, assumed true effect and the paired SD used.

### A3. Completeness and failures (round-2 item 5)

- The spec declares the expected grid: arm × task_uid × repetition. G4 rejects (INVALID) any missing cell or
  duplicate key not listed in a pre-analysis exclusions file with a reason.
- Failure types:
  - agent failure (invalid action, loop, budget exhausted, wrong item): scored as recorded;
  - infrastructure failure (server/LLM error after retries, timeout, crash): the cell is rerun under the same
    manifest; if it still fails, it is excluded from **every** arm of the contrast and listed. Infrastructure
    failures > 2% in any arm → INVALID.
- Normalized configs (not only hashes) are stored in the run manifest; G2 diffs them field by field against the
  declared interventions. Resume is allowed only when the manifest hash matches.

### A4. Legacy logs

Logs missing required provenance (the r2 logs: no repetition id, latency, manifest) are accepted in
**DESCRIPTIVE** mode: same statistics and replays, verdict replaced by "DESCRIPTIVE — not a release decision",
exit code 4. Nothing is back-filled or invented.

### A5. Scope of the first release

- Ship: G1–G5, G7, G8; SkillRC and Inspect adapters; the WebShop case as a curated example.
- G6 in the first release: action-category and outcome-type shifts only. Item-level attribution is cut; the case's
  item story is written by hand from block D.
- Inspect: one **real** baseline/candidate agent comparison through Inspect on local Qwen3-8B (vLLM), with tool
  calls, ≥ 2 epochs, candidate = baseline + a declared added text block, ~50–100 samples, no Docker (candidate
  task to verify during implementation: an `inspect_evals` tool-use benchmark that runs without Docker, e.g.
  AgentDojo or BFCL). Plus fixtures with errored samples and missing scores. The adapter keeps sample id, epoch,
  every named score, events (tool calls) and usage.

## B. Frozen parameters for the WebShop runs (pre-registration draft)

| item | value |
|---|---|
| Reader | Qwen3-8B, pinned HF revision (to record), vLLM pinned version (to record), thinking off, temperature 0, max 256 completion tokens |
| Harness | SkillRC ReAct, WebShop 1k server `--seed 0`, PYTHONHASHSEED=0, page label ON, 15 steps, whole-pool injection, BM25 order by PUB |
| Panel T | test sessions 0–499 |
| Panel D | sessions 520–1019. Usage manifest: 0–499 test; 500–519 excluded (configs/webshop_qwen_train20.yaml); ≥ 1500 collection; vocabulary-overlap scripts read all instruction texts (disclosed). Described as "held out from outcome-based development", same product catalog |
| Suspect criterion | "explicitly directs another search, a revised search, or pagination after inadequate results, and contains no search-count, step-budget or stopping rule". Excluded: initial-query construction, filters, inspecting current results |
| Suspect set | pool lines 4, 11, 16, 21, 27, 30 (beauty #4, electronics #5, fashion #4, garden #3, grocery #3, grocery #6); sha256 of each text to be written into the pre-registration |
| Complement draws | from the 24 non-suspects, 3 subsets with category profile beauty 1 / electronics 1 / fashion 1 / garden 1 / grocery 2 and total body words within ±20% of the suspects' total; rejection sampling with seeds 20261116+i, i = 1..3, distinct subsets; overlap between subsets reported; each result reported plus their mean; position/style differences acknowledged |
| Replacement | each replaced item → choral sentence matched to that item's word count and token count under o200k and the Qwen3 tokenizer (r2 choral bank and builder); retrieval ranked by the original PUB text; injected bytes, positions and token counts audited for every modified payload |
| Arms | T: NONE r1/r2, TRUE r1/r2. D: NONE r1/r2, TRUE r1/r2, ABL r1/r2, COMP1, COMP2, COMP3. 13 arm-runs |
| Schedule | two blocks (time windows), each split across two GPU chains; every repeated arm has r1 in block 1 and r2 in block 2 on different GPUs; COMP1, COMP2 in block 1, COMP3 in block 2; order within each chain drawn with seed 20261116 and written into the pre-registration before launch |
| Primary metric | mean WebShop score per episode |
| Release-required contrasts | C1 "ship PUB memory": T_TRUE vs T_NONE. C2 "ship the repaired memory": D_ABL vs D_NONE |
| Confirmatory-diagnostic | L1 D_ABL − D_TRUE > 0 (lower bound); L2 D_ABL − each COMP_i and − mean COMP > 0 (lower bound); replication of r2: T_TRUE − T_NONE upper bound < 0 |
| Descriptive | historical r2 bundle decomposition and choral contrasts; per-block Δ; A/A NONE r1 vs r2; search counts and outcome types |
| δ | 0.05, declared as an illustrative policy (≈ 9% relative to 0.577); sensitivity reported at 0, 0.02, 0.05 |
| Guardrails | prompt tokens per episode ≤ +30% relative; p95 episode latency ≤ +30% relative; infrastructure-failure rate ≤ +1 point (illustrative tolerances, declared) |
| Power note | one repetition: paired SD 0.509, SE 0.023, MDE ≈ 0.064 (80%, two-sided α 0.05, true effect = MDE); two repetitions reduce SE only to the extent that within-task rollout noise is a large share of the paired variance — to be reported, not assumed |

## C. Order of work

1. Offline gate on existing logs (DESCRIPTIVE mode on r2), simulation tests, decision-table tests — CPU only.
2. Runner fixes (append/resume, latency, error accounting, manifest) + mock tests.
3. Build and audit the D payloads (ABL, COMP1–3) as a compute-node job; write the pre-registration with all frozen
   values, hashes and the schedule; Codex check of the artifacts; launch the 13 arm-runs.
4. Inspect adapter + the real Inspect comparison.
5. Case page, README, CI example; Docker and video last.

## D. Questions for round 3

1. Do A1–A5 resolve round-2 items 2, 3 and 5? Any remaining contradiction in A1 (give a counterexample if so)?
2. Is the complement rule change (total removed words instead of per-item) sound?
3. Is table B complete enough that, once hashes, payloads and the schedule are filled in, it can be approved as a
   pre-registration? List exactly what is still missing.
4. Is the order of work right?
