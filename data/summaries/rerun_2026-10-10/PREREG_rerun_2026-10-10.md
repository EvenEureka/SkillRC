# Pre-registration: coherent task-irrelevant control, and a same-window rerun of the placebo gate

Written 2026-10-10 before any new episode was run. Author: Yiwen Lu. Reason: the external review of
the SkillRC manuscript (2026-10-09) and the storyline gap analysis both note that the order placebo
confounds "no task content" with "incoherent text", so the coherent, equal-length, task-irrelevant
control is required before the content increment can be interpreted. The author lifted the
no-rerun rule for this project on 2026-10-10.

## Design

Tasks: the frozen 60-task gate (10 per task type, `results/placebo/frozen_tasks.json`, seed
20260713), identical task ids as in `configs/papero_placebo_*.yaml`.
Demonstration pairs: seeds `[0, 1, 5]` (the three distinct pairs 02, 12, 01 of the paper; seed 2
duplicated seed 1's pair in the seed-axis audit and is not rerun).
Executors: hosted `gpt-4o-mini` (alias as in the paper; the model id returned by the API is recorded
per episode) and locally served `Qwen/Qwen3-8B` (vLLM, thinking off, temperature 0).
Scaffold, prompts, decoding, step limit (30), admissible-action injection, BM25 presentation,
`memory_k=20` and `token_budget=645` are exactly those of the original gate.

Conditions, all rerun in the same time window (the per-episode logs of the 2026-07 gate are lost and
the hosted alias is mutable, so no new condition is compared with an old run):
- `none`: no memory.
- `true`: the 33 ExpeL insights (`data/pools/expel_insights.jsonl`).
- `placebo`: the order placebo (`expel_insights_order_placebo.jsonl`, rebuilt deterministically).
- `irrelevant`: the coherent task-irrelevant control (`expel_insights_irrelevant_coherent.jsonl`):
  33 well-formed imperative sentences about vegetable gardening, each with the same task tag and the
  same word count as the corresponding insight, no ALFWorld vocabulary, footprint matched to the true
  memory with the placebo's minimal nonce procedure under both tokenizers, injected at the same
  indices through the index-aligned reference ranking (`rank_reference_pool`), so item identity,
  order and footprint are identical across `true`, `placebo` and `irrelevant`.

Episodes: 60 tasks x 3 pairs x 4 conditions x 2 executors = 1,440.

## Estimands (per executor E, equal-pair task means, 60 tasks)

- `T_E` = true - none (the memory effect), `P_E` = placebo - none (context effect, incoherent),
  `I_E` = irrelevant - none (context effect, coherent but irrelevant),
  `S_E` = true - placebo (content increment, as in the paper), `R_E` = true - irrelevant
  (relevance increment), `C_E` = irrelevant - placebo (coherence increment).
- Transport gaps for each: `Delta_X = X_Q - X_G`.
- Intervals: stratified task bootstrap (10 per type), 20,000 draws, seed 20260713, percentile 95%,
  as in `scripts/analysis/analyze_semantic_placebo.py`.
- Rerun drift (descriptive, not a contrast): new `none` and `true` means vs the 2026-07 gate values
  (GPT 0.544 / 0.681; Qwen 0.728 / 0.683) and vs the 134-task values.

## Decision rules (frozen)

1. If `|C_E| <= 0.05` for both executors (irrelevant behaves like the placebo) and `R_G` excludes
   zero positively, the content increment is attributed to task relevance, not to coherence.
2. If `|R_E| <= 0.05` for both executors (irrelevant behaves like the true memory), the memory's
   effect is that of any coherent instruction block; task relevance is not needed to produce it.
3. If `C_E` excludes zero (coherence alone moves the executor) and `R_G` also excludes zero, both
   coherence and relevance contribute; report both increments.
4. Otherwise: inconclusive under this gate; report the intervals and the minimum detectable
   difference implied by the bootstrap variance.
The executor reversal of the paper (`Delta_T < 0`) is re-tested in the new window; if it does not
reproduce, the 2026-07 gate results are reported as not replicated at this date.

## Predictions

- P1: `T_G > 0` with CI excluding zero (the memory still helps GPT-4o-mini); `T_Q` includes zero.
- P2: `P_G > 0` and `P_Q < 0` reproduce in direction.
- P3 (our expectation): `I_E` lies between `P_E` and `T_E` for GPT, i.e. part of the context effect
  is coherence; `R_G` is positive but smaller than `S_G`.
- P4: `I_Q` is less negative than `P_Q` (a coherent block disrupts Qwen3-8B less than a scrambled one).

## What is not changed

No other pool, task, prompt or decoding setting. No 134-task run. No WebShop run. The analysis
script is `scripts/analysis/analyze_rerun_2026-10-10.py`, written before the runs finish and run
once on the complete data.

## Addendum A (2026-10-10 11:50, after the runs, before the analysis was run)

The 60 task ids are those of the frozen gate, but the ALFWorld game behind each id is not the same
game as in the 2026-07 gate: the adapter enumerates the re-downloaded `valid_unseen` games in
directory order, which differs from the order of the deleted 2026-07 data directory (already noted on
2026-10-08, when re-matching task ids to games recovered only 4 of 27 types). Within this rerun the
mapping is identical for all eight runs: every task id has the same `task_text` and `task_type` in
all four conditions and both executors (verified before analysis). The 60 games now comprise 11
`clean`, 9 `put` and 10 of each other type. The stratified bootstrap therefore stratifies by the
observed types (unequal strata), with everything else unchanged. Because the task set differs, the
"rerun drift" against the 2026-07 values is descriptive only and is not a replication test of
those numbers; the executor reversal (`Delta_T < 0`) is re-tested on this new task sample.
