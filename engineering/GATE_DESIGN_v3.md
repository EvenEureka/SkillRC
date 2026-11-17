# `skillrc-gate` design v3 (2026-11-16, after Codex round 3)

Delta on GATE_DESIGN_v2.md. This file is the implementation contract for the statistics, metrics, failures and the
WebShop runs. Reviews: REVIEW_r{1,2,3}_codex.md. Internal; not committed.

## 1. Release targets (replaces v2 A1 "overall verdict")

- A spec declares one or more **release targets**. Each target = one candidate arm vs one baseline arm, a primary
  metric with margin δ, and its guardrails. Targets are independent: each gets its own verdict and exit code
  (`skillrc-gate report spec.yaml --target C2`; without `--target`, the exit code is that of the target marked
  `default: true`, and every target's verdict is printed). No target can veto another.
- A spec with no release target, or a `--target` that does not exist → INVALID (exit 3).
- If any arm used by a target lacks required provenance → that target is DESCRIPTIVE (exit 4). DESCRIPTIVE sits
  outside the INVALID > BLOCK > INCONCLUSIVE > PASS ordering; it is never PASS.
- The "detectable but tolerated regression" flag applies to the primary metric only.
- Error-rate scope, stated in the report: per target, false BLOCK probability ≤ 0.025 (primary, one-sided) +
  0.025 (guardrail family, Holm) ≤ 0.05 by the union bound. No guarantee across targets or across diagnostic
  contrasts.

## 2. Statistical contract

- **Unit and weights.** Task = resampling unit. Per arm, the task value = mean over that task's repetitions. A
  contrast's estimate = unweighted mean over the panel's tasks of (candidate − baseline) task values.
- **Bootstrap.** Stratified by task type: within each stratum, draw as many tasks as the stratum has, with
  replacement; B = 20,000; NumPy `default_rng(seed)` (PCG64), seed declared in the spec (case: 20261116). The same
  resampled task index set is used for every metric of a target within one draw.
- **Interval.** Percentile interval [q_0.025, q_0.975] of the bootstrap estimates (NumPy `quantile`, method
  "linear").
- **One-sided p for a threshold claim**, consistent with the percentile interval: for the claim "Δ > t",
  p = (1 + #{b : Δ*_b ≤ t}) / (B + 1); for "Δ < t", p = (1 + #{b : Δ*_b ≥ t}) / (B + 1).
- **Degenerate cases.** If every paired task difference is identical, the interval is the point and p follows the
  formula above (tested). Rate metrics with zero events in both arms → reported as 0 with no interval; they are not
  used as guardrails (see §3).
- **Relative change** for guardrails: r = mean_cand / mean_base − 1, computed inside each bootstrap draw on the same
  tasks. A zero baseline mean → INVALID for that target.
- **Latency quantiles.** p50/p95 over the raw episodes of the resampled tasks (each drawn task contributes all of its
  repetitions' episodes, with multiplicity); NumPy "linear" quantile. Relative change computed per draw.
- **Diagnostic contrasts** (pre-registered, never feed a verdict) use the same machinery.
- **Localization claim (case).** One joint claim, "the suspect items explain the loss on these goals", holds only if
  all four lower bounds are > 0: D_ABL − D_TRUE, and D_ABL − COMP_i for i = 1, 2, 3. Intersection–union, no
  adjustment. Individual comparisons and the mean over COMPs are reported. The claim covers these three frozen
  comparator interventions, not arbitrary matched removals.

## 3. Metrics and guardrails

- Primary: mean WebShop score of the scored attempt (definition of "scored attempt" in §4).
- Guardrails (case values, illustrative, declared): prompt tokens per episode, relative change ≤ +30%; p95 episode
  latency, relative change ≤ +30%. Tokens and latency come from the scored attempt and include that attempt's LLM
  retries. Tokens and time of failed attempts are reported separately and are not in the guardrails.
- The infrastructure-failure rate is not a statistical guardrail; it is a validity cap (§4).

## 4. Failures, missing cells, duplicates

- **Attempts.** Every attempt of every cell (arm × task_uid × repetition) is logged with an attempt id. LLM calls
  retry up to 5 times with backoff; WebShop requests time out at 120 s.
- **Agent outcome vs infrastructure failure.** Agent outcomes (invalid action, loop, budget exhausted, wrong item,
  model output unparseable) are scored as recorded. Infrastructure failures = WebShop server error/unreachable, LLM
  HTTP error after retries, runner crash. An attempt that times out on an LLM call after retries is infrastructure.
- **Reruns.** A cell whose attempt failed for infrastructure is rerun up to 2 more times, same manifest, same block.
  The scored attempt = the first attempt without an infrastructure failure.
- **Duplicates.** Two scored attempts for one cell cannot occur under the rule above; if they do → INVALID.
- **Missing cells** (no scored attempt after reruns): no exclusions are allowed. If missing cells exceed 0.5% of an
  arm's expected cells → INVALID. Otherwise the verdict is bounded: PASS requires PASS when candidate-missing cells
  score 0 and baseline-missing cells score 1; BLOCK requires BLOCK when candidate-missing cells score 1 and
  baseline-missing cells score 0; otherwise INCONCLUSIVE. Tokens/latency of missing cells are left out of the
  guardrails, with the count reported.
- **Validity cap.** Cells with ≥ 1 infrastructure-failed attempt > 2% of an arm's expected cells → INVALID.

## 5. WebShop runs: changes to v2 table B

- **Complements.** Category profile 1/1/1/1/2 and total removed body words within ±5% of the suspects' 86
  (→ 210 eligible subsets, totals 82–85). Three distinct subsets by rejection sampling, seeds 20261116 + i. ABL removes
  86 words of advice, the complements 82–85; the residual is disclosed.
- **Payload matching.** Each replaced item is matched to the original item's word count. The whole payload is
  matched in token count under o200k and the Qwen3 tokenizer (as the existing builder does). Per-item token
  differences are reported, not required to be zero. Every payload re-audited for bytes, positions, retrieval order
  and token counts.
- **Schedule.** Two time blocks, two GPU chains per block.
  - Repeated arms (T_NONE, T_TRUE, D_NONE, D_TRUE, D_ABL): r1 in block 1, r2 in block 2, on different GPUs.
  - Complements: each COMP_i's 500 tasks split into two stratified halves H_a, H_b (alternate assignment within each
    task type in task-id order). COMP1 runs H_a in block 1 and H_b in block 2; COMP2 the reverse; COMP3 like COMP1.
    The GPU for each half is the one not used by that block's half of the previous complement.
  - Per-block contrasts are reported only where both arms have observations on the same tasks in that block.
  - Order within each chain drawn with seed 20261116; the whole schedule is written into the pre-registration.
- **Execution controls.** One episode at a time per chain; one vLLM server per GPU serving only its chain; vLLM
  settings frozen and recorded (max_model_len, gpu_memory_utilization, prefix caching ON as in default serving,
  dtype); 5 discarded warm-up requests after each server start; latency = client-side wall time per call including
  retries, summed per episode; LLM timeout 120 s.

## 6. Order of work (revised)

1. Offline core: schema, ingest (SkillRC adapter), pairing, statistics contract, verdict table; decision-table and
   simulation tests; DESCRIPTIVE report on r2 logs. CPU only.
2. Runner integrity: attempts log, append/resume by cell key, manifest with normalized config, latency, error
   accounting; mock tests.
3. Thin Inspect path: pick the benchmark (tool-use, no Docker), run a small baseline/candidate comparison on local
   Qwen3-8B with ≥ 2 epochs, import it — before the WebShop launch.
4. WebShop payloads (compute-node job) + audits; pre-registration with all frozen values, hashes and the schedule;
   Codex check of the artifacts; launch the 13 arm-runs.
5. Report HTML, case page, README, CI example; Docker and video last.

## 7. Question for round 4

Is this ready for implementation steps 1–3 to start (with the pre-registration and payloads to be checked again
before launch)? If not, what exactly blocks it?
