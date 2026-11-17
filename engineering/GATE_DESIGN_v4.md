# `skillrc-gate` design v4 (2026-11-16, after Codex round 4)

Delta on GATE_DESIGN_v3.md (v3 stays the contract except where this file overrides it). Reviews:
REVIEW_r{1..4}_codex.md. Internal; not committed.

## 1. Interval and test convention (overrides v3 §2 "Interval" and "One-sided p")

- Sort the B = 20,000 bootstrap estimates ascending: Δ*_(1) ≤ … ≤ Δ*_(B) (1-indexed).
- Reported 95% interval: [L, U] = [Δ*_(500), Δ*_(19501)] (order statistics, no interpolation).
- Decisions use strict comparisons: "Δ > t" is supported iff L > t; "Δ < t" is supported iff U < t.
- Reported one-sided p: p(Δ > t) = (1 + #{b : Δ*_b ≤ t}) / (B + 1); p(Δ < t) = (1 + #{b : Δ*_b ≥ t}) / (B + 1).
  With B = 20,000, L > t ⇔ #{Δ*_b ≤ t} ≤ 499 ⇔ p(Δ > t) < 0.025, ties included; likewise for U. The interval
  decision and the p-value decision are identical by construction (unit-tested, including ties and the round-4
  example: 500 zeros + 19,500 ones, t = 0.5 → L = 0, not supported, p = 0.02505).
- Holm for guardrail-violation claims uses these p-values at family level 0.025, with ties in p broken by
  declaration order.

## 2. Degenerate bootstrap distributions (overrides v3 §2 "Degenerate cases")

- If all paired task differences of a metric are identical, or the bootstrap distribution has fewer than 20 distinct
  values, that component is INCONCLUSIVE (it can neither PASS nor BLOCK), and the report says why.
- Rate metrics with zero events in both arms are reported as 0 without an interval and are never guardrails.

## 3. Error language (overrides v3 §1 "Error-rate scope")

- The report states a **nominal error budget**: one-sided 0.025 for the primary test and 0.025 for the guardrail
  family per target. It is not a finite-sample guarantee: percentile bootstrap error rates can exceed nominal
  levels, notably for rare-event mixtures and small strata. Calibration evidence comes from the simulation suite
  (§5), and its results are shown in the README.

## 4. Missing cells and guardrails (overrides v3 §4 "Missing cells", guardrail part)

- Primary metric: the bounding rule of v3 §4 stands (pessimistic completion for PASS, optimistic completion for
  BLOCK, same bootstrap draws).
- Guardrails: if any cell of either arm is missing, every guardrail component of that target is INCONCLUSIVE (no
  PASS, no BLOCK); the guardrails on completed cells are still reported, labelled "completed attempts only".
  A primary-metric BLOCK that survives optimistic completion still yields BLOCK.
- Consequence, stated in the docs: with any missing cell, a target's best possible verdict is INCONCLUSIVE.

## 5. Localization pairing and half assignment (overrides v3 §2 "Localization claim" and §5 "Schedule")

- Half assignment: within each task type, tasks are shuffled with seed 20261116 and split alternately into
  H_a / H_b (frozen list in the pre-registration).
- D_ABL − COMP_i is computed per task on **same-block** pairs: COMP_i's single observation of a task is paired with
  D_ABL's repetition from the same block (r1 for block-1 tasks, r2 for block-2 tasks); the task differences are then
  bootstrapped as in v3 §2. This is a documented exception to "average all repetitions". D_ABL − D_TRUE keeps the
  averaged rule (both arms have both blocks).
- The mean-over-COMPs contrast uses, per task, the mean of the three same-block differences.

## 6. Tests added to the suite

- The four round-4 counterexamples: (a) missing candidate cost 40,000 among 1,000 cells of cost 100 → guardrails
  INCONCLUSIVE; (b) rare-event mixture −0.051 (p = 0.995) / +0.949 → degenerate sample INCONCLUSIVE, and a Monte
  Carlo estimate of the false-BLOCK rate reported, not asserted; (c) task × block interaction with no intervention
  effect → same-block pairing gives 0; (d) interval/p consistency at ties.
- Exactly one `default: true` target is required when `--target` is omitted; otherwise INVALID.
- Simulation suite: normal-like and rare-event mixtures, stratified panels with small strata, n ∈ {50, 200, 500};
  outputs coverage and verdict error rates per scenario into a committed table.

## 7. Question for round 5

With v3 + v4, can the verdict engine be implemented as specified? Any remaining contradiction?
