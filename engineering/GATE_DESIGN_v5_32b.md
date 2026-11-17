# Design v5 addendum: Qwen3-32B extension (2026-11-16)

Adds a second reader to GATE_DESIGN_v3 + v4. Review: REVIEW_r6_codex_32b.md (verdict REVISE; all points
adopted below). Internal; not committed.

## Why

Answers the obvious objection "is the regression just a weak 8B model?" (within-family scaling). Qwen3-8B/14B/32B
share a byte-identical tokenizer, so PUB/ABL/COMP payloads and token audits carry over; rendered prompts and
context limits are still re-verified. A different model family would add breadth but needs new payload audits;
it is not in this addendum.

Risk: a bigger reader is not guaranteed to be interpretable on this harness. In r2 arm P, Qwen3-14B scored 0.052
(no label) / 0.168 (label) vs Qwen3-8B 0.326 / 0.577; gpt-4.1-mini scored 0.029. Hence a premise pilot.

## Premise pilot (operational screen, not used for any claim)

- Reader Qwen3-32B (TP = 2), same harness, page label ON.
- 100 sessions from 1020–1499 (unused range), stratified by category, frozen list in the pre-registration; this
  keeps the formal T panel 0–499 untouched.
- Arms: L_NONE, L_PUB_TRUE, one repetition each.
- Proceed to formal 32B arms only if L_NONE mean score ≥ 0.30 and infrastructure-failed cells ≤ 2%.
  Both are declared usefulness criteria, not statistical certification (at n = 100 the score SE can approach 0.05).
- No outcome-driven extension of the pilot, no prompt or harness tuning after seeing it. A failed pilot is reported
  as "32B is near the floor on this harness" and the extension stops.

## Formal 32B arms (if the pilot passes)

- T panel 0–499: NONE ×2, TRUE ×2. Separate release target per reader (C1-32B); no pooling across readers.
- Conditional D panel (520–1019): the same 9 arm-runs as for 8B, launched only if C1-32B's **primary-score**
  component is BLOCK (not a BLOCK caused only by tokens/latency). The trigger, arms, suspects and analysis are frozen
  in the pre-registration; the conditional launch and any branch not taken are disclosed.
- Schedule: TP = 2 uses both GPUs, so 32B arms run serially after the 8B runs, in a newly frozen interleaved order
  (r1 of every repeated arm before any r2); the 8B two-chain schedule does not carry over.

## Cross-reader comparison

- Different verdicts are not a finding. Whether the memory effect differs between readers is estimated directly:
  Δ32B − Δ8B on the T panel, paired by task (task-level deltas of both readers, same stratified task bootstrap),
  with its interval (Gelman & Stern: the difference between significant and not significant is not itself
  significant).
- INCONCLUSIVE means unknown; PASS means non-inferior within δ, possibly with a detectable tolerated regression.
- Stated limitation: readers ran in different windows and need different serving configurations (TP = 2), so the
  comparison is conditional on those, not a clean causal effect of model size.
