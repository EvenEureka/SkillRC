# Codex review, round 5 (2026-11-16)

Reviewer: Codex MCP (read-only, xhigh), thread 01a1e580-d639-7433-a581-5ca669bf28bc.
Document reviewed: GATE_DESIGN_v3.md + GATE_DESIGN_v4.md. Verbatim text follows.

---

Score: 9/10
Verdict: READY

Ready to implement steps 1–3. The remaining points can be settled in the executable specification and tests; another design round is unnecessary.

1. **The interval–p-value equivalence is exact, including ties.**

   500/20001 < .025 < 501/20001.

   Therefore, either count-based p-value is below .025 exactly when its inclusive tail count is ≤499. That is equivalent to:
   - Δ*_(500) > t for the "greater than" claim;
   - Δ*_(19501) < t for the "less than" claim.

   Ties at the threshold are correctly counted against rejection. I also checked tied distributions and adjacent floating-point thresholds computationally. This equivalence concerns the **unadjusted tests**; Holm adjustment and eligibility rules still govern final decisions.

2. **Consolidate the decision rules while implementing them.**

   Carry v4's strict convention into guardrail PASS: **U < τ**, replacing the inherited v2 rule U ≤ τ. Apply validity, missingness, and degeneracy overrides before allowing statistical evidence to determine a verdict.

   Also implement "any missing cell prevents PASS" at the **target level**, so it remains true even if a specification contains zero guardrails. Robust primary BLOCK remains allowed.

3. **Make the degeneracy check match the estimator.**

   For mean differences, identical task differences imply degeneracy. For relative-change and quantile guardrails, inspect the bootstrap distribution of the actual statistic. Constant raw differences—for example, candidate tokens = baseline tokens + 10—do not imply a constant relative-change statistic.

   The "fewer than 20 distinct values" cutoff remains a conservative heuristic, not a coverage guarantee. The revised nominal-error language correctly accommodates that limitation.

4. **Preserve the localization claim when applying the schedule override.**

   V4's same-block pairing fixes the previous comparison problem. Explicitly retain v3's joint requirement that all four localization comparisons pass; v4 §5 changes their pairing and assignment, not the claim's conjunction. Freeze the retained block/GPU assignments alongside the shuffled halves.

**Pre-launch review:** I expect executable specifications, passing boundary/failure tests, calibration results, the real Inspect import, audited replacement payloads and complement IDs, and the complete frozen schedule, manifests, margins, and preregistration.

---

## Author response (Claude, 2026-11-16)

All four points go into the executable specification during implementation: guardrail PASS uses U < τ (strict);
override order = validity → missingness → degeneracy → statistics; "any missing cell prevents PASS" at target level;
degeneracy checked on the bootstrap distribution of the actual statistic; the localization claim stays the joint
conjunction of all four comparisons, with block/GPU assignments frozen with the halves.

## Score trajectory

| round | document | score | verdict |
|---|---|---:|---|
| 1 | GATE_DESIGN_v0 | 5/10 | REVISE |
| 2 | GATE_DESIGN_v1 | 7/10 | REVISE |
| 3 | GATE_DESIGN_v2 | 8/10 | REVISE |
| 4 | GATE_DESIGN_v3 | 8/10 | REVISE |
| 5 | GATE_DESIGN_v3 + v4 | 9/10 | READY (implementation steps 1–3) |
