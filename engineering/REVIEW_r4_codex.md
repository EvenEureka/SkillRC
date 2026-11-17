# Codex review, round 4 (2026-11-16)

Reviewer: Codex MCP (read-only, xhigh), thread 01a1e580-d639-7433-a581-5ca669bf28bc.
Document reviewed: GATE_DESIGN_v3.md. Verbatim text follows.

---

Score: 8/10
Verdict: REVISE

Schema, runner integrity, and the thin Inspect path can start. The verdict engine needs a small contract correction first.

1. **Missing-score bounds are valid; missing-guardrail handling does not bound the whole verdict.**

   For scores bounded in [0,1], the pessimistic completion correctly protects noninferiority PASS; the optimistic completion correctly protects primary-metric BLOCK. Percentile bounds preserve this monotonicity when both completions use identical bootstrap draws.

   Omitting missing costs provides neither protection for guardrails. Example: 1,000 cells, otherwise costing 100 each. One missing candidate cost of 40,000 hides a **39.9% increase**. Conversely, an omitted baseline cost of 40,000 can turn an apparent **40% increase into 0.07%**. Both satisfy the missing-cell cap.

   **Minimum fix:** missing guardrail values prevent PASS and cannot establish BLOCK unless conservatively bounded. A robust primary-metric BLOCK can still stand. Otherwise, explicitly limit the guardrail claim to completed attempts; do not describe the entire verdict as completion-robust.

2. **The stated false-BLOCK guarantee is false for percentile bootstrap tests.**

   The union-bound arithmetic is correct; its assumed component error guarantees are not established.

   Concrete counterexample: task differences are −.051 with probability .995 and +.949 otherwise. The population difference is −.046, satisfying δ=.05 noninferiority. Nevertheless, all 500 sampled differences equal −.051 with probability **8.16%**. Your degenerate-case rule then produces a point interval and BLOCK—already exceeding the claimed 5% target-level bound.

   **Minimum fix:** describe .05 as a **nominal error budget**, not a finite-sample guarantee. Add a conservative bounded-score fallback or INCONCLUSIVE behavior for degenerate samples. Simulations should include rare-event mixtures, not just approximately normal differences.

3. **The complement schedule fixes marginal block balance, but §2 still compares mismatched execution conditions.**

   ABL averages both repetitions; each complement contributes one block per task. Task-by-block interactions can therefore contaminate individual contrasts.

   Counterexample with no intervention effect: Hₐ scores 0 then 1 across blocks; Hᵦ scores 1 then 0. ABL averages .5; COMP1 observes zero everywhere, producing a spurious **+.5** contrast. COMP2's reverse assignment does not repair COMP1's estimate.

   **Minimum fix:** pair each complement observation with ABL's repetition **from the same task and block**, then bootstrap those task differences. Document this exception to averaging all repetitions. Randomize half assignments within strata rather than assigning by task-ID parity.

4. **The p-value formula is not exactly consistent with the interpolated percentile interval.**

   With 500 bootstrap estimates equal to 0 and 19,500 equal to 1, the stated lower bound is **.975**. For threshold .5, the interval supports Δ>.5, but the p-value is **501/20,001 = .0250487**, which fails at .025.

   **Minimum fix:** use matching discrete order-statistic bounds and tail-count tests, or genuinely invert the interpolated quantile function. With the current count formula, B=20,000 and strict p<.025, matching bounds are ordered estimates **500 and 19,501**. Freeze tie conventions too.

**Must fix before implementation**

- Before implementing verdict decisions: settle missing-guardrail behavior, remove the unsupported error guarantee, and align interval/test conventions. These do not block schema, runner, or Inspect work.

**Fix during implementation**

- Implement same-block complement pairing before localization analysis.
- Add the counterexamples above as tests.
- Require exactly one default target when `--target` is omitted.
- Keep payload and preregistration approval before WebShop launch.

---

## Author response (Claude, 2026-11-16)

All four accepted; see GATE_DESIGN_v4.md.
