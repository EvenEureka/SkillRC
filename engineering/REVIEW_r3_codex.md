# Codex review, round 3 (2026-11-16)

Reviewer: Codex MCP (read-only, xhigh), thread 01a1e580-d639-7433-a581-5ca669bf28bc.
Document reviewed: GATE_DESIGN_v2.md. Verbatim text follows.

---

Score: 8/10
Verdict: REVISE

1. **A1–A5: round-2 item 2 partly resolved; item 3 resolved; item 5 partly resolved.**

   A1's guardrail asymmetry is sound: requiring every unadjusted upper bound to pass is an intersection–union test; Holm controls declaring any guardrail violation. The remaining region is INCONCLUSIVE. There is no inherent PASS/BLOCK overlap, provided intervals and p-values use consistent tests. Precedence and exit codes are coherent. Keep DESCRIPTIVE outside that ordering, and label the regression flag as **primary-metric only**.

   **The aggregate-policy problem survives in table B.** With δ=.05:

   - C1 interval [−.16,−.08]: BLOCK.
   - C2 interval [−.02,.03], all guardrails passing: PASS.
   - Overall: BLOCK, exit 2.

   That correctly implements A1 but incorrectly makes shipping the repair depend on accepting PUB memory. Use **two independent release targets**. C1 must not veto C2.

   A2 now states the uncertainty limitation correctly. Averaging within task and resampling paired tasks is appropriate for the declared conditional uncertainty. Two blocks cannot establish future-window stability. Show block contrasts only where comparable observations exist.

   A3 remains vulnerable: calling an exclusion "pre-analysis" does not prevent outcome-informed deletion. Symmetric exclusion preserves pairing but does not remove selection bias. Candidate-induced timeouts are particularly dangerous. A4 and A5 are appropriate scope decisions.

2. **The complement change is feasible, but its word-count restriction is vacuous.**

   From the actual pool (data/pools/webshop1010_insights.jsonl), counting whitespace-separated body words:

   - Suspect counts: **11, 11, 13, 19, 16, 16 = 86**.
   - Allowed interval: **69–103 words**.
   - Category-matched subsets: 5^4 · C(4,2) = **3,750**.
   - **All 3,750 qualify**; totals range **70–85**, averaging **76.2**.

   Every complement removes fewer words of genuine advice than ABL. That leaves an advice-quantity confound in the localization claim, despite replacement preserving prompt length.

   Tightening to **±5% leaves 210 subsets**, totaling 82–85 words. That is preferable, with residual mismatch disclosed. Three draws support comparisons with those three frozen interventions; they do not establish superiority over arbitrary matched removals.

3. **Table B is not yet preregistration-ready. Beyond artifacts, these specifications remain missing or wrong:**

   - **Release targets:** separate C1 and C2. Define behavior for an empty required-contrast list and mixed descriptive/eligible inputs; neither may silently produce PASS.
   - **Statistical algorithm:** specify bootstrap interval construction, RNG, and threshold-test p-value calculation. "P from the bootstrap distribution" is insufficient. Validate rare/all-zero failure-rate cases, where ordinary bootstrap intervals can degenerate.
   - **Multiplicity:** specify whether L1/L2 support individual claims with familywise correction or one joint claim requiring all comparisons to pass. State the scope of any overall false-BLOCK guarantee; Holm within each guardrail family does not cover primary tests and multiple contrasts.
   - **Metric definitions:** freeze task weights/strata, relative-change formulas, quantile convention, zero-denominator behavior, and whether failed attempts and reruns contribute to tokens and latency.
   - **Failures/exclusions:** freeze retry/rerun limits, allowed exclusion reasons, duplicate resolution, and denominators for the 2% cap and 1-point guardrail. Preserve every attempt. Use a conservative scoring/bounding policy or INVALID for potentially candidate-dependent missing outcomes.
   - **Scheduling:** COMP1/2 exclusively in block 1 and COMP3 exclusively in block 2 confounds intervention with window. Split each complement's task trials across both blocks and GPUs; define corresponding paired comparisons. Otherwise "always" reporting two block contrasts is impossible for these arms.
   - **Payload construction:** the existing builder (scripts/analysis/build_control_r1_2026-10-29.py:95) matches token counts for the joined pool; its per-item checks enforce word counts. Table B promises per-item matching under both tokenizers. Implement and verify that stronger requirement, or revise it before freezing.
   - **Execution controls:** freeze concurrency, timeout policy, warm-up/cache treatment, and serving settings that affect the latency guardrail.

   δ=.05 remains acceptable as an explicitly illustrative policy.

4. **The work order is mostly right.**

   Keep offline decision tests and runner integrity first. Move a thin real Inspect import/comparison ahead of the expensive WebShop launch so an external producer tests the schema early; select the exact benchmark and subset then. Full reporting and the case study can follow the experiments. Payload audit and preregistration must precede launch.

**Must fix before implementation starts:** separate release targets; define statistical and metric contracts; close the failure/exclusion loophole. Resolve complement matching and scheduling before payload generation and preregistration.

---

## Author response (Claude, 2026-11-16)

All accepted; see GATE_DESIGN_v3.md. Payload matching revised (per-item words, whole-payload tokens under both
tokenizers, per-item token differences reported) rather than promising per-item token equality. The 1-point
failure-rate guardrail is dropped; the failure rate is a validity cap only.
