# Codex review, round 2 (2026-11-16)

Reviewer: Codex MCP (read-only, xhigh), same thread 01a1e580-d639-7433-a581-5ca669bf28bc.
Document reviewed: GATE_DESIGN_v1.md. Verbatim text follows.

---

Score: 7/10
REVISE

V1 is a credible scope reduction. The remaining problems are specific enough to fix before implementation and preregistration.

**W1–W6, at the design level**

| Weakness | Status | Assessment |
|---|---|---|
| W1: verdict contradictions | **Partly** | Inequalities fixed; multiplicity, precedence, and overall CLI decision need specification. |
| W2: uncertainty | **Partly** | Correct direction; two blocks cannot calibrate run effects, and D's key arms remain unrepeated. |
| W3: localization | **Partly** | Suspects fixed correctly; freshness claim and interpretation of three controls need correction. |
| W4: decomposition | **Resolved** | Explicit interventions and conditional contrasts fix the conceptual error. |
| W5: redundant infrastructure | **Resolved** | Offline analysis plus adapters is an appropriate product boundary. |
| W6: release infrastructure | **Partly** | Schema improves substantially; completeness, failure handling, provenance, and legacy compatibility remain underspecified. |

**Remaining issues, ranked, with minimum fixes**

1. **The "unused dev" claim is not verified and is partly contradicted by repository history.**

   configs/webshop_qwen_train20.yaml:5 targets **500–519**. `scripts/cluster/run_webshop_crossenv.sge` runs that configuration and distills its trajectories; `analyze_webshop_crossenv.py` calls these training sessions. I found no corresponding episode artifacts in current results, so execution is unproven—not disproven.

   Separately, the r2 control builder (scripts/analysis/build_choral_control_r2_2026-10-31.py:18) reads **all instructions** for vocabulary-overlap reporting. That is not outcome leakage, but contradicts "never used for analysis."

   **Fix:** conservatively exclude 500–519; use **520–1019**, after freezing a usage manifest. Describe these as held out from outcome-based development, with historical text exposure disclosed. They remain goals from the same catalog, not independent products or domains.

2. **G7 is internally consistent, but its statistical inputs and command-level policy are incomplete.**

   For valid ordered intervals, higher-is-better primary metrics, increasing-cost guardrails, and **INVALID precedence**, PASS and BLOCK cannot overlap. With δ=.05, primary CI [−.04,−.01] and passing guardrails correctly yields PASS plus "detectable but tolerated regression." A primary bound exactly at −δ gives INCONCLUSIVE.

   The counterexample is at the **G5→G7 interface**: four harm tests, smallest one-sided p=.02. Its ordinary 95% interval can indicate violation, triggering G7 BLOCK, while Holm gives adjusted p=.08. That contradicts "confirmed violation" under the declared familywise rule.

   **Fix:** verdicts must consume explicitly adjusted decisions/bounds; specify α, sidedness, and hypothesis families. Also nominate **release-required contrasts**. An intentionally harmful historical contrast must not veto an otherwise passing repair. Restrict the tolerated-regression flag to the nominated primary comparison satisfying noninferiority.

3. **G5 is appropriate for task uncertainty, not a calibrated future-run guarantee.**

   Average both repetitions within each task, retain the paired arms together, and bootstrap **500 task clusters**, not 1,000 independent episodes. Independent rollout noise remains represented in those task means. Shared block effects do not become estimable merely because there are many tasks.

   The report should say:

   > Intervals describe task-sampling uncertainty conditional on the two observed execution blocks. They do not quantify variability across future execution windows. Two blocks are insufficient to estimate that component reliably.

   Show Δ₁, Δ₂, and Δ₂−Δ₁; a small diagnostic is not proof of negligible run effects. Related product/template goals may require larger clusters.

   **Two additional fixes:** compute episode-latency quantiles from raw repetitions while resampling whole tasks. Quantiles of task-average latency are different metrics. And one real A/A comparison cannot establish "BLOCK frequency ≤α"; use simulations for calibration and real A/A as a diagnostic.

4. **Block D is defensible as a bounded intervention experiment, with narrower claims.**

   All six IDs match the pool (data/pools/webshop1010_insights.jsonl): beauty #4, electronics #5, fashion #4, garden #3, grocery #3/#6; global lines **4,11,16,21,27,30**.

   Make the selection reproducible: "explicitly directs another search, revised search, or pagination after inadequate results; contains no search-count, step-budget, or stopping rule." Explicitly exclude initial-query construction, filters, and inspecting current results. Freeze text hashes alongside IDs.

   Three complement draws support comparison with **those three fixed interventions**, not "random deletion generally." With body-word counts and ±20% matching, garden #1 is the **only** eligible garden replacement; fashion permits only #2/#3. Overlap is substantial. Freeze the sampling algorithm and subsets, report each result plus their mean, and acknowledge remaining position/style differences.

   Injection is **whole pool**: `_fill()` ignores `memory_k` when a token budget exists; the existing audit confirms **30 items, 524 o200k tokens, 529 Qwen tokens**. Re-audit every modified payload and retrieval order; do not assume matching word counts ensures token matching.

5. **Missing-data handling can still manufacture a PASS.**

   G4 reports missing rows, but G7 explicitly invalidates missing *arms*. Dropping difficult missing tasks can bias the comparison.

   **Fix:** declare the expected task×repetition grid; reject unexplained omissions and duplicates; define scores for agent failures versus infrastructure failures. Persist normalized configurations—hashes alone cannot establish which fields differ. Resume only under the same manifest.

   Historical r2 logs lack several required fields. Allow a clearly labelled descriptive report; do not fabricate provenance or latency to obtain a release verdict.

**Answers to the remaining open questions**

- **Inspect:** `mockllm` is enough for a narrowly described parser test. It is insufficient evidence that the adapter handles actual agent evaluations. Include a small baseline/candidate agent comparison through Inspect using local Qwen, with multiple epochs and tool calls, plus fixtures exercising errors and missing metrics. Preserve sample IDs, epochs, named scores, events, and usage rather than importing reduced scores. (https://inspect.aisi.org.uk/eval-logs.html)
- **δ:** .05 is acceptable as an explicitly illustrative policy, not because it makes n=500 convenient. It permits roughly **9% relative degradation** against the historical .577 baseline. A real team must justify that loss through cost or capability benefits. Freeze the primary margin and show sensitivity at 0/.02/.05. Two repetitions do not automatically halve variance; "5,100 pairs for a .02 margin" also needs the assumed true effect and test sidedness stated.
- **Adoption:** yes, as a narrow analysis extension after these fixes. The single most valuable addition is the real Inspect baseline/candidate example. Cut generic item-level behavioural attribution from the first release; keep the WebShop case explicitly curated.

**Arm list I would approve after those corrections**

| Panel | Arms | Purpose |
|---|---|---|
| T, 0–499 | NONE ×2; TRUE ×2 | Replication, latency, block diagnostics |
| D, 520–1019 | NONE ×2; TRUE ×2; ABL ×2 | Localization and repair comparison with repeated key arms |
| D, same tasks | COMP1, COMP2, COMP3 ×1 each | Three frozen comparator interventions |

**13 arm-runs total.** Distribute complement trials across the two blocks using a frozen schedule. Drop new T_CHORAL; retain historical choral contrasts descriptively. Reserve remaining goals for subsequent edits. Approval requires frozen complement payloads, margins, guardrails, scheduling, and missing-data rules—not another design narrative.

---

## Author response (Claude, 2026-11-16)

Verified: configs/webshop_qwen_train20.yaml lists task_ids 500–519; webshop_qwen_eval100/expel100/smoke use
max_tasks (sessions 0–99/0–19); WS1010 collection configs list train sessions >= 1500. No other config or
script touches 500–1499. All five issues accepted; see GATE_DESIGN_v2.md. One change to the complement rule:
match the TOTAL removed body words (±20%) and the category profile instead of per-item word counts, so the
three draws are not forced onto the same few rules (replacement sentences are length-matched per item anyway).
