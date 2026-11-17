# Codex review, round 1 (2026-11-16)

Reviewer: Codex MCP (read-only sandbox, model_reasoning_effort=xhigh), thread 01a1e580-d639-7433-a581-5ca669bf28bc.
Document reviewed: GATE_DESIGN_v0.md. Verbatim text of the review follows.

---

Score: 5/10
REVISE

The regression is real. The proposed release gate is not yet statistically coherent, and too much of the plan rebuilds existing tooling.

**Evidence check**

- **§1a matches.** All means, search counts, and `R_L_*`/`I_L_*` intervals match the results (results/placebo/r2_2026-10-31_results.json), allowing for rounding. The bundle gain is +0.1322, label gain without memory +0.2514, and memory effect with labels −0.1191. Preregistration P (results/placebo/PREREG_r2_2026-10-31.md) explicitly includes labelled and unlabelled arms in one window. PUB is the published pool; R1/R2 are Qwen3-32B distillations from stored GPT-4o-mini training successes.
- **§1b mislabels a metric.** In `results/WS1010_{NONE,TRUE}_GPT.episodes.jsonl`, checkout completion is **0.366 → 0.204**, explaining "0.37 → 0.20." Actual recorded task success is **0.194 → 0.086**. Nonterminal budget exhaustion is **0.634 → 0.796**; those rounded numbers match.
- **"No reading cost" is unsupported.** PUB's labelled control effect is +0.0206 **[−0.0191, +0.0604]**; R1/R2 intervals permit roughly five-point losses. Failure to reject zero is not equivalence. Likewise, removing all cooling rules (results/placebo/cool_loo_2026-10-12.json) gives +0.1429 **[−0.0159, +0.3016]**: partial recovery is a point estimate, not established recovery.
- The 30 rules and quoted examples match. Search changes **3.33/1.06/0.11** are TRUE minus **irrelevant control**, not no-memory. Their association with harm across three readers is descriptive; GPT-4.1-mini's near-zero baseline creates a floor effect.
- Remaining §1 numbers check out: ALFWorld disagreement 17.41–19.65%; WebShop score disagreement 13.2–17.4%; heterogeneity p=0.00032, τ=0.0189; topic shift −0.0249; `R_web`=0.0108. Inspected injection audits report zero mismatches. Latency is absent. These findings do not establish universal reading cost, pipeline reliability, or production usefulness.

**Ranked weaknesses and minimum fixes**

1. **The verdict rule contradicts its own tolerance.** With δ=0.02 and CI [−0.015, −0.005], both PASS and BLOCK conditions hold. Choose a policy: PASS when the lower bound exceeds −δ; BLOCK when the upper bound falls below −δ; otherwise INCONCLUSIVE. Report "detectable but tolerated regression" separately. For an increasing-cost guardrail, PASS requires its upper bound below tolerance; confirmed violation requires its lower bound above tolerance. Uncertainty is not confirmed violation. Invalid evidence needs a distinct status that prevents release. CI must require explicit PASS; merely failing on BLOCK lets INCONCLUSIVE through.

2. **The uncertainty model confuses task sampling with execution variability.** The premise that task resampling captures no run noise is too strong: with independent tasks and independent rollout noise, paired task differences already contain both. They do **not** identify shared run/window effects, or isolate rerun uncertainty on a fixed task panel. Do not mechanically add BASE_DUP variance.

   Repeat **both** arms in randomized, interleaved blocks. Average repetitions within task, then cluster/resample tasks; estimate differential run effects separately when enough blocks exist. One run cannot identify them; two provide a very unstable estimate. Likewise, three pools provide weak pipeline-level uncertainty, not a dependable generalization guarantee.

   From the requested episode files: **500 unique pairs, seed 0, no errors; Δ=−0.11914; SD(difference)=0.50940; SE≈0.0228; stratified bootstrap CI≈[−0.1636, −0.0750].** Approximate 80%-power MDE is **0.064** at two-sided α=.05, or **0.057** one-sided. Detecting 0.02 needs approximately **5,100 independent pairs** at the same spread. Product/template clustering could worsen this.

   Predeclare contrasts and sample size. Requiring *all* noninferiority tests to pass is an intersection-union test and does not automatically require Bonferroni. Declaring *any* harmful guardrail/factor significant does require familywise control, such as Holm or simultaneous intervals. Keep exploratory factorial findings separate. Replace "n needed for the observed effect" with power curves for prespecified meaningful effects; repeated peeking needs sequential control.

3. **Localization is promising but currently ill-defined.** Literal whole-word matching selects **17/30 rules**, leaving only **13** non-suspects: the proposed equal-size complement cannot be constructed. It also misses "re-searching" while selecting benign query-quality advice.

   In-place substitution tests that particular intervention's effect within the full pool. It does not prove that over-search mediates harm. Redundancy, interactions, ordering, and control semantics remain. Freeze a small, manually reviewed suspect set and its IDs; preserve retrieval ranking using the original pool; audit actual injected bytes and positions. Use multiple matched random subsets, matching category, length, position, and instruction style. Require confidence bounds—not positive point estimates—for both recovery contrasts. Confirm on fresh held-out goals. A suspect-only add-back can test sufficiency; a larger step-budget comparison can probe the proposed mechanism.

4. **The decomposition is wrong for bundled changes.** N7's `CTRL − BASE` includes every other changed factor. With page label plus memory, it includes the label benefit and cannot be called reading cost. Compare CTRL against the **same candidate harness without added text**. Report conditional effects and interactions; factorial main effects are not context-free verdicts for individual factors. Config fields also need not be independently executable factors. Use explicit, validated interventions and content hashes, not automatic field counting.

5. **Most infrastructure is redundant.** Inspect (https://inspect.aisi.org.uk/) already provides agent execution and logs; its metrics (https://inspect.aisi.org.uk/metrics.html) include clustered uncertainty. promptfoo (https://www.promptfoo.dev/docs/configuration/expected-outputs/) covers assertions, cost/latency constraints, and trajectory checks. LangSmith (https://docs.langchain.com/langsmith/compare-experiment-results) and Braintrust (https://www.braintrust.dev/articles/what-is-prompt-evaluation) already compare experiments and inspect regressions. OpenAI Evals (https://github.com/openai/evals) supplies evaluation infrastructure and a benchmark registry. τ-bench's pass^k (https://arxiv.org/abs/2406.12045) measures consistency across repeated trials; it complements a release comparison rather than replacing one. Paired differences, clustering, and power analysis are established recommendations in "Adding Error Bars to Evals" (https://arxiv.org/html/2411.00640v1).

   Your distinctive contribution is **audited memory interventions, bundle decomposition, and evidence-linked localization**. Build an analysis extension: import Inspect `EvalLog` samples, epochs, scores, and traces; retain original trace links; export verdict JSON for existing CI. Cut the replacement runner and mandatory factorial/control machinery.

6. **Research scripts are not reliable release infrastructure.** The harness, memory ranking, controls, trajectories, and study-specific bootstrap exist. The report is manageable; general execution and provenance are substantial work. Runner restart overwrites logs (skillrc/runner.py:151); the WebShop analysis loader (scripts/analysis/r2_2026-10-31.py:114) keys only by task ID and silently overwrites duplicates. `passport.py` is a memory admission card, not a complete run passport. Error episodes lose consumed tokens/steps; local dollar cost defaults to zero.

   Minimum adoption requirements: validated immutable task/environment identities, repetition IDs, explicit missing/error policy, artifact/model/scorer hashes, measured retry-inclusive latency, and a custom-data adapter. The current panel is **393/500 fashion**; category stratification does not fix representativeness or dependence among related goals.

**§4 arms**

| Arm | Decision | Reason |
|---|---|---|
| L_NONE | Run | Relevant baseline. |
| L_NONE_DUP | Modify | Second randomized block, paired with candidate repetition. |
| L_PUB_TRUE | Run | Replicate loss and measure latency. |
| L_PUB_CHORAL | Run | Historical comparator; call its effect control-specific. |
| L_PUB_ABL | Modify | Small semantic suspect set; freeze ranking and IDs. |
| L_PUB_ABLCOMP | Modify | Several matched subsets; one arbitrary draw is weak evidence. |
| L_PUB_ADVICE | Modify | Useful sensitivity test; debugging advice may transfer, so cannot isolate "style." |
| NONE, PUB_TRUE unlabelled | Drop initially | Existing factorial already establishes the case. |
| L_PUB_FIX | Modify | Freeze exact edit; validate on held-out goals, not another tuned pass over these 500. |
| **Add L_PUB_TRUE_DUP** | **Run** | Candidate variability is essential. |
| **Add suspect-only add-back** | Defer | Follow a convincing recovery result. |

Counterbalance GPU assignment and order; pin model/tokenizer/vLLM and goal snapshots; record warm-up, caching, concurrency, and retries. Temperature zero and server seed zero do not establish determinism.

**Smallest version I would ship**

Ship **offline `gate report`**: two-arm paired comparison, optional explicit control/factor contrasts, one primary metric, a few guardrails, strict input validation, JSON/HTML output, and deterministic paired replays. Add one Inspect adapter and coverage/decision tests.

The interviewer's first probes will be: why existing tools are insufficient, whether pairing survives environment changes, whether intervals have correct coverage, and whether diagnosis predicts a successful repair.

The ten-minute experience needs bundled real logs, one CPU-only command, a visible verdict-to-contrast-to-trajectory path, and a reproducible report. A live endpoint-dependent Docker run is an optional demonstration with measured prerequisites; n=20 cannot guarantee INCONCLUSIVE.

Defer automatic arm generation, universal control synthesis, pipeline-level certification, and automatic causal verdicts.

---

## Author response (Claude, 2026-11-16)

All six weaknesses accepted. Verified independently: runner.py:151 opens the episode file with "w" (restart
overwrites); r2 loader keys by task_id only. WebShop splits per scripts/webshop/make_webshop_configs.py: test
0–499, pool collection on train sessions ≥ 1500, so sessions 500–1499 are unused and available as held-out goals.
Changes are in GATE_DESIGN_v1.md.
