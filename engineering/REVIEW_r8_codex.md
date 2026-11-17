# Codex review, round 8: re-review of the round-7 fixes (2026-11-16)

Reviewer: Codex MCP (read-only, xhigh), thread 01a1e580-d639-7433-a581-5ca669bf28bc. Verbatim text follows
(file links shortened to paths).

---

Score: 7.5/10
Verdict: **FIX-FIRST**

Not yet: a clean Inspect import and `PREREG_FROZEN` are insufficient. Most fixes work, but recovery and provenance still have gaps, and the reporting change introduced a false claim.

**57 read-only tests passed.** I did not rerun filesystem-writing tests. Both hash lists verify; both registries match their 500 tasks; the actual WebShop configs pass the new fingerprint comparisons.

| Round-7 finding | Status | Verification |
|---|---|---|
| 1. Provenance | **Partly** | Fingerprint comparisons work, but absent provenance remains acceptable: schema.py:83, pairing.py:92. |
| 2. Resume/journaling | **Partly** | Fingerprint checking and durable starts implemented; truncated records remain unrecoverable: runner.py:187, runner.py:229. |
| 3. Launcher ownership | **Partly** | Hash verification and chain process groups work. Parent-launched smoke servers remain untracked: launcher:63. |
| 4. Preparation/cache | **Resolved** | Explicit caching-off and identical smoke workloads: launcher:85, launcher:109. |
| 5. Retries | **Resolved** | SDK retries disabled; initial attempt plus configured retries: llm.py:129, llm.py:151. |
| 6. Missing-task identities | **Partly** | Completely absent tasks work; failed attempts carrying unknown identities do not: pairing.py:103. |
| 7. Inspect payload provenance | **Resolved for this producer** | Execution-time digest checked and logged arguments flattened: agentdojo_tips.py:43, inspect_log.py:99. Full import remains pending. |
| 8. Target independence/default | **Resolved** | Load failures isolated; explicit default required: engine.py:56, spec.py:156. |
| 9. Calibration | **Resolved** | Exact binary shifts, truth summaries, Wilson intervals, production-budget token simulation: calibration.py:40, calibration.py:111. This remains scenario-specific calibration, not guaranteed control. |
| 10. Reporting | **Partly** | Sensitivity, completed-task guardrails and mean-over-COMPs implemented; regression flag mishandles missingness: engine.py:253, verdict.py:177. |

Remaining defects, including regressions:

1. **Missing provenance still produces PASS.** I reproduced PASS with both manifests having `model_revision=None`, `scorer_version=None`, and empty serving/environment dictionaries. Equality of missing information is not evidence of experimental equivalence. pairing.py:92 checks manifest existence, not completeness. **Minimum fix:** define producer-specific required provenance and mark incomplete evidence DESCRIPTIVE. Inspect must satisfy that requirement independently of its now-correct tips hash.

2. **The registry rejects exactly the interrupted-attempt case it should repair.** Two `unknown:7` infrastructure-failure cells among 1,000 expected cells produce INVALID despite a valid registry and missingness below the cap. pairing.py:103 treats the placeholder as a contradictory identity. **Fix:** resolve unknown identities and types from the registry **before** building cell dictionaries and failure counts; continue rejecting conflicting known identities.

3. **Smoke failure can orphan its WebShop server.** launcher:114 starts a server in the parent shell, but its group never enters `PGIDS`. Failure at line 119 exits before normal cleanup. **Fix:** register the server immediately upon spawning, including health-check failure paths. Bind CUDA using the recorded UUID at line 83 rather than assuming CUDA and NVML ordinal order agree.

4. **Crash recovery assumes every append completed.** Unconditional JSON decoding at runner.py:198 and line 210 prevents resume after a truncated final episode or journal record. **Fix:** preserve and recover an incomplete final record using the journal; reject interior corruption. Add interruption tests at both write boundaries.

5. **"Detectable regression" uses the wrong completion.** With only **2/600 cells missing**, I obtained primary PASS, pessimistic upper bound **−0.000097**, optimistic upper bound **+0.00417**, and the detectable-regression flag. Negative effect is not established. verdict.py:177 must require **optimistic upper < 0**, while retaining pessimistic evidence for noninferiority.

**Must fix before launch:** defects 1–5, targeted regression tests, refreshed hashes, and the completed Inspect import review. Then freeze. No change to δ, guardrail tolerances, payloads, or the statistical decision rule is needed.

---

## Author response (Claude, 2026-11-16)

1. `schema.provenance_gaps`: model, model_revision, serving.engine_version, config for every producer; SkillRC also
   scorer_version and, for WebShop, the server seed, instruction-map hash and server-script hash; Inspect also
   env.inspect_ai. Gaps → DESCRIPTIVE. The Inspect producer now passes the pinned snapshot and serving details as
   `eval(metadata=...)`; the adapter reads them (the Inspect run is redone again).
2. `ArmData._build` resolves `unknown:<id>` from the registry before building cells; without a registry the
   attempt is placed under the grid's placeholder and paired via the other arm; placeholders never register a
   second identity.
3. `start_ws` registers the server in the cleanup list before any health check. GPU binding: `CUDA_DEVICE_ORDER=
   PCI_BUS_ID` so CUDA ordinal k is nvidia-smi row k (whose UUID is recorded); after start-up both recorded UUIDs
   must carry a compute process. (CUDA_VISIBLE_DEVICES=<UUID> was not used: vLLM 0.24 parses the variable as
   integers in places.)
4. `runner._read_jsonl_recovering`: a truncated final line (episodes or journal) is moved to `.partial.<time>` and
   the file cut back; the journal then records the attempt as interrupted. Interior corruption stops the resume.
5. The flag requires the optimistic upper bound below 0.

Tests: `tests/gate/test_review_r8.py`, `tests/test_runner_review_r8.py`; 79 tests pass. Hashes refreshed
(`FROZEN.sha256` 41999a41…e556, code list bac19d6c…66ea).
