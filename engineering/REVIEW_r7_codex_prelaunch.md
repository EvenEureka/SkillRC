# Codex review, round 7: pre-launch review of the implementation and the WebShop pre-registration (2026-11-16)

Reviewer: Codex MCP (read-only, xhigh), thread 01a1e580-d639-7433-a581-5ca669bf28bc. Verbatim text follows
(file links shortened to paths).

---

Score: 7/10
Verdict: **FIX-FIRST**

The statistical core is substantially correct: strict interval/p-value equivalence, Holm, degeneracy, missing-cell bounds, same-block localization, and verdict precedence. I reran **43 core tests successfully**, not all 56. Historical means match; complement enumeration reproduces **210 subsets and the three selected draws**. Payload edits, recorded hashes, halves, and chain files agree. The blockers are mainly integrity and execution.

1. **The gate can PASS an undeclared experimental change.** engine.py:57 checks config differences and declared candidate file hashes, but ignores environment, serving, scorer, and undeclared file-hash differences. I reproduced PASS after changing all four. Within-arm checking also ignores environment/serving differences (skillrc_jsonl.py:143). **Minimum fix:** compare an immutable scientific fingerprint across runs and arms, permitting only explicitly declared interventions and execution metadata. Missing required provenance must prevent a release verdict.

2. **Resume can mix experiments and overwrite the evidence.** runner.py:168 compares only config hashes; changing payload bytes at the same path or serving settings is accepted. runner.py:269 then replaces the old manifest. Separately, attempts are recorded only after completion (runner.py:289); process death leaves no failed attempt, bypassing rerun/failure accounting. **Fix:** immutable fingerprints, append-only execution segments, and durable attempt-start/completion records with interrupted-attempt recovery.

3. **The job does not reliably own its GPUs or descendants.** run_gate_ws_block.sge:64 ignores scheduler GPU allocation; `CUDA_VISIBLE_DEVICES` does not make `nvidia-smi ... | head -1` identify the selected GPU, so both chains record the first GPU. Cleanup at line 39 does not track runner/WebShop descendants inside background chains. Health checks can accept a pre-existing server on a colliding port. **Fix:** use allocated GPU UUIDs, supervise process groups, and verify server ownership. The sentinel checks at line 33 must verify frozen artifact hashes, not merely file existence.

4. **Latency receives asymmetric preparation.** run_gate_ws_block.sge:91 runs three full ABL episodes only on chain A after both servers' tiny warm-ups. Persistent prefix caching can advantage subsequent arms differently. **Fix:** use identical held-out smoke/warm-up procedures, reset cache state before measurement, and explicitly freeze caching behavior.

5. **"Five retries" is not what executes.** llm.py:128 leaves OpenAI SDK retries enabled; line 150 adds five outer attempts. Retryable failures can therefore produce **15 transport attempts**, while recorded retries omit SDK retries. **Fix:** disable SDK retries, define initial attempt plus retry count precisely, and test timeout/retry accounting.

6. **A wholly missing task defeats the promised bounding policy.** pairing.py:101 invents `missing-task:<id>` when an arm has no observation. The other arm has its real UID, so line 143 declares different task sets. I reproduced INVALID with one absent task out of 500, below the missingness cap. **Fix:** derive identities and strata from a frozen task registry, independently of successful attempts.

7. **Inspect payload provenance is retrospective.** inspect_log.py:92 hashes the tips file at ingestion time, not execution time. Matching today's hash does not establish what ran. AgentDojo gate.yaml:24 also permits arbitrary changes to the entire `task_args` object. **Fix:** embed producer-time hashes or verify the logged system prompt; allow only the specific tips argument. The smoke's utility/epoch mapping is correct, but adapter error/retry/provenance fixtures are still needed.

8. **Target independence breaks on loading failures.** cli.py:23 evaluates everything before selecting a target; an absent unrelated arm can invalidate an otherwise evaluable target. spec.py:158 also accepts a sole target without the explicitly required default. **Fix:** isolate loading failures by target and enforce the default rule.

9. **Calibration is evidence of limitations, not demonstrated error control.** The n=500 WebShop boundary row has 23/600 PASSes: **3.83%, Monte Carlo 95% Wilson interval 2.57–5.69%**. Binary "boundary" scenarios are not exactly at the boundary because of clipping (calibration.py:45); line 100 reports only the final replicate's truth. **Fix:** correct the generator/truth summary, add Monte Carlo intervals, and calibrate representative guardrail cases at production draw counts.

10. **Reporting still falls short of the contract.** Completed-cell guardrails disappear (engine.py:235); cost BLOCK suppresses the primary-only tolerated-regression flag (verdict.py:172). Promised δ sensitivity (prereg:74) and mean-over-COMP analysis are absent. **Fix:** implement them or explicitly amend commitments before freezing.

**Keep +30%.** Historical prompt tokens increase 55.36%; changing tolerance to obtain PASS would undermine the case. However, prereg:87 should clarify that equal payload length does not imply equal episode cost: ABL can change trajectory length.

**Must fix before launch:** 1–7, targeted regression tests, corrected frozen commitments/hashes, and review the completed Inspect run.

**Fix later:** 8–10 implementation/calibration improvements can follow data collection, but must precede final interpretation and release of the tool.

---

## Author response (Claude, 2026-11-16)

All ten fixed before launch (8–10 included), with regression tests in `tests/gate/test_review_r7.py` and
`tests/test_runner_review_r7.py`; 74 tests pass.

1. `schema.fingerprint` / `fingerprint_diff`: config, every file hash, model and revision, scorer, scientific
   serving and env fields; only declared intervention keys/files and execution metadata (ports, GPU index/uuid,
   chain, block, job id) may differ. Used for targets (`engine.check_interventions`) and within arms (both adapters).
2. Resume compares the full fingerprint; the manifest is append-only (`segments`, one per launch); an attempt-start
   log (`<run>.attempts.jsonl`, fsynced) lets a resume record interrupted attempts as infrastructure failures.
3. Job script: exactly two visible GPUs required (cgroup isolation), UUID/name per chain from the nvidia-smi list;
   `set -m` process groups killed as groups; ports checked free before use; vLLM checked to serve qwen3-8b;
   `sha256sum -c FROZEN.sha256` instead of file-existence sentinels.
4. Prefix caching OFF (`--no-enable-prefix-caching`); the same smoke (no-memory config, sessions 1500–1502, outside
   every panel) on both chains before measuring.
5. SDK retries off (`max_retries=0`); `max_retries` = retries after the first try; accounting tested.
6. Frozen task registries (`configs/gate_ws_2026_11/registry_{T,D}.jsonl`, checked against 500 logged identities);
   without a registry, a wholly missing task takes the other arm's identity.
7. `agentdojo_tips` takes `tips_sha256`, checked when the task is built; the adapter flattens task args, records
   only producer-time hashes, and hashes the logged system message; the intervention allows only
   `task_args.tips_file` and `task_args.tips_sha256`. The Inspect run is redone with this.
8. Arm load failures are isolated per arm; `default: true` required even for a single target.
9. Binary generator without clipping, mean truth over replicates, Wilson intervals, token-guardrail calibration at
   20,000 draws.
10. Completed-cell guardrails shown (not used) when cells are missing; tolerated flag independent of guardrails;
    margin sensitivity (0, 0.02, 0.05); list-baseline contrasts (mean over COMPs).
