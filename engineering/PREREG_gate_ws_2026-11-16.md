# Pre-registration: WebShop release check with skillrc-gate (2026-11-16)

Status: written before any episode of these runs. Frozen when `configs/gate_ws_2026_11/PREREG_FROZEN` is written
(after the Codex pre-launch review recorded in `REVIEW_r7_codex_prelaunch.md`). Later changes go into dated
addenda at the end of this file and say what had been seen.

Design contract: `GATE_DESIGN_v3.md` as amended by `GATE_DESIGN_v4.md` (Codex rounds 1–5, READY). The 32B extension
(`GATE_DESIGN_v5_32b.md`) gets its own addendum before its pilot.

## 1. Question

A team wants to ship a distilled memory (30 rules, `data/pools/webshop1010_insights.jsonl`, "PUB") for a WebShop
agent. Does it pass a release check against the agent without memory? If it is blocked, does replacing six rules
that tell the agent to keep searching (with no stopping rule) repair it, and is that repair specific to those six
rules compared with three matched edits of other rules?

## 2. Runs

| item | value |
|---|---|
| Reader | Qwen3-8B, snapshot `b968826d9c46dd6066d109eabc6255188de91218`, vLLM 0.24.0, bf16, max_model_len 16384, gpu_memory_utilization 0.90, enforce-eager, **prefix caching off**, thinking off, temperature 0, 256 completion tokens |
| Harness | SkillRC ReAct (thought_action, 2 few-shot, admissible actions shown), WebShop 1k products, server `--seed 0`, PYTHONHASHSEED=0, page label ON, 15 steps |
| Memory | `expel`, BM25, token budget 524 (= PUB's o200k length): whole pool injected for every session |
| Panel T | test sessions 0–499 (categories: fashion 393, garden 57, beauty 23, electronics 19, grocery 8) |
| Panel D | sessions 520–1019 (fashion 405, garden 67, electronics 14, beauty 7, grocery 7). Usage manifest: 0–499 test; 500–519 used by `configs/webshop_qwen_train20.yaml`, excluded; ≥ 1500 pool collection; vocabulary-overlap scripts read all instruction texts. Held out from outcome-based development; same product catalog |
| Arms | T: NONE r1/r2, TRUE r1/r2. D: NONE r1/r2, TRUE r1/r2, ABL r1/r2, COMP1–3 one repetition each (13 arm-runs; 16 run configs because each COMP is split into two halves) |
| Configs | `configs/gate_ws_2026_11/GWS_*.yaml` from `scripts/gate/make_gate_ws_configs.py` |
| Execution | one episode at a time per chain; one vLLM server per GPU serving only its chain (the job must see exactly its two granted GPUs; UUIDs recorded per chain); a fresh seeded WebShop server per run on a port checked free; 5 discarded warm-up requests per server, then the same smoke on both chains (no-memory config, sessions 1500–1502, outside every panel, output not analysed); latency = client-side wall time of LLM calls incl. retries, summed per episode; LLM timeout 120 s, one try plus up to 5 retries (SDK retries disabled) |
| Task identity | frozen registries `configs/gate_ws_2026_11/registry_{T,D}.jsonl` (task_id, task_uid, category from the seeded instruction map; identical to the identities computed from 500 logged r2 observations) |

### Payloads (`data/pools/gate_ws_2026_11/`, built by `scripts/gate/build_ablation_payloads.py`)

- Suspect criterion: "explicitly directs another search, a revised search, or pagination after inadequate results,
  and contains no search-count, step-budget or stopping rule"; excluded: initial-query construction, filters,
  inspecting current results.
- Suspects (PUB lines, 1-based): 4 (beauty), 11 (electronics), 16 (fashion), 21 (garden), 27 and 30 (grocery);
  86 body words; text sha256 per line in `payload_audit.json`.
- Complements: category profile 1/1/1/1/2, total removed body words within ±5% of 86 → 210 eligible subsets.
  Draws (seeds 20261117–19): COMP1 lines 3, 10, 17, 19, 26, 29 (82 words); COMP2 1, 8, 14, 19, 26, 29 (83);
  COMP3 6, 10, 15, 19, 28, 29 (82). Pairwise overlap 3, 3, 2 lines; lines 19 and 29 are in all three. ABL removes
  86 words of advice, the complements 82–83; both residuals are disclosed.
- Every payload: replaced items have the original body word counts, the injected block has PUB's token count under
  o200k (524) and the Qwen3 tokenizer (529); no nonce word was needed; per-item token differences −2 to +4.
- sha256: PUB `89893797…54b4`, ABL `0a601d39…d4ad`, COMP1 `7d879c40…87a6`, COMP2 `f6b8a536…8d2f`,
  COMP3 `c227eb67…6cf6`; `payload_audit.json` `6a030194…098f`.
- Injection audit (`scripts/gate/audit_gate_ws_injection.py`, `results/gate_ws_2026_11/injection_audit.json`
  `1122c5a9…e8e0`): for every memory config and every session, 30 items injected, same order as TRUE, 524 o200k and
  529 Qwen3 tokens, payload differs from PUB exactly at its declared lines. Passed (`AUDIT_OK`).

### Schedule (`schedule.json` `9fe3cd7d…0f2`, `halves.json` `3abf819f…bccf`)

Two blocks (time windows; block 2 starts after block 1 ends), two GPU chains per block, order within each chain
drawn with seed 20261116 + 10·block + chain:

| block | chain A (GPU 0) | chain B (GPU 1) |
|---|---|---|
| 1 | D_ABL_R1 → T_NONE_R1 → D_NONE_R1 → D_COMP3_HA | D_COMP2_HB → D_TRUE_R1 → D_COMP1_HA → T_TRUE_R1 |
| 2 | D_COMP2_HA → T_TRUE_R2 → D_COMP1_HB → D_TRUE_R2 | D_ABL_R2 → D_NONE_R2 → D_COMP3_HB → T_NONE_R2 |

Halves: within each category of panel D, tasks shuffled with seed 20261116 and assigned alternately; H_A 252,
H_B 248 tasks. Every repeated arm has r1 in block 1 and r2 in block 2 on different chains; each complement has one
half in each block, on different chains.

## 3. Analysis (frozen)

Tool: `skillrc-gate report examples/webshop_release_check/gate.yaml` (spec sha256 `71185c2c…3961`; analysis-code
hashes in `engineering/gate_code_sha256_2026-11-16.txt`, sha256 of that list `a997f364…2529`). Execution inputs
(configs, payloads, pools, registries, schedule, harness code, server and job scripts, analysis spec) are listed with
their sha256 in `configs/gate_ws_2026_11/FROZEN.sha256` (sha256 `e0eba660…1754`, 57 files; regenerated and verified by `scripts/gate/freeze_lists.sh`); the job refuses to start unless
`sha256sum -c` passes. Statistics as in the contract:
task-cluster bootstrap stratified by category, 20,000 draws, PCG64 seed 20261116, order-statistic 95% interval,
one-sided α = 0.025, repetitions averaged within task, same-block pairing for ABL vs each COMP.

Release targets (independent):
- **C1 ship PUB** (default target): T_TRUE vs T_NONE, intervention `memory_pub`.
- **C2 ship the repaired memory**: D_ABL vs D_NONE, intervention `memory_abl`.
- Each: primary = mean WebShop score, δ = 0.05 (illustrative policy; ≈ 9% relative to the r2 baseline 0.577;
  sensitivity at δ = 0, 0.02, 0.05 reported); guardrails prompt tokens per episode (relative mean ≤ +30%) and p95
  episode latency (relative ≤ +30%), illustrative tolerances.

Diagnostic contrasts (never feed a verdict): R_replication (T_TRUE − T_NONE < 0); localization joint claim =
L1 (D_ABL − D_TRUE > 0) and L2_i (D_ABL − COMP_i > 0, same-block pairs, i = 1, 2, 3), all four lower bounds > 0,
no adjustment (intersection–union); the claim covers these three comparator edits only. Also reported, not part of
the claim: D_ABL minus the mean of the three COMPs (per task, mean of the three same-block differences).
Descriptive: D_TRUE − D_NONE, A/A (T_NONE r2 − r1), per-block contrasts, behaviour shifts, replays.
Reported with every target: margin sensitivity of the primary component (δ = 0, 0.02, 0.05); the primary-only
"detectable but tolerated regression" flag independent of the guardrails; guardrails on completed cells (shown,
not used) when any cell is missing.

## 4. Expectations stated in advance

- From r2 arm P (same reader, harness and panel T; memory given label −0.119 [−0.164, −0.075]): C1's primary
  component is expected to BLOCK and R_replication to be supported.
- In r2 the memory raised prompt tokens per episode by about +55%, so the +30% token guardrail is expected to be
  violated for C1 and probably for C2: ABL injects the same 524 tokens per call, but equal payload length does not
  mean equal episode cost, since ABL can change how many steps an episode takes. C2 can therefore BLOCK on cost
  alone even if its score is non-inferior; the report shows the primary component separately. The tolerance stays
  at +30% as frozen; it is not tuned to let either target pass.
- No prediction for the direction of C2's primary component or for the localization claim; every outcome is reported.

## 5. Failures and completeness

As in the contract: every attempt is logged, with a start record written (and fsynced) before it runs, so an
attempt cut off by a process death is recorded as an interrupted infrastructure failure on resume; LLM one try plus
up to 5 retries; a cell with an infrastructure failure is rerun up to 2 more times in the same block under the same
fingerprint (resume refuses any change to config, file contents, model, scorer or scientific serving/env fields;
each launch is appended to the manifest's `segments`); a run stops after 2 failed attempts in a row (exit 75) and
resumes after its WebShop server is restarted (at most 3 restarts); no exclusions; missing cells > 0.5% or cells with an
infrastructure failure > 2% of an arm → INVALID; missing cells bound the primary metric and make guardrails
INCONCLUSIVE. A crashed block is resubmitted and resumes; runs already complete are not repeated.

Job chain (`scripts/gate/submit_all.sh`): 8B block 1 → block 2 → 32B pilot → T1 → T2 → D1 → D2, each held on the
previous job. SGE holds wait for the previous job to end, not to succeed, so every job checks its predecessors' success
markers itself (`scripts/gate/prereq.py`; `BLOCK_<n>_DONE` / `PHASE_<p>_DONE` written by `scripts/gate/phase_marker.py`
with the episode hashes; a marker counts only if its structure and recorded hashes check out, and state files are
published atomically). Finalization is idempotent: a resubmitted job whose runs are complete only re-verifies its
marker and recomputes its decision (`scripts/gate/decision_io.py`), restoring a missing launch marker or stopping if a
saved decision no longer matches the evidence.

## 6. Calibration of the decision rule

`docs/gate/calibration.md` (simulated paired data, known effects): interval coverage 0.92–0.96 in regular
scenarios; at a true effect of exactly −δ, P(PASS) 0.018–0.041 and P(BLOCK) 0.018–0.035 against a nominal 0.025;
token guardrail at its tolerance (20,000 draws, n = 500): P(PASS) 0.037 [0.021, 0.065], P(BLOCK) 0.007. The error
budget is nominal; the bootstrap decision is mildly anti-conservative in some scenarios.

## 7. Limitations

Synthetic WebShop goals, fashion-heavy panels and small strata (7 sessions); one reader; 15-step budget; two
execution blocks do not estimate run-to-run variability across future windows; the replacement text is choral
singing, so ABL tests "these six rules replaced by unrelated text", not "these rules deleted".

## Addenda

### Addendum A (2026-11-16, written before any episode of any run): Qwen3-32B extension

Design `GATE_DESIGN_v5_32b.md`, reviewed in Codex round 6. Runs after the 8B blocks, because tensor parallel 2 needs
both of the project's GPUs.

- Reader: Qwen3-32B, snapshot `9216db5781bf21249d130ec9da846c4624c16137` (17 shards verified), vLLM 0.24.0,
  tensor parallel 2, prefix caching off; everything else as the 8B runs. Every 32B config equals its 8B counterpart
  except reader identity and output location (checked by `scripts/gate/make_gate_ws32_configs.py`), so the 8B
  injection audit carries over (identical Qwen3 tokenizer).
- Premise pilot (operational screen, not used for any claim): 100 sessions from 1020–1499, proportional to category
  (seed 20261116; {"fashion": 76, "garden": 16, "beauty": 3, "electronics": 3, "grocery": 2}), arms NONE and TRUE, one repetition. Proceed only if the NONE arm averages
  ≥ 0.30, no arm has > 2% of cells with an infrastructure failure, and every cell is scored
  (`scripts/gate/pilot32_check.py`, which first validates that both arms are exactly the frozen pilot: task set,
  identities, seed, repetition, frozen config, provenance, and differences only in the memory keys). No prompt or
  harness change after the pilot. A failed pilot ends the extension and is reported by which screen failed: the
  score screen ("the no-memory arm averaged below 0.30 on these 100 sessions") or the operational screen
  (unscored cells or infrastructure failures above 2%). Neither establishes that the reader is at the floor in
  general.
- Phase order is enforced by the job: T1 needs the pilot decision, T2 needs T1 complete, D1 needs T2 complete and
  the trigger, D2 needs D1 complete (`PHASE_<p>_DONE` markers with episode hashes,
  `scripts/gate/phase_marker.py`). The trigger is produced by `scripts/gate/trigger32_check.py` from the frozen
  32B analysis spec at the end of T2.
- Formal arms if the pilot passes: panel T NONE ×2, TRUE ×2, phases T1 (r1) then T2 (r2), order within each phase
  drawn with seed 20261116 + phase index. Release target C1_32 (same δ and guardrails as C1).
- Panel D (same nine arm-runs and payloads as 8B, phases D1, D2) is launched only if C1_32's **primary** component
  is BLOCK; the trigger is recorded in `results/gate_ws32_2026_11/D_TRIGGERED`; a branch not taken is reported.
- Cross-reader question, answered only by DD_reader = (32B TRUE − 32B NONE) − (8B TRUE − 8B NONE), paired by task
  on panel T, with its interval; different verdicts alone are not a finding. The readers run in different windows and
  serving configurations (TP 2 vs 1), so DD is conditional on those, not a causal effect of model size.
- Analysis spec: `examples/webshop_release_check_32b/gate.yaml` (sha256 `023e9966…f02d`); configs, schedule,
  chain files and pilot registry in `configs/gate_ws32_2026_11/`; job `scripts/gate/run_gate_ws32.sge` (one phase per
  job); frozen execution inputs `configs/gate_ws32_2026_11/FROZEN.sha256` (sha256 `df590434…40a1`, 63 files).
