# `skillrc-gate` design v1 (2026-11-16, revised after Codex round 1)

Supersedes GATE_DESIGN_v0.md. Review record: REVIEW_r1_codex.md. Internal; not committed. Nothing implemented yet.

## 0. What changed from v0

| v0 | v1 | review item |
|---|---|---|
| Runner + planner + control synthesis + analysis | Offline analysis tool over episode logs; the SkillRC runner only produces this case's logs | W5 |
| PASS/BLOCK rule could fire both | Four states with one margin; CI exits 0 only on PASS | W1 |
| Duplicate-baseline variance "added" to the interval | Both key arms repeated in interleaved blocks; repetitions averaged within task; task-cluster bootstrap; block effect as a diagnostic; A/A calibration | W2 |
| Automatic factor detection; CTRL − BASE as reading cost | Interventions declared explicitly with content hashes; contrasts declared in the spec; reading cost = CTRL vs the same harness without the text | W4 |
| Lexicon-selected suspects (17/30), one complement draw, test set | 6 manually reviewed suspects frozen by id; 3 matched complement subsets; confirmatory on unused dev goals 500–999; interval bounds required | W3 |
| Research-grade logs | Episode schema v1 with identities, repetitions, error policy, hashes, retry-inclusive latency; runner fixes | W6 |
| Over-stated evidence | §1 corrected (below) | evidence check |

## 1. Evidence, corrected

- Bundled release (Qwen3-8B, WebShop 1k, test 0–499, one window, `PREREG_r2_2026-10-31.md` arm P):
  label + memory vs old baseline +0.132; label alone +0.251; memory given label −0.119
  (paired SD 0.509, SE 0.023, stratified CI [−0.164, −0.075]). Content value vs choral control, given label:
  PUB −0.140 [−0.183, −0.096], R1 −0.284, R2 −0.173.
- Reading cost (choral − no memory, given label) is not distinguishable from zero, but the intervals allow losses
  up to about 0.05 (PUB +0.021 [−0.019, +0.060]). We do not claim it is zero.
- gpt-4o-mini, WebShop, unlabelled: checkout completion 0.366 → 0.204, task success 0.194 → 0.086, 15-step
  exhaustion 0.634 → 0.796, score 0.294 → 0.154.
- Searches per episode, TRUE minus irrelevant control: +3.33 / +1.06 / +0.11 (gpt-4o-mini / Qwen3-8B /
  gpt-4.1-mini); content values −0.158 / −0.116 / −0.002. A descriptive association over three readers; gpt-4.1-mini
  is near the floor.
- ALFWorld precedent: removing all five cooling rules gave +0.143 [−0.016, +0.302] on cool games — a point
  estimate of partial recovery, not established recovery; removing the single suspected rule did not recover.
- MDE at n = 500 paired WebShop sessions, one repetition: ≈ 0.064 (80% power, two-sided α = .05). A 0.02 margin
  would need ≈ 5,100 pairs.
- Test panel composition: 393/500 fashion. Category stratification does not make it representative.

## 2. Product definition

**One sentence.** `skillrc-gate` reads the episode logs of two or more agent versions run on the same tasks and
returns a release verdict for each declared change, with paired intervals, cost/latency guardrails, a breakdown
of the change into declared interventions, a behaviour-shift table, and paired replays that link the verdict to
real trajectories.

**Distinctive part (what existing tools do not do together):** audited context interventions (did the memory reach
the prompt, at the declared length), conditional effects of bundled changes, length-matched control contrasts, and
evidence-linked localization from verdict → contrast → behaviour → memory item → trajectory.

**Not built here:** agent execution (Inspect AI, SkillRC runner, others), assertion-style output checks (promptfoo),
experiment dashboards (LangSmith/Braintrust). The gate imports their logs and emits JSON a CI job can act on.

## 3. Pipeline (nodes and edges)

Nodes:

- **G1 Ingest.** Input: episode files in a supported format (SkillRC JSONL; Inspect `EvalLog` via `inspect_ai.log`).
  Converts each to episode schema v1 (§5) and validates it. Output: validated episode tables, one per arm.
- **G2 Spec check.** Input: the gate spec (YAML): arms (name → log files), declared interventions (name → config
  patch + content hash of any added text), contrasts (name → arm pairs or a factorial cell set), primary metric, margin
  δ, guardrails with tolerances, α. Checks that every arm in a contrast differs only by the declared interventions
  (config hashes) and that task identities match across arms. Output: contrast plan, or INVALID with reasons.
- **G3 Injection audit.** Input: arms with added text + their audit records (from the run: injected block token
  count, position, content hash). Checks each against the declared intervention. Output: pass/fail per arm.
- **G4 Pairing.** Input: validated tables. Joins arms on task_uid (task id + instruction hash + env snapshot hash);
  averages repetitions within task per arm. Output: paired task table, plus coverage report (missing, errored,
  mismatched tasks).
- **G5 Statistics.** Input: paired task table, contrast plan. For each contrast: mean difference, 95% interval by
  task-cluster bootstrap stratified by task type (pools resampled if the contrast is over replicate pools); block
  effect (rep1 vs rep2 difference-in-differences) reported as a diagnostic; power curve for the declared δ and MDE at
  this n. Guardrails: Δ tokens/episode, Δ steps, Δ p50/p95 episode latency, Δ error rate. Holm adjustment when the
  report claims any harm among guardrails or factorial cells; exploratory contrasts marked as such. Output: stats JSON.
- **G6 Behaviour diff.** Input: paired task table with trajectories. Per-env action categories and episode outcome
  types (success, wrong item, budget exhausted, loop, invalid streak), Δ shares with paired intervals, ranked by |Δ|.
  Optional: item-level linkage — for each memory item, whether the episode's actions match the item's declared
  behaviour tag (tags written by a person before the run, not inferred). Output: behaviour table, suspect items.
- **G7 Verdict.** Input: stats JSON, audits, coverage. States:
  - INVALID: a failed audit, task-identity mismatch, error rate above the declared cap, or missing arms. Blocks release.
  - BLOCK: upper bound of Δprimary < −δ, or a guardrail's lower bound exceeds its tolerance (confirmed violation).
  - PASS: lower bound of Δprimary > −δ and every guardrail's upper bound is within tolerance. Flag "detectable but
    tolerated regression" when the upper bound is < 0.
  - INCONCLUSIVE: otherwise; prints the power curve, not "n needed for the observed effect".
  One verdict per declared contrast; the bundle verdict and the per-intervention conditional verdicts are shown side
  by side. CLI exit code 0 only for PASS.
- **G8 Report.** Input: everything above. Static HTML + JSON: verdict card → contrasts table → decomposition chart
  → behaviour shifts → suspect items → paired replays (examples chosen by a fixed rule: the first k task ids where the
  baseline arm scores higher, plus k random pairs with a fixed seed; first divergent step highlighted).

Edges (trigger → payload):

- G1 → G2: every input file parsed; schema-valid tables (a schema failure → INVALID).
- G2 → G3: contrast plan valid; arms with declared added text.
- G3 → G4: audits done; arm list with audit status.
- G4 → G5: pairing done; paired task table + coverage report.
- G4 → G6: same table (trajectories).
- G5 → G7: stats JSON. G3, G4 → G7: audit and coverage status.
- G6 → G8: behaviour table, suspect items. G7 → G8: verdicts.

Localization is not an automatic node. G6 produces hypotheses; testing them needs new arms declared by a person
(§4, block D), which then re-enter G1.

## 4. New runs for the case (pre-registration and Codex check before launch)

Common: Qwen3-8B (pinned revision), vLLM (pinned version), thinking off, temperature 0, page label ON, WebShop 1k
server `--seed 0`, PYTHONHASHSEED=0, 15 steps, latency recorded per call including retries; arms interleaved and
GPU assignment counterbalanced; two GPUs.

**Block T — test 0–499: replicate the regression, measure latency, noise.**

| arm | content |
|---|---|
| T_NONE_r1, T_NONE_r2 | no memory, two interleaved repetitions |
| T_TRUE_r1, T_TRUE_r2 | PUB, two repetitions |
| T_CHORAL | PUB-shaped choral control (historical comparator; its contrast is control-specific) |

**Block D — dev 500–999 (never used for collection, distillation or analysis): localization and repair.**

Suspect set, frozen now by id (rules telling the agent to repeat or widen the search with no stop condition),
6 of 30: beauty #4 (pagination), electronics #5 (pagination), fashion #4 (re-search same criteria), garden #3
(repeat search same parameters), grocery #3 (refine terms and try again), grocery #6 (return to search and adjust).
Category profile 1/1/1/1/2.

| arm | content |
|---|---|
| D_NONE | no memory |
| D_TRUE | PUB |
| D_ABL | PUB with the 6 suspects replaced in place by length-matched choral sentences; retrieval ranked by the original pool; bytes and positions audited |
| D_COMP1–3 | PUB with 6 non-suspect rules replaced the same way, three seeded draws matched on category profile and word count (±20%) |

Confirmatory contrasts (D): ABL − TRUE > 0 and ABL − mean(COMP) > 0, both by interval lower bound.
Repair verdict through the gate: ABL vs NONE with δ declared below. Power: a full recovery (+0.12) is well above the
MDE; a half recovery (+0.06) is near it, and we say so in advance.

**Deferred:** off-topic advice control (cannot isolate "style"), 30-step budget pair (mechanism probe), suspect-only
add-back (sufficiency), hand-written stop-condition fix, unlabelled harness rerun.

About 11 arms × ~1–1.5 GPU-h on two GPUs.

## 5. Episode schema v1 and runner fixes

Episode: run_id, arm, repetition, block_order, env, env_snapshot_hash, split, task_id, task_uid, task_type, seed,
score, success, steps, prompt_tokens, completion_tokens, memory_tokens, n_calls, n_retries, latency_s (episode) and
per-call latencies, error (string) with tokens/steps consumed so far, trajectory [{step, action, obs}].
Run manifest: config hash, model id + revision, tokenizer hash, vLLM version, GPU type, server seed, memory pool
sha256, scorer version, start/end time.

Runner fixes: resume by appending and skipping finished (task_uid, seed, repetition) keys instead of overwriting;
time every call with retries; keep consumed tokens on error; write the manifest.

## 6. Ten-minute path, tests, packaging

- `pip install -e .` then `skillrc-gate report examples/webshop_memory/spec.yaml -o report/` — CPU only, bundled real
  logs (r2 arm P + block T/D once run), under a minute. The HTML opens on the verdict card.
- Static copy of that report on the project page.
- Tests: decision-table unit tests for G7; coverage simulation (synthetic paired data with known effects, check
  interval coverage and verdict error rates); A/A on real duplicate logs (T_NONE_r1 vs r2 must not BLOCK more than α);
  schema and identity validation (a deliberately shuffled task mapping must yield INVALID — the ALFWorld re-download
  changed id → game mapping in this project); Inspect adapter round-trip on a real Inspect log produced by Inspect's
  built-in `mockllm` model on a toy task (no GPU, no API).
- GitHub Action example: run `skillrc-gate report` on committed logs; fail unless PASS.
- Docker live demo and video: later, optional.

## 7. Limitations stated up front

Synthetic 1k-product WebShop with generated goals and a fashion-heavy panel; a 15-step budget; one reader for the new
runs; one environment where memory clearly hurts; controls are topic-specific; two repetitions estimate run effects
weakly; benchmark harness, not a production system.

## 8. Open questions for round 2

1. Does v1 resolve W1–W6? What is still wrong?
2. Is generating the Inspect test log with `mockllm` enough to claim an Inspect adapter, or does v1 need one real
   agent eval through Inspect?
3. Block D on dev goals only: right call? Is the suspect set acceptable as frozen, and is 3 complement draws enough?
4. Margin δ for the case: 0.05 with n = 500 and two repetitions — sensible, or should the case declare δ from a
   product argument?
5. Anything to cut further for a first release.
