# skillrc-gate

**Should this agent change ship?** Teams that build LLM agents change them constantly: a memory of lessons from past
runs, a new prompt, a new feature. Before each release someone compares the old and new version on a test set. That
comparison goes wrong in predictable ways: a small difference is just luck, a good change hides a bad one shipped with
it, something else changed between the two runs, crashed tasks are quietly dropped, or the score holds while the cost
explodes.

`skillrc-gate` reads the logs of the old and new version on the same tasks and answers, for each declared change:
**PASS** (ship), **BLOCK** (don't ship), **INCONCLUSIVE** (not enough evidence, not approved) or **INVALID** (the
experiment is broken). Logs too old to check get **DESCRIPTIVE** (numbers only, no decision). The exit code is the
answer, so a CI job can stop a release like a failing unit test. Case page with real results:
<https://eveneureka.github.io/SkillRC/gate/>.

Before answering it checks four things:

1. **Same conditions.** The versions may differ only in the declared change; model revision, settings, task list and
   the hash of every input file must match.
2. **Nothing dropped.** Every planned task needs a result; crashed tasks are rerun and counted, never removed.
3. **Real or luck.** A paired bootstrap over tasks gives the range of the true difference; BLOCK needs the whole range
   below the allowed drop, PASS needs the whole range above it.
4. **Cost.** Tokens and latency per task must stay within a limit.

It also shows how behaviour changed (e.g. how often the agent searched again) and puts both versions' runs on the same
task side by side. It does not run agents; it reads logs from the SkillRC runner, from
[Inspect AI](https://inspect.aisi.org.uk/), or from a plain JSONL format, and writes `result.json`, `summary.md` (for a
CI comment) and a self-contained `report.html`.

**The real case behind it.** A Qwen3-8B shopping agent on WebShop got a page label and a memory in one release: +0.13,
looked good. Split, the label was +0.25 and the memory −0.12. Re-tested in a pre-registered experiment, the gate
blocks the memory (score 0.57 → 0.44, tokens +58%) and refuses a tempting "fix": replacing the six "keep searching"
lessons stopped the extra searching but won back little of the score, and a random replacement of six other lessons
did better.

## Ten-minute path

```bash
pip install -e .                          # numpy, pydantic, pyyaml, jinja2; no GPU, no API key
skillrc-gate report examples/webshop_bundled_release/gate.yaml --out /tmp/gate_demo
open /tmp/gate_demo/report.html           # verdict cards -> intervals -> behaviour shift -> paired replays
```

The bundled logs are real (Qwen3-8B on 500 WebShop sessions). They predate the provenance contract, so their verdicts
are DESCRIPTIVE: the statistics are real, the decision is withheld.

## How it works

```
spec.yaml ─▶ G1 ingest ─▶ G2 spec + fingerprint check ─▶ G4 pairing ─▶ G5 statistics ─▶ G7 verdict ─▶ G8 report
                                                             └──────▶ G6 behaviour ─────────────────────┘
```

Nodes:

- **G1 ingest** — reads each arm's logs with its adapter (`skillrc`, `inspect`, `episodes`) into validated
  `Episode` records and a `RunManifest`; checks that every run of an arm has the same fingerprint.
- **G2 spec + fingerprint check** — for each release target, compares the two arms' fingerprints; only the declared
  interventions (config keys and the files they name) and execution metadata (ports, GPU ids, job ids) may differ;
  declared file hashes must match the producer-time hashes in the manifest.
- **G4 pairing** — builds the expected grid (tasks × seeds × repetitions, identities from a frozen registry if
  given), picks each cell's scored attempt (the first without an infrastructure failure), counts missing and
  failed cells, and aligns the two arms by task.
- **G5 statistics** — bootstrap over tasks (stratified by task type, PCG64 seed in the spec), order-statistic 95%
  interval, one-sided p-values that agree with the interval by construction, Holm across guardrails, degeneracy
  check on the statistic itself.
- **G6 behaviour** — per-episode action categories and outcome flags, candidate minus baseline with paired intervals.
- **G7 verdict** — the decision table below.
- **G8 report** — JSON, Markdown, HTML.

Edges (trigger → what is passed):

- G1 → G2: all logs parsed → arms with episodes and manifests (an arm that fails to load is marked INVALID;
  only targets using it are affected).
- G2 → G4: fingerprints compared → list of undeclared differences (non-empty → target INVALID).
- G4 → G5 and G6: arms paired → task × cell table, missing and failed cells.
- G5 → G7: estimates for the primary metric (two completions of missing cells) and each guardrail.
- G6 → G8 and G7 → G8: behaviour table; verdicts with reasons.

## Verdicts

For each release target (candidate arm vs baseline arm, primary metric where higher is better, margin δ, guardrails
with tolerances τ), in this order:

| step | rule |
|---|---|
| validity | failed fingerprint check, task identity mismatch, duplicate scored attempts, cells outside the grid, > 0.5% missing cells or > 2% cells with an infrastructure failure in an arm → **INVALID** |
| provenance | any arm without a manifest, repetition ids, declared grid or required provenance fields → **DESCRIPTIVE** (statistics shown as evidence, no decision) |
| primary | **BLOCK** if the upper bound (optimistic completion of missing cells) < −δ; **PASS** if the lower bound (pessimistic completion) > −δ; otherwise **INCONCLUSIVE**; degenerate bootstrap → INCONCLUSIVE |
| guardrails | **PASS** if every upper bound < τ; **BLOCK** only for a Holm-confirmed violation; any missing cell → INCONCLUSIVE (shown on completed cells) |
| target | the worst component (BLOCK > INCONCLUSIVE > PASS); any missing cell prevents PASS |

Exit codes: 0 PASS, 1 INCONCLUSIVE, 2 BLOCK, 3 INVALID, 4 DESCRIPTIVE. Without `--target`, the exit code is that of
the target marked `default: true` (exactly one must be). A CI job should pass only on 0.

The error budget (one-sided 0.025 for the primary test, 0.025 for the guardrail family) is nominal. Simulations in
[`calibration.md`](calibration.md): at a true effect of exactly −δ, P(PASS) is 0.018–0.041; at n = 500 and no true
difference, P(PASS) is 0.63–0.89; at a true loss of 2δ, P(BLOCK) is 0.65–0.87.

## Writing a spec

```yaml
name: "WebShop: distilled memory release check"
base_dir: ../..                     # paths are relative to this
seed: 20261116
bootstrap_draws: 20000
arms:
  T_NONE: {logs: ["results/gate_ws/GWS_T_NONE_R*_Q8.episodes.jsonl"],
           grid: {task_range: [0, 500], repetitions: [1, 2], registry: configs/registry_T.jsonl}}
  T_TRUE: {logs: ["results/gate_ws/GWS_T_TRUE_R*_Q8.episodes.jsonl"], grid: {...same...}}
interventions:
  memory_pub:
    config_patch: {memory: expel, retriever: bm25, token_budget: 524, demo_pool: data/pools/pub.jsonl}
    files_sha256: {demo_pool: 89893797...}          # must match the producer-time hash in the manifest
targets:
  C1_ship_pub:
    default: true
    baseline: T_NONE
    candidate: T_TRUE
    interventions: [memory_pub]
    primary: {metric: score, margin: 0.05}
    guardrails:
      - {metric: prompt_tokens, statistic: relative_mean, tolerance: 0.30}
      - {metric: latency_s, statistic: relative_p95, tolerance: 0.30}
contrasts:                          # diagnostics: reported, never part of a verdict
  L1: {candidate: D_ABL, baseline: D_TRUE, role: diagnostic, claim: greater, threshold: 0}
  L2: {candidate: D_ABL, baseline: [D_COMP1, D_COMP2, D_COMP3], pairing: same_block}
claims:                             # joint claims: every listed contrast must hold (intersection-union)
  localization: {all_of: [L1, ...]}
```

`skillrc-gate schema` prints the JSON Schemas of `Episode`, `RunManifest` and the spec.

## Producing release-eligible logs

- **SkillRC runner** (`python -m skillrc.runner --config ...`): writes `<run>.episodes.jsonl` (one line per attempt,
  with repetition, block, attempt, latency, retries, infrastructure error), `<run>.attempts.jsonl` (start records,
  fsynced) and `<run>.manifest.json` (normalized config, producer-time file hashes, append-only launch segments).
  It resumes by cell under the same fingerprint only, reruns infrastructure failures up to `max_infra_reruns`,
  records interrupted attempts, and stops with exit 75 after `max_consecutive_infra` failed attempts in a row so the
  job can restart its servers. Serving details come from `SKILLRC_SERVING_JSON` / `SKILLRC_ENV_JSON`.
- **Inspect AI** (`format: inspect`): one sample = one scored attempt, epochs = repetitions, `error_retries` =
  earlier failed attempts, task arguments flattened to `task_args.<name>`. Provenance must be producer-time: pass
  `eval(metadata={"model_revision": ..., "serving": {...}})` and hash any file argument inside the task
  (`<name>_sha256`). See `examples/inspect_agentdojo/`.
- **Anything else**: write `Episode` JSON lines and a `RunManifest` (`format: episodes`).

## In CI

```yaml
# .github/workflows/release-check.yml (sketch)
- run: pip install -e .
- run: skillrc-gate report gate.yaml --out gate_report     # exit code = verdict
- if: always()
  run: cat gate_report/summary.md >> "$GITHUB_STEP_SUMMARY"
```

## Examples

- `examples/webshop_bundled_release/` — historical logs, DESCRIPTIVE; the bundled-release case.
- `examples/inspect_agentdojo/` — Inspect AI on AgentDojo, Qwen3-8B, a tips block in the system prompt.
- `examples/webshop_release_check/` — the pre-registered release check (`engineering/PREREG_gate_ws_2026-11-16.md`).
  Its logs are published compressed in `data/summaries/gate_ws_2026_11/`; `bash scripts/gate/restore_release_logs.sh`
  puts them where the frozen spec reads them, then `skillrc-gate report examples/webshop_release_check/gate.yaml`.
- Case page with all three: <https://eveneureka.github.io/SkillRC/gate/>.

### Example: a bundled release hides a regression

Qwen3-8B, 500 WebShop test sessions, one execution window (historical SkillRC study):

| target | Δ score [95%] | prompt tokens |
|---|---|---|
| page label + memory vs old release | +0.132 [+0.089, +0.174] | +12.5% |
| page label alone | +0.251 [+0.208, +0.295] | −27.6% |
| memory, given the page label | −0.119 [−0.164, −0.075] | +55.4% |

### Example: the pre-registered release check

Qwen3-8B, fresh runs with full provenance, two execution blocks, 6,500 episodes, no infrastructure failures or
missing cells (`examples/webshop_release_check/report/`):

| target | verdict | Δ score [95%] | prompt tokens |
|---|---|---|---|
| C1 ship the published memory (sessions 0–499) | BLOCK | −0.134 [−0.176, −0.092] | +58.1% (Holm-confirmed over +30%) |
| C2 ship the repaired memory (sessions 520–1019) | INCONCLUSIVE | −0.074 [−0.119, −0.029] | +38.5% [+29.8%, +48.1%] |

The localization claim (replacing six "keep searching" rules beats the published memory and three comparison edits
of other rules) is not met: the repair cut returns to search from 0.84 to 0.26 per episode, but its score gain over
the published memory was +0.036 [−0.003, +0.075], and one comparison edit did better than the repair.

## Limitations

Benchmark harnesses, not production traffic. Intervals cover task sampling given the observed execution blocks, not
future windows. The bootstrap is mildly anti-conservative in some simulated scenarios. Item-level attribution
(which memory rule caused which action) is not automated; localization needs declared ablation arms.

## Design history

The contract was reviewed in eight rounds with an independent reviewer (Codex): `engineering/GATE_DESIGN_v0–v5`,
`engineering/REVIEW_r1–r8`.
