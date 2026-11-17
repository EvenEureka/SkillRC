# skillrc-gate

**A release check for agent changes, computed from episode logs.** You changed your agent: added or re-distilled a
memory, edited the prompt, flipped a harness flag. `skillrc-gate` reads the logs of the old and new versions on the
same tasks and answers, for each change you declare: ship it (**PASS**), don't (**BLOCK**), not enough evidence
(**INCONCLUSIVE**), the evidence itself is broken (**INVALID**), or the logs cannot support a decision
(**DESCRIPTIVE**). Every verdict links to the contrast behind it and to paired trajectories.

It does not run agents. It reads logs from the SkillRC runner, from [Inspect AI](https://inspect.aisi.org.uk/), or
from a plain JSONL format, and writes `result.json`, `summary.md` (for a CI comment) and a self-contained
`report.html`. The CLI exit code is the verdict, so a CI job can block a release.

## Why another eval tool

Experiment dashboards (LangSmith, Braintrust) and eval runners (Inspect, promptfoo) compare runs well. What this
tool adds, and what the SkillRC study needed:

- **Bundled changes are split.** A release that adds a page label and a memory looked +0.13 better than the old
  version; split, the label was +0.25 and the memory −0.12 ([example](#example-a-bundled-release-hides-a-regression)).
  Each declared intervention gets its own target and verdict, and targets never veto one another.
- **Only declared differences are allowed.** Each arm's manifest has a fingerprint: config, every file hash, model
  revision, scorer, and the scientific part of serving and environment. Two arms may differ only in the
  interventions the spec declares; anything else is INVALID. Missing provenance is DESCRIPTIVE, never PASS.
- **Uncertainty is honest about what it covers.** Paired, task-clustered, stratified bootstrap; repetitions averaged
  within task; per-block contrasts shown; the report says the intervals do not cover future execution windows.
- **Missing data cannot manufacture a PASS.** Infrastructure failures are rerun and logged; no exclusions; missing
  cells bound the primary metric and make guardrails INCONCLUSIVE.
- **From verdict to trajectory.** Behaviour shifts (action categories, outcome types) and paired replays chosen by a
  fixed rule, not by hand.

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
