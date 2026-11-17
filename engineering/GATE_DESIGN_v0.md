# `skillrc gate`: a release check for agent memory and harness changes (design v0, 2026-11-16)

Status: draft for review. Nothing here is implemented or run yet. Internal; not committed.

## 0. Purpose and audience

SkillRC is today a research project (public report "A Gain Is Not Proof of Learning", repo github.com/EvenEureka/SkillRC).
The owner is applying for industry ML/agent engineering internships and wants SkillRC to become an engineering case
that an interviewer can understand, run and discuss in ten minutes. The target is a genuinely useful tool, not a
repackaged paper.

The question the tool answers:

> I changed my agent (added or re-distilled a memory, edited the prompt, flipped a harness flag).
> On a fixed task set: is the new version better or worse, by how much, with what confidence, at what cost in
> tokens, steps and latency? If it is worse, which part of the change and which behaviour caused it?
> Should I ship it, block it, or collect more data?

Rules inherited from the project: local open models only for new runs (no OpenAI API), at most 2 GPUs at a time,
pre-register every new run, no "production/deployed" claims, figure examples copied from real data.

## 1. Evidence already in hand (all real, file paths given)

### 1a. Bundled release hides a regression (Qwen3-8B, WebShop 1k-product, 500 test sessions, 15-step budget, same window)

Source: `results/placebo/r2_2026-10-31_results.json` key `P.Q8` (not yet public). Pre-registration
`results/placebo/PREREG_r2_2026-10-31.md`. Per-episode trajectories `results/R2031_P_*_Q8.episodes.jsonl`.
"Page label" = harness flag `webshop_page_label`, prepends `[Current page: ...]` to each observation.
PUB = the published WebShop pool `data/pools/webshop1010_insights.jsonl` (30 rules distilled by gpt-4o-mini from
gpt-4o-mini successes). R1, R2 = two independent Qwen3-32B-distilled replicate pools. CHORAL = length-matched
control (same items, tags, per-item word counts, token count under the reader tokenizer; rule text replaced by
choral-singing sentences).

| arm | mean score | searches / session |
|---|---:|---:|
| NONE (old harness, no memory) | 0.326 | 1.84 |
| PUB_TRUE (old harness + memory) | 0.197 | 2.83 |
| PUB_CHORAL | 0.330 | 1.63 |
| L_NONE (page label, no memory) | 0.577 | 1.41 |
| L_PUB_TRUE (page label + memory) | 0.458 | 2.12 |
| L_PUB_CHORAL | 0.598 | 1.16 |
| L_R1_TRUE / L_R1_CHORAL | 0.285 / 0.570 | 3.02 / 1.17 |
| L_R2_TRUE / L_R2_CHORAL | 0.395 / 0.568 | 2.87 / 1.20 |

Content value with label (TRUE − CHORAL): PUB −0.140 [−0.183, −0.096], R1 −0.284 [−0.330, −0.240],
R2 −0.173 [−0.216, −0.130]. Reading cost (CHORAL − L_NONE) ≈ 0 for all three (|I| ≤ 0.021, CIs include 0).

The "release story": shipping {page label + memory} as one version vs the old baseline = +0.132 (looks like a win);
factorised, page label = +0.251 and memory = −0.119 on top of it.

### 1b. Mechanism evidence so far

- gpt-4o-mini, WebShop (Oct 10, 500 sessions, unlabelled harness): searches/episode 2.4 (none) → 5.7 (true rules);
  successful-purchase rate 0.37 → 0.20; 15-step exhaustion 0.63 → 0.80; score 0.294 → 0.154; coherent irrelevant
  control +0.018 (no reading cost).
- Across readers the rule-specific marker tracks harm: change in searches 3.33 / 1.06 / 0.11 for gpt-4o-mini /
  Qwen3-8B / gpt-4.1-mini vs content value −0.158 / −0.116 / −0.002.
- The pool contains several "keep searching" rules with no stop condition, e.g.
  "[fashion] If the desired product is not found, consider re-searching with the same criteria.",
  "[garden] If the first search does not yield satisfactory results, repeat the search with the same parameters...",
  "[grocery] If initial search results do not yield desired products, refine your search terms and try again.",
  "[beauty] Utilize pagination to explore multiple pages of search results if necessary."
- NOT yet shown: that these rules cause the loss. Cautionary precedent from ALFWorld in the same project: the
  suspected rule ("for cooling, always check the fridge first") moved the behavioural marker, but deleting it did not
  recover the loss (pre-registered ablation, AB1 failed); deleting all five cooling rules recovered only part.
  Behavioural markers locate compliance, not cause.
- Caveat on file: on ALFWorld, an advice-style off-topic control (shopping advice) erased most of the content value
  (R_web +0.011), so "advice-form text" itself may be part of the effect; the WebShop analogue is untested.

### 1c. Other facts the tool design should respect

- Run-to-run noise: identical gpt-4o-mini configs flip 17–20% of ALFWorld episodes and 13–17% of WebShop sessions.
- Pool-to-pool variability: pools distilled by the same pipeline differ about as much as the effect (heterogeneity
  p = 0.0003, τ ≈ 0.019 over 18 ALFWorld pools; two gpt-sourced pools +0.080 vs +0.032).
- Control-topic sensitivity: changing the irrelevant control's topic (gardening → astronomy) shifts the estimated
  reading cost by 0.025.
- Injection must be audited: token counts of the injected block per reader tokenizer (existing audit scripts report
  0 mismatches); WebShop server must be seeded (`--seed 0`, PYTHONHASHSEED=0) or instructions change.
- Latency is NOT recorded today. Per episode we record success, reward, steps, prompt/completion tokens, cost_usd,
  memory_tokens, full trajectory (action + observation per step).

## 2. Existing code to reuse

- Harness: `skillrc/` (ReAct agent, ALFWorld + WebShop envs, OpenAI-compatible LLM client incl. local vLLM,
  memory stores with token budgets, mock env/provider for offline smoke tests).
- Controls: `scripts/analysis/build_irrelevant_control*.py`, `build_order_placebo*.py`, choral/gardening banks in
  `data/pools/`, tokenizer-matched with minimal nonce replacement.
- Audits: `scripts/analysis/audit_*injection*.py`, `skillrc/passport.py` (SHA-256 run passport, paired stats).
- Stats: paired bootstrap stratified by task type in the per-study analysis scripts (`r2_2026-10-31.py` etc.),
  `scripts/analysis/report_paired_results.py`.
- Released per-episode outcomes: `data/outcomes/*.jsonl.gz` (+ `scripts/outcomes.py`).

## 3. Tool design

### 3a. Pipeline (nodes and edges)

Nodes:

- **N1 Spec loader.** Input: a gate spec (YAML) naming the baseline config, the candidate config, the task set
  (ids + seeds), the primary metric, guardrail metrics and tolerances. Output: a normalised spec.
- **N2 Change diff.** Compares baseline and candidate configs field by field and classifies each differing field as
  a factor: `memory` (pool file / retrieval / budget), `prompt`, `harness flag`, `model`, `decoding`. Output: list of
  factors.
- **N3 Arm planner.** From the factor list, builds the arms to run:
  - always: BASE, BASE_DUP (noise floor), CAND;
  - if a `memory` or `prompt` factor adds text: CTRL = candidate with the added text replaced by a length-matched
    control (N4);
  - if ≥2 factors (bundled change, k ≤ 3): the full 2^k factorial cells; k > 3 → refuse and ask for one change at a time;
  - if the change is a memory *pipeline* (distiller/prompt that produces pools) rather than one pool file:
    ≥3 replicate pools, and the estimate is the mean over pools with a pool-resampling interval.
  Output: arm list with configs.
- **N4 Control builder.** For every added text block: same items, tags, per-item word counts, retrieval order, and
  token count under the reader's tokenizer (minimal nonce replacement; refuses if the match fails). Optional second
  topic to report reading cost as a range. Output: control pool + audit JSON.
- **N5 Runner.** Runs every arm on the same task set in one time window (or ingests existing episode JSONL files that
  match the schema). Records per episode: success, score, steps, prompt/completion tokens, memory tokens, per-call
  wall time, trajectory. Output: one episode JSONL per arm + run passport (config hashes, model id, server seed).
- **N6 Injection audit.** Re-renders the prompt for a sample of steps per arm and checks the injected block's token
  count and position. Output: pass/fail; a failing arm is excluded and the verdict cannot be PASS.
- **N7 Paired statistics.** Pairs episodes by (task_id, seed). For each contrast: mean difference and 95% bootstrap
  interval (tasks resampled, stratified by task type; pools resampled when replicate pools exist). Contrasts:
  net T = CAND − BASE; reading cost I = CTRL − BASE; content value R = CAND − CTRL; factorial main effects and
  interactions; guardrails Δtokens/episode, Δsteps, Δp50/p95 latency. Noise floor from BASE vs BASE_DUP: flip rate,
  Δ distribution, and the minimum detectable effect for this n. Output: stats JSON.
- **N8 Behaviour diff.** Maps each action to an env-specific category (WebShop: search, click-product,
  click-option, next/prev page, back-to-search, buy, invalid; ALFWorld: goto, take, put, open, heat/cool/clean,
  examine, invalid) and each episode to an outcome type (success, wrong purchase / wrong object, budget exhausted,
  loop = same action 3×, invalid-action streak). Reports per-arm category counts and outcome-type shares with
  paired intervals, ranked by |Δ|. Then links the top shifted categories to memory items whose text matches a
  category lexicon (whole-word match, lexicon fixed per env before any data). Output: "suspect items" list.
- **N9 Localization run (optional, costs runs).** Two arms: ABL = candidate with suspect items replaced in place by
  length-matched control sentences (keeps length, count and positions); ABL_COMP = candidate with an equal number of
  randomly chosen non-suspect items replaced the same way (seeded). Recovery attributed to the suspect items only
  if ABL − CAND > 0 and ABL − ABL_COMP > 0. Output: localization verdict.
- **N10 Verdict.** Primary metric decision with a pre-declared non-inferiority margin δ:
  - BLOCK if the upper bound of Δprimary < 0, or any guardrail bound exceeds its tolerance, or N6 failed;
  - PASS if the lower bound of Δprimary > −δ and guardrails hold;
  - INCONCLUSIVE otherwise (prints the n needed for the observed effect at the measured noise).
  For bundled changes, one verdict per factor plus the bundle.
- **N11 Report + replay.** Static HTML (GitHub Pages-able) + JSON: verdict card, contrasts table, decomposition bar,
  behaviour-shift table, suspect items with their text, and a side-by-side replay of paired episodes (baseline vs
  candidate on the same task) with the first divergent step and suspect-category actions highlighted. Replay
  examples chosen by a fixed rule (e.g. the first k task ids where BASE succeeds and CAND fails, plus k random
  pairs), never hand-picked.

Edges (trigger → payload):

- N1 → N2: always; normalised spec.
- N2 → N3: always; factor list.
- N3 → N4: when an arm needs a control; the added text block + reader tokenizer.
- N4 → N3: control built and audited; control pool path (a failed match stops the gate).
- N3 → N5: arm list complete; configs.
- N5 → N6: each arm finished (or ingested); episode file + passport.
- N6 → N7: audits done; the list of arms that passed.
- N7 → N8: always; paired episode table.
- N8 → N9: only if Δprimary is negative with its upper bound < 0 and suspect items exist, and the user opts in;
  suspect item ids.
- N9 → N7: localization arms finished; their episode files (re-enter statistics).
- N7, N8, N9 → N10: stats JSON, behaviour diff, localization verdict.
- N10 → N11: verdict + everything above.

### 3b. Interfaces

```
skillrc gate plan   spec.yaml            # prints arms, run count, MDE estimate; no runs
skillrc gate run    spec.yaml            # N3–N6 (runs or resumes arms)
skillrc gate report spec.yaml --out report/   # N7–N11 from episode files (works offline)
skillrc gate replay a.jsonl b.jsonl --task 17 # one paired replay page
```

Episode schema (JSONL, one line per episode): task_id, seed, task_type, success, score, steps, prompt_tokens,
completion_tokens, memory_tokens, latency_s (per episode, plus per-call list), trajectory [{step, action, obs}].
An adapter can convert other harness logs (e.g. Inspect AI eval logs) into this schema.

### 3c. Packaging for a ten-minute visitor

1. Static demo page (no install): the WebShop case from §1a rendered by `gate report` + replays.
2. `docker compose up demo`: seeded WebShop server + harness pointed at any OpenAI-compatible endpoint,
   20-session stratified subset (~10 min). Expected verdict at n = 20: INCONCLUSIVE, with the n needed — shown on
   purpose as the noise-floor lesson.
3. Full reproduction: existing README path (ALFWorld/WebShop setup, vLLM).
4. CI example: GitHub Action that runs `gate report` on committed episode logs and fails on BLOCK.
5. Unit tests: statistics on synthetic data with known effects, control builder token matching, schema validation,
   factorial planner.
6. 2-minute video last.

## 4. New run to complete the case (pre-registration to follow; local Qwen3-8B only)

Purpose: (a) a same-window run with latency recorded, (b) a localization test for the WebShop loss, (c) the
off-topic-advice question. WebShop 1k, test sessions 0–499, seed 0, 15 steps, Qwen3-8B via vLLM, thinking off,
temperature 0, page label ON (the better harness is the realistic baseline). Candidate arms:

| arm | content | question |
|---|---|---|
| L_NONE | no memory | baseline |
| L_NONE_DUP | no memory, second run | noise floor, flip rate |
| L_PUB_TRUE | published pool | replicate r2 effect, with latency |
| L_PUB_CHORAL | length-matched choral control | reading cost vs content |
| L_PUB_ABL | suspect "keep searching / paginate" items replaced in place by choral sentences | do those items cause the loss? |
| L_PUB_ABLCOMP | same number of random non-suspect items replaced | is any k-item removal enough? |
| L_PUB_ADVICE | PUB shape filled with off-topic advice (e.g. code-debugging advice bank from r2) | does advice-form text itself induce over-search? |
| (optional) NONE, PUB_TRUE unlabelled | | full 2×2 (label × memory) in one window with latency |
| (optional) L_PUB_FIX | PUB + one stop-condition sentence, length-compensated | demo of gating a candidate fix |

Suspect items would be chosen by the N8 lexicon (search/re-search/repeat/refine/keywords/pagination/next page),
fixed before the run. Rough cost: ~1–1.5 GPU-h per arm, 7–11 arms, 2 GPUs.

## 5. Known limitations to state up front

Synthetic 1k-product WebShop with generated goals; 15-step budget (the over-search harm may depend on it);
8B–32B open readers plus two small hosted models; one environment where memory clearly hurts (ALFWorld effects are
small and pool-dependent); controls depend on topic; benchmark harness, not a production system.

## 6. Questions for the reviewer

1. Worth building? Compared with existing eval tooling (Inspect AI, promptfoo, LangSmith/Braintrust experiment
   comparison, OpenAI Evals, τ-bench's pass^k, Anthropic's "Adding Error Bars to Evals" recommendations), what is
   genuinely distinctive here and what is redundant? What should be cut, what is missing for it to be a tool an
   agent team would actually use?
2. Statistics and verdict (N7, N10): is the paired, stratified bootstrap with a duplicate-baseline noise floor the
   right design? How should run-to-run variance enter the interval (it is not captured by resampling tasks)? Is the
   non-inferiority PASS/BLOCK/INCONCLUSIVE rule sound, and how to handle multiplicity across guardrails and factorial
   cells?
3. Localization (N8, N9): is in-place replacement + complement ablation a sound causal test? Is the lexicon-based
   suspect selection defensible, and what are its failure modes?
4. The new run (§4): which arms are worth running, which are not, what is missing; power at n = 500 sessions; threats
   (e.g. vLLM nondeterminism, server seeding, choral vs advice control mismatch).
5. Packaging (§3c): what would a strong industry interviewer probe first, and what must exist for the ten-minute
   experience to work?
6. Anything in §1 that is over-claimed for this purpose.

Please give: an overall score (1–10) for "useful, credible engineering tool + case", a verdict, ranked weaknesses
each with the minimum fix, and the arms you would actually run.
