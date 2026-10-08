# Provenance of this release

This repository packages the research code behind the SkillRC manuscript.

**From the original research code (unchanged except for paths):** the `skillrc`
package modules, the environment adapters, all YAML configurations, the analysis,
placebo, and audit scripts, the SGE job scripts, the focal 33-insight pool, and the
passport tests. Cluster-specific absolute paths were replaced by environment
variables (`${SKILLRC_ROOT}`, `${SKILLRC_PY}`, `${HF_HOME}`, ...), and pool paths
moved from `results/pools/` to `data/pools/`.

**Re-implemented while packaging** (missing from the archived code; written from
their call sites and covered by the unit tests):

- `skillrc/envs/base.py` — the `Observation` / `StepResult` / `BaseEnv` interface;
- `skillrc/__init__.py`, `skillrc/envs/__init__.py`;
- the `extra_body` pass-through in `skillrc/llm.py`, used to send
  `chat_template_kwargs` (Qwen3 thinking off) to vLLM;
- `configs/mock_smoke.yaml` and the pool-memory case in `tests/test_smoke_mock.py`;
- the optional `--episodes` path in `scripts/analysis/build_order_placebo.py`.

**Regenerated:** `data/pools/expel_insights_order_placebo.jsonl`, by the deterministic
builder; its audit (645 → 673 → 645 tokens, 28/472 nonce substitutions, 94.07% of
body words retained, 645/645 tokens under the GPT-4o-mini and Qwen3-8B tokenizers)
matches the values reported in the paper, and its items match the paper's examples.

**Run summaries:** `data/summaries/*.summary.json` are the original summaries of the
four nominal-seed runs (GPT-4o-mini and Qwen3-8B, with and without the 645-token
memory) behind the per-type and cost tables of the paper appendix.

**Not included:** per-episode logs, hosted-model generations, the raw-trajectory and
Markdown pools, and intermediate result tables. The paper figures are drawn by
`paper/figures/make_figures.py` from the frozen summary statistics reported in the
paper, not re-estimated from episodes.

**Case-study images (added after the runs):** the environment images in the appendix
case studies were made for the paper and are not run outputs. The ALFWorld scenes are
AI2-THOR 2.1.0 renderings of ALFRED trials from the benchmark data
(`scripts/figures/render_cases.py`); for tasks 0–2 the rendered trial is one whose goal
text matches the run log, and for the clean, cool, and examine families it is an example
task of the family, not the representative task. The WebShop images are screenshots of
the official WebShop app on evaluation session 0; the search and clicks were chosen by us
and the result ranking comes from a BM25 stand-in for the Lucene index
(`scripts/figures/serve_webshop.py`, `webshop_walk.py`). No model was run for either.
