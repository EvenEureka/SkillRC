"""Run configuration: dataclass + YAML/JSON loader.

Every knob the EXPERIMENT_PLAN pins (frozen executor, temp 0, K=20, token budget B,
seeds) is a field here so runs are fully specified by one config file.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:  # pragma: no cover
    _HAS_YAML = False


@dataclass
class RunConfig:
    # --- identity ---
    run_id: str = "run"
    milestone: str = "M0"
    out_dir: str = "results"

    # --- environment ---
    env: str = "mock"                       # mock | alfworld | webshop
    task_ids: Optional[List[Any]] = None    # explicit subset; None => env default
    max_tasks: Optional[int] = None
    max_steps: int = 30
    webshop_base_url: str = "http://127.0.0.1:3000"
    webshop_n_tasks: int = 100

    # --- frozen executor (LLM) ---
    provider: str = "openai"                # openai | mock  (openai == any OpenAI-compatible endpoint)
    model: str = "PLACEHOLDER-choose-a-model"
    temperature: float = 0.0
    max_completion_tokens: int = 256
    base_url: Optional[str] = None          # set for a local vLLM / compatible server
    request_timeout: float = 120.0
    max_retries: int = 5
    api_key: Optional[str] = None           # override OPENAI_API_KEY (e.g. "EMPTY" for local vLLM)
    chat_template_kwargs: Optional[dict] = None  # e.g. {"enable_thinking": false} to turn off Qwen3 thinking

    # --- ReAct scaffold (held constant across systems) ---
    n_fewshot: int = 2
    system_prompt_key: str = "default"
    prompt_style: str = "thought_action"    # thought_action | react_alfworld
    inject_admissible: bool = False          # show env's admissible_commands each step (ALFWorld "valid actions" setting)
    fewshot_file: Optional[str] = None       # .jsonl of {"text": ...}  OR  .json dict (react_alfworld)
    save_trajectories: bool = False          # retain action/observation traces for small audits

    # --- memory / skill store ---
    memory: str = "none"                    # none | random_k | raw_traj | full_context | expel | skillos | evolve | passport
    memory_k: int = 20                      # active-skill cap K
    token_budget: Optional[int] = None      # matched context budget B for injected memory
    demo_pool: Optional[str] = None         # jsonl pool of prior experiences/skills
    rank_reference_pool: Optional[str] = None  # optional index-aligned corpus used only for ranking
    retriever: str = "lexical"              # lexical | bm25 | dense  (B2 retrieval factor)
    embed_model: str = "text-embedding-3-small"  # dense retriever embedding model
    passport_file: Optional[str] = None      # memory=passport: frozen global admission card
    passport_strict_identity: bool = True   # reject executor/payload hash mismatches

    # --- self-evolving curator (memory=evolve): online per-skill utility learning ---
    curator: str = "static"                 # static | online_uniform | online_gate (utility-gated firewall)
    gate_threshold: float = 0.35            # online_gate: drop skills whose posterior-mean utility < this
    explore: bool = True                    # Thompson-sample utility (explore) vs posterior-mean (exploit)
    prior_a: float = 1.0                    # Beta prior alpha (utility successes)
    prior_b: float = 1.0                    # Beta prior beta  (utility failures)
    credit: str = "uniform"                 # uniform (whole retrieved set) | cf_judge (per-skill counterfactual, +API)
    cluster_by_type: bool = True            # maintain a separate utility posterior per task-type cluster
    stream_shuffle_seed: Optional[int] = None  # if set, shuffle task stream order by this seed (online seed axis)
    credit_file: Optional[str] = None       # curator=precomputed_gate: offline debiased-credit json (credit_analysis.py)
    credit_gate_cut: float = 0.0            # precomputed_gate: drop skills with debiased credit < cut

    # --- experiment control ---
    seeds: List[int] = field(default_factory=lambda: [0])

    @staticmethod
    def load(path: str) -> "RunConfig":
        with open(path) as f:
            if path.endswith((".yaml", ".yml")):
                if not _HAS_YAML:
                    raise RuntimeError("pyyaml not installed but a .yaml config was given")
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        known = {f_.name for f_ in RunConfig.__dataclass_fields__.values()}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return RunConfig(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
