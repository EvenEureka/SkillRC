"""Self-evolving skill curator — the METHOD contribution the static baselines lack.

The M1/M2 skill libraries (skillos/expel) in `memory.py` are STATIC pools: their
`update()` is a no-op, so nothing self-evolves. `SelfEvolvingMemory` instead learns,
online over the task stream, a per-skill *utility* posterior (Beta-Bernoulli) from
episode outcomes and uses it at retrieval time. Three curators, sharing one pool /
retriever / token-budget so any difference isolates the learning mechanism:

  - static        : ignore utility -> reproduces skillos (retrieval-only). SANITY.
  - online_uniform: re-rank the topically-relevant shortlist by learned utility
                    (Thompson-sampled), i.e. exploit high-utility + explore uncertain.
  - online_gate   : a learned "skill firewall" — DROP skills whose posterior-mean
                    utility falls below `gate_threshold`, keep the rest in relevance
                    order. Directly tests "can the curator learn to suppress the
                    skills that HURT?" (real data shows injected skills can drop
                    success below no-mem: full_context 0.54 < no-mem 0.62).

Credit assignment (which retrieved skill caused the outcome):
  - uniform : whole retrieved set shares the episode's success/failure (training-free,
              1 Beta update per used skill). Noisy but O(1).
  - cf_judge: per-skill counterfactual credit stub (LLM-judge marginal attribution);
              wired but off by default to keep pilots cheap.

State is per-stream: `reset_state()` re-inits posteriors so each seed/order pass is
an independent online run. `event_log` records (step, task_type, retrieved_ids,
success) for the longitudinal / degradation curves.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .memory import _PoolMemory, _text_of
from .tokens import count_tokens, trim_to_budget

# keyword -> task cluster (for per-cluster utility; mirrors alfworld_env task types)
_CLUSTER_KEYWORDS = [
    ("puttwo", ("two", "both")),
    ("examine", ("examine", "look at", "desklamp", "under the")),
    ("heat", ("hot", "heat", "microwave")),
    ("cool", ("cool", "cold", "fridge")),
    ("clean", ("clean", "wash", "rinse")),
    ("put", ("put", "place", "move")),
]


def _infer_cluster(query: str) -> str:
    q = (query or "").lower()
    # look only near the stated task line to avoid room-description false hits
    line = q
    if "your task is to:" in q:
        line = q.split("your task is to:", 1)[1][:120]
    for cluster, kws in _CLUSTER_KEYWORDS:
        if any(k in line for k in kws):
            return cluster
    return "put"


class SelfEvolvingMemory(_PoolMemory):
    name = "evolve"

    def __init__(self, pool_path: Optional[str] = None, k: int = 20,
                 token_budget: Optional[int] = None, model: str = "gpt-4o-mini",
                 retriever: str = "dense", embed_model: str = "text-embedding-3-small",
                 curator: str = "static", gate_threshold: float = 0.35,
                 explore: bool = True, prior_a: float = 1.0, prior_b: float = 1.0,
                 credit: str = "uniform", cluster_by_type: bool = True,
                 shortlist_mult: int = 3, credit_file: Optional[str] = None,
                 credit_gate_cut: float = 0.0, **_):
        super().__init__(pool_path=pool_path, k=k, token_budget=token_budget,
                         model=model, retriever=retriever, embed_model=embed_model)
        self.curator = curator
        self.gate_threshold = gate_threshold
        # precomputed_gate: fixed firewall from an offline debiased-credit file
        # (scripts/credit_analysis.py output). Drops skills with debiased credit < cut.
        self.credit_gate_cut = credit_gate_cut
        self._precomp = None
        self._precomp_by_cluster = None
        if credit_file:
            import json as _json
            blob = _json.load(open(credit_file))
            dc = blob.get("debiased_credit", {})
            self._precomp = {int(k): float(v) for k, v in dc.items()}
            bc = blob.get("by_cluster")
            if bc:
                self._precomp_by_cluster = {c: {int(k): float(v) for k, v in d.items()}
                                            for c, d in bc.items()}
        self.explore = explore
        self.prior_a, self.prior_b = prior_a, prior_b
        self.credit = credit
        self.cluster_by_type = cluster_by_type
        self.shortlist_mult = shortlist_mult
        # stable id per pool item
        for i, it in enumerate(self.pool):
            it.setdefault("_id", i)
            it.setdefault("task_type", it.get("task_type", "unknown"))
        self.last_retrieved_ids: List[int] = []
        self.last_cluster: str = "_global"
        self._step = 0
        # do-operation probe: when set, retrieve() injects EXACTLY these ids (bypassing
        # ranking/gating), padded with a neutral filler to `force_pad_tokens` so an
        # ablation S\{s} is TOKEN-MATCHED to S and isolates skill CONTENT, not length.
        self.force_ids: Optional[List[int]] = None
        self.force_pad_tokens: Optional[int] = None
        self.reset_state()

    _NEUTRAL_FILLER = ("## Note\nThink step by step and act carefully. "
                       "Re-read the current observation before choosing an action. ")

    def _by_id(self, sid: int) -> Optional[Dict[str, Any]]:
        for it in self.pool:
            if it.get("_id") == sid:
                return it
        return None

    # ---- utility posterior bookkeeping ----
    def reset_state(self) -> None:
        self.post: Dict[Tuple[str, int], List[float]] = {}
        self.event_log: List[Dict[str, Any]] = []
        self._step = 0
        self.last_retrieved_ids = []

    def _key(self, cluster: str, sid: int) -> Tuple[str, int]:
        return (cluster if self.cluster_by_type else "_global", sid)

    def _ab(self, cluster: str, sid: int) -> List[float]:
        k = self._key(cluster, sid)
        if k not in self.post:
            self.post[k] = [self.prior_a, self.prior_b]
        return self.post[k]

    def _util(self, cluster: str, sid: int, rng: random.Random) -> float:
        a, b = self._ab(cluster, sid)
        if self.explore:
            return rng.betavariate(a, b)          # Thompson sample
        return a / (a + b)                         # posterior mean

    def util_mean(self, cluster: str, sid: int) -> float:
        a, b = self._ab(cluster, sid)
        return a / (a + b)

    # ---- retrieval (utility-aware) ----
    def _fill_ids(self, items: List[Dict[str, Any]]) -> Tuple[str, List[int]]:
        """Greedy fill to token budget B (matched-budget), returning text + chosen ids."""
        chosen_txt, chosen_ids, used = [], [], 0
        for it in items:
            t = _text_of(it)
            n = count_tokens(t, self.model)
            if self.token_budget and chosen_txt and used + n > self.token_budget:
                break
            chosen_txt.append(t)
            chosen_ids.append(it["_id"])
            used += n
            if self.token_budget and used >= self.token_budget:
                break
            if not self.token_budget and len(chosen_txt) >= self.k:
                break
        ctx = "\n\n".join(chosen_txt)
        if self.token_budget:
            ctx = trim_to_budget(ctx, self.token_budget, self.model, keep="head")
        self.last_tokens = count_tokens(ctx, self.model) if ctx else 0
        return ctx, chosen_ids

    def retrieve(self, query: str, seed: int = 0) -> str:
        if not self.pool:
            self.last_tokens, self.last_retrieved_ids = 0, []
            return ""
        if self.force_ids is not None:
            # do-operation probe: inject exactly force_ids, token-matched via neutral pad
            items = [self._by_id(s) for s in self.force_ids]
            items = [it for it in items if it is not None]
            ctx, ids = self._fill_ids(items)
            if self.force_pad_tokens:
                cur = count_tokens(ctx, self.model)
                need = self.force_pad_tokens - cur
                if need > 0:
                    reps = max(1, need // max(1, count_tokens(self._NEUTRAL_FILLER, self.model)) + 1)
                    pad = trim_to_budget(self._NEUTRAL_FILLER * reps, need, self.model, keep="head")
                    ctx = ctx + "\n\n" + pad
            self.last_tokens = count_tokens(ctx, self.model) if ctx else 0
            self.last_retrieved_ids = ids
            self.last_cluster = _infer_cluster(query) if self.cluster_by_type else "_global"
            return ctx
        cluster = _infer_cluster(query) if self.cluster_by_type else "_global"
        self.last_cluster = cluster
        rng = random.Random((seed << 20) ^ (self._step * 2654435761) & 0xFFFFFFFF)
        ranked = self._rank(query)                 # topical relevance order (dense/bm25/lexical)

        if self.curator == "static":
            ordered = ranked
        elif self.curator == "precomputed_gate":
            # fixed firewall: drop skills whose OFFLINE debiased credit < cut
            pc = self._precomp or {}
            kept = [it for it in ranked
                    if pc.get(it["_id"], 0.0) >= self.credit_gate_cut]
            ordered = kept if kept else ranked[:1]
        elif self.curator == "mondrian_gate":
            # per-cluster CAUSAL gate: credit file maps cluster -> {skill: AAE};
            # drop a skill only in clusters where its measured causal credit < cut,
            # keep it (default 0.0) where unprobed. Fixes the global gate's
            # cluster-mismatch failure (T1 co-fire non-identifiability).
            pc = (self._precomp_by_cluster or {}).get(cluster, {})
            kept = [it for it in ranked
                    if pc.get(it["_id"], 0.0) >= self.credit_gate_cut]
            ordered = kept if kept else ranked[:1]
        elif self.curator == "online_gate":
            # learned firewall: keep relevance order, drop learned-harmful skills
            kept = [it for it in ranked
                    if self.util_mean(cluster, it["_id"]) >= self.gate_threshold]
            ordered = kept if kept else ranked[:1]  # never inject empty if pool non-trivial
        elif self.curator == "online_uniform":
            # re-rank a topically-relevant shortlist by learned utility (explore+exploit)
            M = max(self.k * self.shortlist_mult, self.k)
            shortlist = ranked[:M]
            ordered = sorted(shortlist,
                             key=lambda it: self._util(cluster, it["_id"], rng),
                             reverse=True)
        else:
            raise ValueError(f"unknown curator {self.curator!r}")

        ctx, ids = self._fill_ids(ordered)
        self.last_retrieved_ids = ids
        return ctx

    # ---- online learning from the just-finished episode ----
    def update(self, task_id: Any, trajectory: Any, success: bool, reward: float,
               task_type: Optional[str] = None, **_) -> None:
        self._step += 1
        cluster = task_type if (self.cluster_by_type and task_type) else self.last_cluster
        ids = list(self.last_retrieved_ids)
        self.event_log.append({"step": self._step, "task_type": task_type,
                               "cluster": cluster, "n_retrieved": len(ids),
                               "retrieved_ids": ids, "success": bool(success)})
        if self.curator == "static" or not ids:
            return
        if self.credit == "uniform":
            for sid in ids:
                ab = self._ab(cluster, sid)
                ab[0 if success else 1] += 1.0
        elif self.credit == "cf_judge":
            # per-skill counterfactual credit (LLM-judge). Stub: falls back to uniform
            # unless a judge is wired. Kept cheap for pilots.
            for sid in ids:
                ab = self._ab(cluster, sid)
                ab[0 if success else 1] += 1.0

    def state_summary(self) -> Dict[str, Any]:
        util = {f"{c}:{s}": round(self.util_mean(c, s), 3)
                for (c, s) in self.post}
        return {"curator": self.curator, "credit": self.credit,
                "n_posteriors": len(self.post), "n_events": len(self.event_log),
                "utilities": util}
