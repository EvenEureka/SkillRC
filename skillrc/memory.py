"""Memory / skill stores injected into the ReAct context.

M0: NoMemory (no-mem ReAct baseline).
M1 confound baselines, ALL filled to the SAME token budget B (matched-budget is the
plan's central fairness control) but differing ONLY in *selection*:
  - RandomKMemory     : random experiences, filled to B          (Random-k demos)
  - RawTrajMemory     : retrieval-ranked experiences, filled to B (Retrieval-of-raw-traj)
  - FullContextMemory : pool-order experiences (NO retrieval), filled to B (Full-context)

Because all three target the same B, a skill-library win over them cannot be
attributed to "more context"; the only thing that varies is retrieval vs random vs
none. `retrieve()` records the realized injected-memory token count in
`self.last_tokens` so the runner can audit that budgets are actually matched
(finding [1]/[10] from the code review). The ranker is lexical-overlap here — a
dependency-free stand-in for the plan's retrieval factor (BM25/dense); swap `_rank`
for the ablation, the fill-to-B path is unchanged.
"""
from __future__ import annotations

import json
import hashlib
import random
from typing import Any, Dict, List, Optional

from .tokens import count_tokens, trim_to_budget


class MemoryStore:
    name = "none"

    def __init__(self, *a, **k):
        self.last_tokens = 0

    def retrieve(self, query: str, seed: int = 0) -> str:
        self.last_tokens = 0
        return ""

    def update(self, task_id: Any, trajectory: str, success: bool, reward: float,
               **kwargs) -> None:
        pass

    def reset_state(self) -> None:
        """Online curators reset per-stream state here; no-op for static stores."""
        pass

    def size(self) -> Dict[str, int]:
        return {"n_items": 0, "n_tokens": 0}


class NoMemory(MemoryStore):
    name = "none"


def _load_pool(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    items: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _text_of(item: Dict[str, Any]) -> str:
    return item.get("text") or json.dumps(item, ensure_ascii=False)


def _sha256_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lexical_overlap(query: str, text: str) -> int:
    return len(set(query.lower().split()) & set(text.lower().split()))


class _PoolMemory(MemoryStore):
    """Confound baselines drawing from a fixed pool of prior experiences."""

    def __init__(self, pool_path: Optional[str] = None, k: int = 20,
                 token_budget: Optional[int] = None, model: str = "gpt-4o-mini",
                 retriever: str = "lexical", embed_model: str = "text-embedding-3-small", **_):
        super().__init__()
        self.pool = _load_pool(pool_path)
        self.k = k
        self.token_budget = token_budget
        self.model = model
        self.retriever = retriever
        self.embed_model = embed_model
        self.rank_reference_pool = _load_pool(_.get("rank_reference_pool"))
        if self.rank_reference_pool and len(self.rank_reference_pool) != len(self.pool):
            raise ValueError("rank_reference_pool must be index-aligned with the payload pool")
        self._bm25 = None
        self._pool_norm = None      # cached L2-normalized pool embeddings (dense)

    def _fill(self, items: List[Dict[str, Any]]) -> str:
        """Greedily append items until reaching the token budget B (matched budget).
        Falls back to the K-item cap when no budget is set."""
        texts = [_text_of(it) for it in items]
        if not self.token_budget:
            chosen = texts[:self.k]
        else:
            chosen, used = [], 0
            for t in texts:
                n = count_tokens(t, self.model)
                if chosen and used + n > self.token_budget:
                    break
                chosen.append(t)
                used += n
                if used >= self.token_budget:
                    break
        ctx = "\n\n".join(chosen)
        if self.token_budget:
            ctx = trim_to_budget(ctx, self.token_budget, self.model, keep="head")
        self.last_tokens = count_tokens(ctx, self.model) if ctx else 0
        return ctx

    def _rank(self, query: str) -> List[Dict[str, Any]]:
        """B2 retrieval factor: lexical overlap / BM25 / dense embeddings. All feed the
        SAME fill-to-B path, so only the ranking changes across the ablation."""
        r = self.retriever
        if r == "bm25":
            return self._rank_bm25(query)
        if r == "dense":
            return self._rank_dense(query)
        return sorted(self.pool, key=lambda it: _lexical_overlap(query, _text_of(it)),
                      reverse=True)

    def _rank_bm25(self, query: str) -> List[Dict[str, Any]]:
        from rank_bm25 import BM25Okapi
        if self._bm25 is None:
            ranking_pool = self.rank_reference_pool or self.pool
            self._bm25 = BM25Okapi([_text_of(it).lower().split() for it in ranking_pool])
        scores = self._bm25.get_scores(query.lower().split())
        order = sorted(range(len(self.pool)), key=lambda i: scores[i], reverse=True)
        return [self.pool[i] for i in order]

    def _rank_dense(self, query: str) -> List[Dict[str, Any]]:
        import numpy as np
        from .llm import embed_texts
        if self._pool_norm is None:
            emb = np.array(embed_texts([_text_of(it) for it in self.pool], self.embed_model))
            self._pool_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        q = np.array(embed_texts([query], self.embed_model)[0])
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self._pool_norm @ q
        order = sorted(range(len(self.pool)), key=lambda i: sims[i], reverse=True)
        return [self.pool[i] for i in order]

    def size(self) -> Dict[str, int]:
        return {"n_items": len(self.pool),
                "n_tokens": sum(count_tokens(_text_of(it), self.model) for it in self.pool)}


class RandomKMemory(_PoolMemory):
    name = "random_k"

    def retrieve(self, query: str, seed: int = 0) -> str:
        if not self.pool:
            self.last_tokens = 0
            return ""
        shuffled = list(self.pool)
        random.Random(seed).shuffle(shuffled)
        return self._fill(shuffled)


class RawTrajMemory(_PoolMemory):
    name = "raw_traj"

    def retrieve(self, query: str, seed: int = 0) -> str:
        if not self.pool:
            self.last_tokens = 0
            return ""
        return self._fill(self._rank(query))


class FullContextMemory(_PoolMemory):
    name = "full_context"

    def retrieve(self, query: str, seed: int = 0) -> str:
        # NO retrieval: pool order, filled to the same B. Isolates "does retrieval
        # ordering matter, vs just filling the budget?"
        if not self.pool:
            self.last_tokens = 0
            return ""
        return self._fill(list(self.pool))


class ExpeLMemory(RawTrajMemory):
    """M2 ExpeL reproduction: NL-INSIGHT representation. Identical retrieval (lexical)
    and budget-fill to raw_traj — only the pool content differs (distilled insights vs
    raw trajectories), so any gain isolates the representation factor (plan B2-2)."""
    name = "expel"


class SkillOSMemory(RawTrajMemory):
    """M2 SkillOS (frozen-executor) reproduction: MARKDOWN-SKILL representation. Same
    retrieval + budget as raw_traj; pool holds distilled Markdown skills."""
    name = "skillos"


class PassportMemory(MemoryStore):
    """Frozen global admission decision wrapping an existing payload store."""

    name = "passport"

    def __init__(self, delegate: MemoryStore, card_path: str,
                 executor_identity: Dict[str, Any], payload_identity: Dict[str, Any],
                 pool_path: Optional[str], strict_identity: bool = True):
        super().__init__()
        if not card_path:
            raise ValueError("memory=passport requires passport_file")
        with open(card_path) as handle:
            self.card = json.load(handle)
        if self.card.get("schema_version") != "memory-passport-v1":
            raise ValueError(f"unsupported passport schema: {self.card.get('schema_version')!r}")
        if self.card.get("scope") != "global":
            raise ValueError("this runner currently supports global passports only")
        self.delegate = delegate
        self.card_path = card_path
        self.deploy = bool(self.card.get("decision", {}).get("deploy"))
        expected_executor = self.card.get("executor", {})
        expected_payload = self.card.get("payload", {})
        mismatches = []
        for field, expected in expected_executor.items():
            actual = executor_identity.get(field)
            if expected is not None and expected != actual:
                mismatches.append(f"executor.{field} card={expected!r} run={actual!r}")
        if expected_payload.get("kind") != delegate.name:
            mismatches.append(
                f"payload kind card={expected_payload.get('kind')} run={delegate.name}"
            )
        for field in ("retriever", "token_budget", "memory_k"):
            expected = expected_payload.get(field)
            actual = payload_identity.get(field)
            if expected is not None and expected != actual:
                mismatches.append(f"payload.{field} card={expected!r} run={actual!r}")
        expected_hash = expected_payload.get("sha256")
        if expected_hash and pool_path:
            if _sha256_path(pool_path) != expected_hash:
                mismatches.append("payload SHA-256 does not match passport")
        elif expected_hash and not pool_path:
            mismatches.append("passport expects a hashed payload but demo_pool is unset")
        self.identity_mismatches = mismatches
        if strict_identity and mismatches:
            raise ValueError("passport identity check failed: " + "; ".join(mismatches))

    def retrieve(self, query: str, seed: int = 0) -> str:
        if not self.deploy:
            self.last_tokens = 0
            return ""
        context = self.delegate.retrieve(query, seed=seed)
        self.last_tokens = self.delegate.last_tokens
        return context

    def reset_state(self) -> None:
        self.delegate.reset_state()

    def size(self) -> Dict[str, int]:
        return {**self.delegate.size(), "deployed": int(self.deploy)}

    def state_summary(self) -> Dict[str, Any]:
        return {
            "passport_file": self.card_path,
            "status": self.card["decision"]["status"],
            "deploy": self.deploy,
            "limiting_lower_bound": self.card["decision"]["limiting_lower_bound"],
            "identity_mismatches": self.identity_mismatches,
        }


_REGISTRY = {
    "none": NoMemory,
    "random_k": RandomKMemory,
    "raw_traj": RawTrajMemory,
    "full_context": FullContextMemory,
    "expel": ExpeLMemory,
    "skillos": SkillOSMemory,
}


def build_memory(cfg) -> MemoryStore:
    kind = (cfg.memory or "none").lower()
    if kind == "evolve":
        from .selfevolve import SelfEvolvingMemory
        return SelfEvolvingMemory(
            pool_path=cfg.demo_pool, k=cfg.memory_k, token_budget=cfg.token_budget,
            model=cfg.model, retriever=getattr(cfg, "retriever", "dense"),
            embed_model=getattr(cfg, "embed_model", "text-embedding-3-small"),
            curator=getattr(cfg, "curator", "static"),
            gate_threshold=getattr(cfg, "gate_threshold", 0.35),
            explore=getattr(cfg, "explore", True),
            prior_a=getattr(cfg, "prior_a", 1.0), prior_b=getattr(cfg, "prior_b", 1.0),
            credit=getattr(cfg, "credit", "uniform"),
            cluster_by_type=getattr(cfg, "cluster_by_type", True),
            credit_file=getattr(cfg, "credit_file", None),
            credit_gate_cut=getattr(cfg, "credit_gate_cut", 0.0))
    if kind == "passport":
        if not getattr(cfg, "passport_file", None):
            raise ValueError("memory=passport requires passport_file")
        with open(cfg.passport_file) as handle:
            card = json.load(handle)
        payload_kind = str(card.get("payload", {}).get("kind", "")).lower()
        if payload_kind not in _REGISTRY or payload_kind == "none":
            raise ValueError(f"passport payload kind must be one of {sorted(_REGISTRY)} except none")
        delegate = _REGISTRY[payload_kind](
            pool_path=cfg.demo_pool, k=cfg.memory_k,
            token_budget=cfg.token_budget, model=cfg.model,
            retriever=getattr(cfg, "retriever", "lexical"),
            embed_model=getattr(cfg, "embed_model", "text-embedding-3-small"),
            rank_reference_pool=getattr(cfg, "rank_reference_pool", None),
        )
        return PassportMemory(
            delegate=delegate, card_path=cfg.passport_file,
            executor_identity={
                field: getattr(cfg, field, None)
                for field in (
                    "provider", "model", "prompt_style", "n_fewshot", "system_prompt_key",
                    "fewshot_file", "chat_template_kwargs", "inject_admissible", "max_steps",
                    "max_completion_tokens",
                )
            } | {"fewshot_sha256": _sha256_path(getattr(cfg, "fewshot_file", None))},
            payload_identity={
                "retriever": getattr(cfg, "retriever", None),
                "token_budget": getattr(cfg, "token_budget", None),
                "memory_k": getattr(cfg, "memory_k", None),
            },
            pool_path=cfg.demo_pool,
            strict_identity=getattr(cfg, "passport_strict_identity", True),
        )
    if kind not in _REGISTRY:
        raise ValueError(
            f"unknown memory kind {kind!r}; have {sorted(_REGISTRY)} + evolve + passport"
        )
    if kind == "none":
        return NoMemory()
    return _REGISTRY[kind](pool_path=cfg.demo_pool, k=cfg.memory_k,
                           token_budget=cfg.token_budget, model=cfg.model,
                           retriever=getattr(cfg, "retriever", "lexical"),
                           embed_model=getattr(cfg, "embed_model", "text-embedding-3-small"),
                           rank_reference_pool=getattr(cfg, "rank_reference_pool", None))
