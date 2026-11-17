"""Release-verdict decision table (contract v3 §1, amended by v4 §2–4 and the round-5 notes).

Pure functions over already-computed evidence, so the table can be tested exhaustively.

Override order: validity -> provenance (DESCRIPTIVE) -> missingness -> degeneracy -> statistics.

States and exit codes: PASS 0, INCONCLUSIVE 1, BLOCK 2, INVALID 3, DESCRIPTIVE 4. Precedence among components:
BLOCK > INCONCLUSIVE > PASS. INVALID overrides everything. DESCRIPTIVE sits outside the ordering and is never
PASS.

Primary metric (higher is better), margin delta, evaluated on two completions of missing cells that share the
bootstrap draws (pessimistic: candidate-missing = 0, baseline-missing = 1; optimistic: the reverse):
    BLOCK         if U_opt  < -delta
    PASS          if L_pess > -delta
    INCONCLUSIVE  otherwise, or if either completion is degenerate.
Guardrails (increasing cost, tolerance tau each), only on complete data:
    PASS          if U < tau (strict)
    BLOCK         if Holm-confirmed violation: p(Δ > tau) < alpha after Holm across the target's guardrails
    INCONCLUSIVE  otherwise, if degenerate, or if any cell of the target is missing.
Target: the worst component; any missing cell prevents PASS at target level (also with zero guardrails).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .stats import Estimate, holm


class State(str, Enum):
    PASS = "PASS"
    INCONCLUSIVE = "INCONCLUSIVE"
    BLOCK = "BLOCK"
    INVALID = "INVALID"
    DESCRIPTIVE = "DESCRIPTIVE"


EXIT_CODE = {State.PASS: 0, State.INCONCLUSIVE: 1, State.BLOCK: 2, State.INVALID: 3, State.DESCRIPTIVE: 4}
_RANK = {State.PASS: 0, State.INCONCLUSIVE: 1, State.BLOCK: 2}


def worst(states: List[State]) -> State:
    return max(states, key=lambda s: _RANK[s]) if states else State.PASS


@dataclass
class Component:
    name: str
    kind: str                      # "primary" | "guardrail"
    state: State
    reason: str
    estimate: Optional[dict] = None
    threshold: Optional[float] = None
    p_value: Optional[float] = None
    p_adjusted_reject: Optional[bool] = None

    def to_dict(self) -> dict:
        return {k: (v.value if isinstance(v, State) else v) for k, v in self.__dict__.items()}


@dataclass
class TargetVerdict:
    target: str
    state: State
    reasons: List[str] = field(default_factory=list)
    components: List[Component] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return EXIT_CODE[self.state]

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "state": self.state.value,
            "exit_code": self.exit_code,
            "reasons": self.reasons,
            "flags": self.flags,
            "components": [c.to_dict() for c in self.components],
        }


@dataclass
class PrimaryEvidence:
    metric: str
    margin: float
    pessimistic: Estimate
    optimistic: Estimate


@dataclass
class GuardrailEvidence:
    name: str
    tolerance: float
    estimate: Optional[Estimate]          # None if it could not be computed on complete data
    note: str = ""


def primary_component(ev: PrimaryEvidence) -> Component:
    """Public for the margin-sensitivity table."""
    return _primary(ev)


def _primary(ev: PrimaryEvidence) -> Component:
    p, o, d = ev.pessimistic, ev.optimistic, ev.margin
    est = {"pessimistic": p.to_dict(), "optimistic": o.to_dict()}
    if p.degenerate or o.degenerate:
        return Component(ev.metric, "primary", State.INCONCLUSIVE,
                         "degenerate bootstrap distribution", est, -d)
    if o.supports_less(-d):
        return Component(ev.metric, "primary", State.BLOCK,
                         f"upper bound {o.upper:+.4f} < -margin {-d:+.4f}", est, -d, o.p_less(-d))
    if p.supports_greater(-d):
        return Component(ev.metric, "primary", State.PASS,
                         f"lower bound {p.lower:+.4f} > -margin {-d:+.4f}", est, -d, p.p_greater(-d))
    return Component(ev.metric, "primary", State.INCONCLUSIVE,
                     f"interval [{p.lower:+.4f}, {o.upper:+.4f}] straddles -margin {-d:+.4f}", est, -d)


def _guardrails(evs: List[GuardrailEvidence], any_missing: bool, alpha: float) -> List[Component]:
    comps: List[Optional[Component]] = [None] * len(evs)
    pvals = []
    for i, g in enumerate(evs):
        est = g.estimate.to_dict() if g.estimate is not None else None
        if any_missing:
            comps[i] = Component(g.name, "guardrail", State.INCONCLUSIVE,
                                 "missing cells: guardrails computed on completed attempts only", est, g.tolerance)
        elif g.estimate is None:
            comps[i] = Component(g.name, "guardrail", State.INCONCLUSIVE,
                                 g.note or "not computable", None, g.tolerance)
        elif g.estimate.degenerate:
            comps[i] = Component(g.name, "guardrail", State.INCONCLUSIVE,
                                 "degenerate bootstrap distribution", est, g.tolerance)
        pvals.append(g.estimate.p_greater(g.tolerance)
                     if comps[i] is None else 1.0)       # overridden components cannot reject
    rejects = holm(pvals, alpha) if evs else []
    for i, g in enumerate(evs):
        if comps[i] is not None:
            continue
        e = g.estimate
        if rejects[i]:
            comps[i] = Component(g.name, "guardrail", State.BLOCK,
                                 f"confirmed violation (Holm): estimate {e.estimate:+.4f}, tolerance {g.tolerance:+.4f}",
                                 e.to_dict(), g.tolerance, pvals[i], True)
        elif e.supports_less(g.tolerance):
            comps[i] = Component(g.name, "guardrail", State.PASS,
                                 f"upper bound {e.upper:+.4f} < tolerance {g.tolerance:+.4f}",
                                 e.to_dict(), g.tolerance, pvals[i], False)
        else:
            comps[i] = Component(g.name, "guardrail", State.INCONCLUSIVE,
                                 f"upper bound {e.upper:+.4f} not below tolerance {g.tolerance:+.4f}; violation not confirmed",
                                 e.to_dict(), g.tolerance, pvals[i], False)
    return comps  # type: ignore[return-value]


def decide(target: str,
           primary: Optional[PrimaryEvidence],
           guardrails: List[GuardrailEvidence],
           alpha: float = 0.025,
           invalid_reasons: Optional[List[str]] = None,
           descriptive_reasons: Optional[List[str]] = None,
           any_missing: bool = False) -> TargetVerdict:
    invalid_reasons = list(invalid_reasons or [])
    descriptive_reasons = list(descriptive_reasons or [])
    if invalid_reasons or primary is None:
        return TargetVerdict(target, State.INVALID, invalid_reasons or ["primary metric could not be computed"])

    comps = [_primary(primary)] + _guardrails(guardrails, any_missing, alpha)
    state = worst([c.state for c in comps])
    reasons = [f"{c.kind} {c.name}: {c.state.value} ({c.reason})" for c in comps]
    if any_missing and state == State.PASS:
        state = State.INCONCLUSIVE
        reasons.append("missing cells prevent PASS at target level")
    flags = []
    if comps[0].state == State.PASS and primary.optimistic.upper < 0:   # a regression must hold under both completions
        flags.append("primary metric: detectable but tolerated regression (non-inferior within the margin, upper "
                     "bound below 0)")
    if descriptive_reasons:
        # Evidence only: component states are kept in JSON for inspection, but the text never states a verdict.
        evidence = [f"evidence, {c.kind} {c.name}: {c.reason}" for c in comps]
        return TargetVerdict(target, State.DESCRIPTIVE,
                             ["not a release decision: " + "; ".join(dict.fromkeys(descriptive_reasons))] + evidence,
                             comps, [])
    return TargetVerdict(target, state, reasons, comps, flags)
