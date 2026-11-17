"""Exhaustive checks of the decision table (verdict.py)."""
import numpy as np
import pytest

from skillrc.gate.stats import Estimate
from skillrc.gate.verdict import EXIT_CODE, GuardrailEvidence, PrimaryEvidence, State, decide

A = 0.025


def est(center: float, half_width: float, n: int = 20_000, name: str = "x") -> Estimate:
    """Estimate whose draws are uniform on [center - hw, center + hw] (so L/U ~ center -/+ 0.95 hw)."""
    return Estimate(name, center, np.linspace(center - half_width, center + half_width, n), alpha=A)


def prim(center, hw, margin=0.05, opt_shift=0.0):
    p = est(center, hw)
    o = est(center + opt_shift, hw)
    return PrimaryEvidence("score", margin, p, o)


def guard(center, hw, tol=0.30, name="tokens:relative_mean"):
    return GuardrailEvidence(name, tol, est(center, hw, name=name))


def test_exit_codes():
    assert EXIT_CODE == {State.PASS: 0, State.INCONCLUSIVE: 1, State.BLOCK: 2, State.INVALID: 3,
                         State.DESCRIPTIVE: 4}


@pytest.mark.parametrize("center,hw,expected", [
    (0.00, 0.02, State.PASS),            # [-0.019, 0.019] > -0.05
    (-0.12, 0.04, State.BLOCK),          # upper -0.082 < -0.05
    (-0.05, 0.04, State.INCONCLUSIVE),   # straddles -0.05
    (-0.03, 0.03, State.INCONCLUSIVE),   # lower -0.0585 < -0.05
])
def test_primary_states(center, hw, expected):
    v = decide("t", prim(center, hw), [])
    assert v.state == expected


def test_pass_and_block_cannot_both_hold():
    # round-1 counterexample: delta = 0.02, interval [-0.015, -0.005] -> PASS + tolerated flag, never BLOCK
    v = decide("t", prim(-0.01, 0.0053, margin=0.02), [])
    assert v.state == State.PASS
    assert any("tolerated" in f for f in v.flags)


def test_tolerated_flag_only_when_upper_below_zero():
    assert decide("t", prim(0.0, 0.02), []).flags == []


def test_guardrail_pass_requires_strict_upper_below_tolerance():
    ok = decide("t", prim(0.0, 0.02), [guard(0.10, 0.05)])
    assert ok.state == State.PASS
    # upper bound exactly at tolerance -> not PASS (round-5: strict U < tau)
    g = GuardrailEvidence("g", 0.30, Estimate("g", 0.2, np.r_[np.linspace(0.1, 0.30, 19_501), np.full(499, 0.31)],
                                              alpha=A))
    assert g.estimate.upper == pytest.approx(0.30)
    assert decide("t", prim(0.0, 0.02), [g]).state == State.INCONCLUSIVE


def test_guardrail_block_needs_holm_confirmed_violation():
    strong = guard(0.60, 0.10)                          # lower ~0.505 > 0.30: p tiny
    v = decide("t", prim(0.0, 0.02), [strong])
    assert v.state == State.BLOCK
    # round-2 counterexample: one guardrail with p ~ 0.02 among four -> not confirmed by Holm -> INCONCLUSIVE
    draws = np.r_[np.full(400, 0.29), np.linspace(0.31, 0.6, 19_600)]   # p(Δ > 0.30) = 401/20001 ~ 0.020
    weak = GuardrailEvidence("weak", 0.30, Estimate("weak", 0.45, draws, alpha=A))
    assert weak.estimate.p_greater(0.30) == pytest.approx(401 / 20_001)
    assert weak.estimate.supports_greater(0.30)          # unadjusted would say "violation"
    others = [guard(0.0, 0.05, name=f"g{i}") for i in range(3)]
    v4 = decide("t", prim(0.0, 0.02), [weak] + others)
    assert v4.state == State.INCONCLUSIVE
    v1 = decide("t", prim(0.0, 0.02), [weak])           # alone, Holm = unadjusted -> confirmed
    assert v1.state == State.BLOCK


def test_missing_cells_prevent_pass_even_without_guardrails():
    v = decide("t", prim(0.0, 0.02), [], any_missing=True)
    assert v.state == State.INCONCLUSIVE
    assert any("missing cells prevent PASS" in r for r in v.reasons)


def test_missing_cells_make_guardrails_inconclusive_but_primary_block_stands():
    v = decide("t", prim(-0.15, 0.03, opt_shift=0.01), [guard(0.0, 0.02)], any_missing=True)
    assert v.state == State.BLOCK
    assert v.components[1].state == State.INCONCLUSIVE


def test_bounding_uses_pessimistic_for_pass_and_optimistic_for_block():
    # pessimistic lower below -delta, optimistic upper above -delta -> INCONCLUSIVE
    v = decide("t", PrimaryEvidence("score", 0.05, est(-0.06, 0.02), est(0.0, 0.02)), [], any_missing=True)
    assert v.state == State.INCONCLUSIVE


def test_degenerate_primary_is_inconclusive_not_block():
    # round-4 counterexample: every sampled difference = -0.051, delta = 0.05
    e = Estimate("x", -0.051, np.full(20_000, -0.051), alpha=A)
    v = decide("t", PrimaryEvidence("score", 0.05, e, e), [])
    assert v.state == State.INCONCLUSIVE
    assert "degenerate" in v.components[0].reason


def test_degenerate_guardrail_cannot_block():
    e = Estimate("g", 0.9, np.full(20_000, 0.9), alpha=A)
    v = decide("t", prim(0.0, 0.02), [GuardrailEvidence("g", 0.30, e)])
    assert v.state == State.INCONCLUSIVE


def test_invalid_overrides_everything_and_descriptive_is_never_pass():
    assert decide("t", prim(0.0, 0.02), [], invalid_reasons=["x"]).state == State.INVALID
    assert decide("t", None, []).state == State.INVALID
    d = decide("t", prim(0.0, 0.02), [], descriptive_reasons=["legacy"])
    assert d.state == State.DESCRIPTIVE and d.exit_code == 4
    d2 = decide("t", prim(-0.2, 0.02), [], descriptive_reasons=["legacy"])
    assert d2.state == State.DESCRIPTIVE
