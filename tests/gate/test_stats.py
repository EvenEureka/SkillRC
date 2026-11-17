import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from skillrc.gate.stats import Estimate, PairedBootstrap, UndefinedStatistic, holm, order_stat_rank


def test_order_stat_rank_contract_values():
    # B = 20,000, alpha = 0.025 -> 500th and 19,501st order statistics (0-indexed 499 and 19,500)
    assert order_stat_rank(0.025, 20_000) == 499
    m = order_stat_rank(0.025, 20_000)
    assert (1 + m) / 20_001 < 0.025 <= (2 + m) / 20_001


def test_round4_tie_example_interval_and_p_agree():
    draws = np.r_[np.zeros(500), np.ones(19_500)]
    e = Estimate("x", 0.975, draws, alpha=0.025)
    assert e.lower == 0.0                          # 500th order statistic
    assert not e.supports_greater(0.5)
    assert e.p_greater(0.5) == pytest.approx(501 / 20_001)
    assert e.p_greater(0.5) >= 0.025


@settings(max_examples=300, deadline=None)
@given(st.lists(st.integers(-5, 5), min_size=1000, max_size=1000), st.integers(-6, 6), st.booleans())
def test_interval_decision_equals_p_decision_with_ties(vals, t, half):
    draws = np.array(vals, dtype=float) / 2.0
    t = t / 2.0 + (0.25 if half else 0.0)          # thresholds on and between tied draw values
    e = Estimate("x", float(draws.mean()), draws, alpha=0.025, min_distinct=1)
    assert e.supports_greater(t) == (e.p_greater(t) < 0.025)
    assert e.supports_less(t) == (e.p_less(t) < 0.025)


def test_stratified_bootstrap_keeps_stratum_sizes_and_is_deterministic():
    strata = ["a"] * 7 + ["b"] * 3 + ["c"] * 10
    b1 = PairedBootstrap(strata, n_draws=500, seed=3)
    b2 = PairedBootstrap(strata, n_draws=500, seed=3)
    assert np.array_equal(b1.idx, b2.idx)
    s = np.array(strata)
    for row in b1.idx[:50]:
        drawn = s[row]
        assert (drawn == "a").sum() == 7 and (drawn == "b").sum() == 3 and (drawn == "c").sum() == 10


def test_mean_diff_point_estimate_and_degeneracy():
    b = PairedBootstrap(["a"] * 50, n_draws=2000, seed=1)
    e = b.mean_diff(np.full(50, -0.051))
    assert e.estimate == pytest.approx(-0.051) and e.degenerate
    e2 = b.mean_diff(np.linspace(-1, 1, 50))
    assert not e2.degenerate and e2.lower < 0 < e2.upper


def test_relative_mean_zero_baseline_is_undefined():
    b = PairedBootstrap(["a"] * 10, n_draws=1000, seed=1)
    with pytest.raises(UndefinedStatistic):
        b.relative_mean(np.ones(10), np.zeros(10))


def test_relative_constant_offset_is_not_degenerate():
    # round-5 note: constant raw differences do not imply a constant relative-change statistic
    b = PairedBootstrap(["a"] * 40, n_draws=2000, seed=2)
    base = np.linspace(100, 500, 40)
    e = b.relative_mean(base + 10, base)
    assert not e.degenerate


def test_relative_quantile_uses_all_episodes_with_multiplicity():
    rng = np.random.default_rng(0)
    cand = rng.gamma(3, 2, size=(30, 2))
    base = rng.gamma(3, 1.5, size=(30, 2))
    b = PairedBootstrap(["a"] * 30, n_draws=1000, seed=5)
    e = b.relative_quantile(cand, base, 0.95)
    i = 17
    idx = b.idx[i]
    brute = np.quantile(cand[idx].ravel(), 0.95, method="linear") / np.quantile(base[idx].ravel(), 0.95,
                                                                                    method="linear") - 1
    assert np.isclose(e.draws, brute, rtol=0, atol=1e-12).any()
    assert e.estimate == pytest.approx(np.quantile(cand, 0.95) / np.quantile(base, 0.95) - 1)


def test_holm_step_down_and_ties():
    # 0.01 < 0.05/4 = 0.0125 -> reject; 0.02 < 0.05/3 = 0.0167? no -> stop
    assert holm([0.01, 0.02, 0.03, 0.04], 0.05) == [True, False, False, False]
    assert holm([0.001, 0.001], 0.025) == [True, True]
    # round-2 counterexample: four harm tests, smallest p = 0.02 -> not confirmed under Holm at 0.025
    assert holm([0.02, 0.5, 0.6, 0.9], 0.025) == [False, False, False, False]
