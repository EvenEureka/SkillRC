"""Paired, task-clustered, stratified bootstrap (contract v3 §2, amended by v4 §1–2).

Conventions, all fixed here and unit-tested:

* Resampling unit: task. Within each task-type stratum, draw as many tasks as the stratum has, with replacement.
  RNG: NumPy PCG64 via ``default_rng(seed)``. Every statistic of one target uses the same index matrix.
* Interval: order statistics, no interpolation. With ``m`` = the largest integer such that (1 + m) / (B + 1) < alpha,
  L = sorted_draws[m] and U = sorted_draws[B - 1 - m] (0-indexed). For B = 20,000 and alpha = 0.025 these are the
  500th and 19,501st order statistics.
* One-sided p-values: p(Δ > t) = (1 + #{Δ* <= t}) / (B + 1);  p(Δ < t) = (1 + #{Δ* >= t}) / (B + 1).
  "Δ > t is supported" is defined as L > t, which is equivalent to p(Δ > t) < alpha, ties included; likewise U < t.
* Degenerate: fewer than ``min_distinct`` distinct draw values of the statistic itself -> no decision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

QUANTILE_METHOD = "linear"
_ROUND = 12  # decimals used when counting distinct draw values


class UndefinedStatistic(ValueError):
    """Raised when a statistic is undefined on the observed sample (e.g. a zero baseline mean)."""


def order_stat_rank(alpha: float, n_draws: int) -> int:
    """Largest m with (1 + m) / (n_draws + 1) < alpha."""
    m = math.ceil(alpha * (n_draws + 1) - 1) - 1
    if m < 0:
        raise ValueError(f"{n_draws} bootstrap draws are too few for alpha={alpha}")
    return m


@dataclass
class Estimate:
    statistic: str
    estimate: float
    draws: np.ndarray = field(repr=False)
    alpha: float
    min_distinct: int = 20

    def __post_init__(self):
        d = np.sort(np.asarray(self.draws, dtype=float))
        if not np.all(np.isfinite(d)):
            raise UndefinedStatistic(f"{self.statistic}: non-finite bootstrap draws")
        self.draws = d
        m = order_stat_rank(self.alpha, d.size)
        self.lower = float(d[m])
        self.upper = float(d[d.size - 1 - m])
        self.n_distinct = int(np.unique(np.round(d, _ROUND)).size)

    @property
    def degenerate(self) -> bool:
        return self.n_distinct < self.min_distinct

    def p_greater(self, t: float) -> float:
        """One-sided p for the claim Δ > t."""
        return (1 + int(np.count_nonzero(self.draws <= t))) / (self.draws.size + 1)

    def p_less(self, t: float) -> float:
        """One-sided p for the claim Δ < t."""
        return (1 + int(np.count_nonzero(self.draws >= t))) / (self.draws.size + 1)

    def supports_greater(self, t: float) -> bool:
        return self.lower > t

    def supports_less(self, t: float) -> bool:
        return self.upper < t

    def to_dict(self) -> Dict[str, object]:
        return {
            "statistic": self.statistic,
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "level": 1 - 2 * self.alpha,
            "n_draws": int(self.draws.size),
            "n_distinct": self.n_distinct,
            "degenerate": self.degenerate,
        }


class PairedBootstrap:
    """Stratified task-level bootstrap index matrix shared by all statistics of a target."""

    def __init__(self, strata: Sequence[str], n_draws: int = 20_000, seed: int = 20261116,
                 alpha: float = 0.025, min_distinct: int = 20):
        strata = np.asarray([str(s) for s in strata])
        if strata.size == 0:
            raise ValueError("no tasks to resample")
        self.n_tasks = strata.size
        self.n_draws = int(n_draws)
        self.alpha = alpha
        self.min_distinct = min_distinct
        rng = np.random.default_rng(seed)
        idx = np.empty((self.n_draws, self.n_tasks), dtype=np.int64)
        col = 0
        for s in sorted(set(strata.tolist())):          # deterministic stratum order
            members = np.flatnonzero(strata == s)
            k = members.size
            idx[:, col:col + k] = members[rng.integers(0, k, size=(self.n_draws, k))]
            col += k
        self.idx = idx

    def _est(self, name: str, estimate: float, draws: np.ndarray) -> Estimate:
        return Estimate(name, float(estimate), draws, self.alpha, self.min_distinct)

    def mean_diff(self, diff: np.ndarray, name: str = "mean_diff") -> Estimate:
        """Mean over tasks of task-level differences (candidate - baseline)."""
        diff = np.asarray(diff, dtype=float)
        self._check(diff)
        return self._est(name, diff.mean(), diff[self.idx].mean(axis=1))

    def relative_mean(self, cand: np.ndarray, base: np.ndarray, name: str = "relative_mean") -> Estimate:
        """mean(cand) / mean(base) - 1 on task-level values, computed inside each draw."""
        cand = np.asarray(cand, dtype=float)
        base = np.asarray(base, dtype=float)
        self._check(cand), self._check(base)
        if base.mean() == 0:
            raise UndefinedStatistic(f"{name}: baseline mean is zero")
        bd = base[self.idx].mean(axis=1)
        if np.any(bd == 0):
            raise UndefinedStatistic(f"{name}: baseline mean is zero in some bootstrap draws")
        return self._est(name, cand.mean() / base.mean() - 1, cand[self.idx].mean(axis=1) / bd - 1)

    def relative_quantile(self, cand_raw: np.ndarray, base_raw: np.ndarray, q: float,
                          name: Optional[str] = None, chunk: int = 2000) -> Estimate:
        """Q_q(cand episodes) / Q_q(base episodes) - 1.

        ``*_raw`` have shape (n_tasks, k) with NaN for absent episodes. Each drawn task contributes all of its
        episodes, with multiplicity. Quantile method: NumPy "linear".
        """
        name = name or f"relative_p{int(round(q * 100))}"
        cand_raw = np.asarray(cand_raw, dtype=float)
        base_raw = np.asarray(base_raw, dtype=float)
        if cand_raw.shape[0] != self.n_tasks or base_raw.shape[0] != self.n_tasks:
            raise ValueError("raw arrays must have one row per task")
        qb = np.nanquantile(base_raw, q, method=QUANTILE_METHOD)
        if not qb > 0:
            raise UndefinedStatistic(f"{name}: baseline quantile is not positive")
        est = np.nanquantile(cand_raw, q, method=QUANTILE_METHOD) / qb - 1
        out = np.empty(self.n_draws)
        for s in range(0, self.n_draws, chunk):
            ix = self.idx[s:s + chunk]
            c = cand_raw[ix].reshape(ix.shape[0], -1)
            b = base_raw[ix].reshape(ix.shape[0], -1)
            qbd = np.nanquantile(b, q, axis=1, method=QUANTILE_METHOD)
            if np.any(~(qbd > 0)):
                raise UndefinedStatistic(f"{name}: baseline quantile not positive in some draws")
            out[s:s + chunk] = np.nanquantile(c, q, axis=1, method=QUANTILE_METHOD) / qbd - 1
        return self._est(name, est, out)

    def _check(self, v: np.ndarray):
        if v.shape != (self.n_tasks,):
            raise ValueError(f"expected shape ({self.n_tasks},), got {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError("task-level values must be finite (impute or drop before calling)")


def holm(pvalues: Sequence[float], alpha: float) -> List[bool]:
    """Holm step-down rejections at family level alpha, strict p < threshold; ties broken by input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: (pvalues[i], i))
    reject = [False] * m
    for rank, i in enumerate(order):
        if pvalues[i] < alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def mde(paired_sd: float, n: int, alpha_two_sided: float = 0.05, power: float = 0.8) -> float:
    """Normal-approximation minimum detectable effect for a paired mean difference."""
    from statistics import NormalDist

    z = NormalDist()
    return (z.inv_cdf(1 - alpha_two_sided / 2) + z.inv_cdf(power)) * paired_sd / math.sqrt(n)
