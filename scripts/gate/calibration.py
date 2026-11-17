#!/usr/bin/env python3
"""Calibration of the gate's primary-metric decision by simulation (contract v4 §3, §6).

For each scenario: simulate paired task-level outcomes with a known true effect, run the gate's bootstrap and decision
table, and record (a) coverage of the 95% interval for the true mean difference, (b) P(PASS), P(BLOCK),
P(INCONCLUSIVE). The nominal targets: coverage 0.95; P(PASS) <= 0.025 when the true effect equals -delta;
P(BLOCK) <= 0.025 when the true effect equals -delta. These are nominal; the table shows where they hold.

Bootstrap draws per replicate: 2,000 (the gate uses 20,000; the decision rule is the same order-statistic rule).
Writes docs/gate/calibration.json and docs/gate/calibration.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from skillrc.gate.stats import PairedBootstrap  # noqa: E402
from skillrc.gate.verdict import GuardrailEvidence, PrimaryEvidence, State, decide  # noqa: E402

DELTA = 0.05
B = 2000
D_STRATA = {"fashion": 405, "garden": 67, "electronics": 14, "beauty": 7, "grocery": 7}


def strata_for(n, kind):
    if kind == "single":
        return np.array(["all"] * n)
    sizes = np.array(list(D_STRATA.values()), dtype=float)
    counts = np.maximum(1, np.round(sizes / sizes.sum() * n)).astype(int)
    counts[0] += n - counts.sum()
    return np.repeat(list(D_STRATA), counts)


def sim_binary(rng, n, theta, reps=2):
    """Bernoulli scores with task heterogeneity: base success prob 0.15 + 0.7 * Beta(2, 2) (mean 0.5), candidate
    prob = base + theta. No clipping for theta in [-0.15, 0.15], so the true mean difference is exactly theta."""
    base_p = 0.15 + 0.7 * rng.beta(2, 2, n)
    cand_p = base_p + theta
    assert cand_p.min() >= 0 and cand_p.max() <= 1
    b = (rng.random((n, reps)) < base_p[:, None]).mean(1)
    c = (rng.random((n, reps)) < cand_p[:, None]).mean(1)
    return c - b, float((cand_p - base_p).mean())


def sim_webshop(rng, n, theta, reps=2):
    """WebShop-like partial scores: 0 with prob 0.45, else a reward in {0.33, 0.5, 0.67, 0.75, 1} weighted to 1."""
    levels = np.array([0.0, 0.33, 0.5, 0.67, 0.75, 1.0])
    w = np.array([0.45, 0.05, 0.08, 0.07, 0.05, 0.30])
    ability = rng.normal(0, 1, n)

    def draw(shift):
        logits = np.log(w)[None, None, :] + (ability[:, None, None] + shift) * levels[None, None, :] * 2
        pr = np.exp(logits - logits.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        u = rng.random((n, reps, 1))
        idx = (pr.cumsum(-1) < u).sum(-1)
        return levels[np.minimum(idx, len(levels) - 1)].mean(1), (pr * levels).sum(-1).mean(1)

    b, eb = draw(0.0)
    # find the logit shift that gives the requested true mean difference (bisection on the expected scores)
    lo, hi = -3.0, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2
        logits = np.log(w)[None, :] + (ability[:, None] + mid) * levels[None, :] * 2
        pr = np.exp(logits - logits.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        logits0 = np.log(w)[None, :] + ability[:, None] * levels[None, :] * 2
        pr0 = np.exp(logits0 - logits0.max(-1, keepdims=True))
        pr0 /= pr0.sum(-1, keepdims=True)
        gap = float(((pr - pr0) * levels).sum(-1).mean())
        lo, hi = (mid, hi) if gap < theta else (lo, mid)
    c, ec = draw((lo + hi) / 2)
    return c - b, float((ec - eb).mean())


def sim_rare(rng, n, theta=None):
    d = np.where(rng.random(n) < 0.995, -0.051, 0.949)
    return d, -0.051 * 0.995 + 0.949 * 0.005


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 4), round(c + h, 4))


def run(name, gen, n, strata_kind, n_rep, seed):
    rng = np.random.default_rng(seed)
    st = strata_for(n, strata_kind)
    cover = 0
    states = {s.value: 0 for s in (State.PASS, State.BLOCK, State.INCONCLUSIVE)}
    truths = []
    for r in range(n_rep):
        d, truth = gen(rng, n)
        truths.append(truth)
        boot = PairedBootstrap(st, n_draws=B, seed=seed + r + 1)
        e = boot.mean_diff(d)
        cover += e.lower <= truth <= e.upper
        v = decide("sim", PrimaryEvidence("score", DELTA, e, e), [])
        states[v.state.value] += 1
    return {"scenario": name, "n": n, "strata": strata_kind, "replicates": n_rep,
            "true_effect_mean": round(float(np.mean(truths)), 4), "true_effect_sd": round(float(np.std(truths)), 4),
            "coverage": cover / n_rep, "coverage_mc95": wilson(cover, n_rep),
            **{f"p_{k.lower()}": v / n_rep for k, v in states.items()},
            "p_pass_mc95": wilson(states["PASS"], n_rep), "p_block_mc95": wilson(states["BLOCK"], n_rep)}


def run_guardrail(name, true_rel, n, n_rep, seed, draws=20_000, tol=0.30):
    """Relative-mean token guardrail at the production draw count; primary fixed at PASS."""
    rng = np.random.default_rng(seed)
    st = strata_for(n, "D-like")
    comp = {s.value: 0 for s in (State.PASS, State.BLOCK, State.INCONCLUSIVE)}
    cover = 0
    for r in range(n_rep):
        base = rng.lognormal(np.log(30_000), 0.5, n)
        cand = base * (1 + true_rel) * rng.lognormal(-0.02, 0.2, n) / np.exp(-0.02 + 0.02)
        truth = (1 + true_rel) * np.exp(-0.02 + 0.02) - 1   # lognormal(-0.02, 0.2) has mean exp(0) = 1
        boot = PairedBootstrap(st, n_draws=draws, seed=seed + r + 1)
        e = boot.relative_mean(cand, base)
        cover += e.lower <= truth <= e.upper
        pe = boot.mean_diff(np.linspace(-0.01, 0.01, n))
        v = decide("sim", PrimaryEvidence("score", DELTA, pe, pe), [GuardrailEvidence("tokens", tol, e)])
        comp[v.components[1].state.value] += 1
    return {"scenario": name, "n": n, "strata": "D-like", "replicates": n_rep, "bootstrap_draws": draws,
            "true_relative_change": true_rel, "tolerance": tol, "coverage": cover / n_rep,
            "coverage_mc95": wilson(cover, n_rep),
            **{f"p_guardrail_{k.lower()}": v / n_rep for k, v in comp.items()},
            "p_guardrail_block_mc95": wilson(comp["BLOCK"], n_rep), "p_guardrail_pass_mc95": wilson(comp["PASS"], n_rep)}


def main():
    t0 = time.time()
    rows = []
    sc = []
    for n in (50, 200, 500):
        for strata_kind in ("single", "D-like"):
            for theta, label in ((0.0, "A/A"), (-DELTA, "boundary -delta"), (-2 * DELTA, "regression -2delta")):
                sc.append((f"binary (mean 0.5), {label}", lambda rng, n, t=theta: sim_binary(rng, n, t), n, strata_kind))
                sc.append((f"webshop-like, {label}", lambda rng, n, t=theta: sim_webshop(rng, n, t), n, strata_kind))
        sc.append(("rare-event mixture (round-4 counterexample)", lambda rng, n: sim_rare(rng, n), n, "single"))
    for i, (name, gen, n, sk) in enumerate(sc):
        reps = 600 if n == 500 else 800
        rows.append(run(name, gen, n, sk, reps, 1000 * (i + 1)))
        print(json.dumps(rows[-1]), flush=True)
    grows = []
    for k, (label, tr) in enumerate((("token guardrail, true change = tolerance (+30%)", 0.30),
                                     ("token guardrail, true change +20%", 0.20),
                                     ("token guardrail, no change", 0.0))):
        grows.append(run_guardrail(label, tr, 500, 300, 900_000 + 1000 * k))
        print(json.dumps(grows[-1]), flush=True)
    out = ROOT / "docs/gate"
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps({"delta": DELTA, "bootstrap_draws": B, "rows": rows,
                                                      "guardrail_rows": grows,
                                                      "seconds": round(time.time() - t0)}, indent=1) + "\n")
    L = ["# Calibration of the primary-metric decision", "",
         f"Simulated paired outcomes with a known true effect; margin δ = {DELTA}; {B} bootstrap draws per replicate "
         "(the gate uses 20,000 with the same order-statistic rule); one-sided α = 0.025. "
         "Nominal targets: coverage 0.95; P(PASS) ≤ 0.025 at true Δ = −δ; P(BLOCK) ≤ 0.025 at true Δ = −δ.", "",
         "Brackets: Monte Carlo 95% Wilson intervals over the replicates.", "",
         "| scenario | n | strata | true Δ (mean) | coverage | P(PASS) | P(BLOCK) | P(INCONCL.) |",
         "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['scenario']} | {r['n']} | {r['strata']} | {r['true_effect_mean']:+.3f} | {r['coverage']:.3f} | "
                 f"{r['p_pass']:.3f} [{r['p_pass_mc95'][0]:.3f}, {r['p_pass_mc95'][1]:.3f}] | "
                 f"{r['p_block']:.3f} [{r['p_block_mc95'][0]:.3f}, {r['p_block_mc95'][1]:.3f}] | {r['p_inconclusive']:.3f} |")
    L += ["", "## Token guardrail (relative mean, tolerance +30%, 20,000 draws, n = 500, D-like strata)", "",
          "| scenario | coverage | P(guardrail PASS) | P(guardrail BLOCK) | P(INCONCL.) |", "|---|---:|---:|---:|---:|"]
    for r in grows:
        L.append(f"| {r['scenario']} | {r['coverage']:.3f} | {r['p_guardrail_pass']:.3f} "
                 f"[{r['p_guardrail_pass_mc95'][0]:.3f}, {r['p_guardrail_pass_mc95'][1]:.3f}] | {r['p_guardrail_block']:.3f} "
                 f"[{r['p_guardrail_block_mc95'][0]:.3f}, {r['p_guardrail_block_mc95'][1]:.3f}] | {r['p_guardrail_inconclusive']:.3f} |")
    L += ["", "Reading: the nominal one-sided level is 0.025. Where a Monte Carlo interval lies above it, the bootstrap",
          "decision is anti-conservative for that scenario; the gate's error budget is nominal, not guaranteed.",
          "The rare-event mixture shows poor interval coverage; the degeneracy rule turns those samples into",
          "INCONCLUSIVE rather than wrong decisions.", "", "Generated by `scripts/gate/calibration.py`."]
    (out / "calibration.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
