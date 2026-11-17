"""End-to-end engine checks on synthetic, release-eligible arms."""
import numpy as np

from skillrc.gate.engine import run_gate
from skillrc.gate.verdict import State

from conftest import arm, episodes, manifest, spec

N = 300
POOL = "data/pools/pub.jsonl"
MEM = {"memory": {"config_patch": {"memory_pool": POOL}, "files_sha256": {"memory_pool": f"sha-{POOL}"}}}


def independent(p, salt):
    u = np.random.default_rng(salt).random((N, 3))
    return lambda i, r: float(u[i, r - 1] < p)


def perturbed(base, salt, frac=0.08):
    """Same as `base` except a symmetric set of flips: no true effect, not degenerate."""
    rng = np.random.default_rng(salt)
    flip = rng.random((N, 3)) < frac
    return lambda i, r: (1.0 - base(i, r)) if flip[i, r - 1] and ((i + r) % 2 == 0) == (base(i, r) == 1.0) else base(i, r)


BASE_SCORE = independent(0.6, 1)
base_tokens = lambda i, r: 1000 + (i % 11) * 20          # noqa: E731
base_lat = lambda i, r: 10.0 + (i % 13) * 0.5 + 0.1 * r   # noqa: E731


def make(name, score, tokens=None, latency=None, reps=(1, 2), cfg=None, skip=()):
    # candidates cost ~10% more, NOT exactly proportionally (a constant ratio would be a degenerate statistic)
    is_base = name == "BASE"
    tk = tokens or (lambda i, r: base_tokens(i, r) + (0 if is_base else 100 + (i % 7) * 5))
    lt = latency or (lambda i, r: base_lat(i, r) + (0.0 if is_base else 1.0 + (i % 5) * 0.3))
    man = manifest(name, **(cfg if cfg is not None else ({} if name == "BASE" else {"memory_pool": POOL})))
    return arm(name, episodes(name, N, score, reps=reps, tokens=tk, latency=lt, skip=skip), N, reps, man)


def two_arms(cand_score, **kw):
    return {"BASE": make("BASE", BASE_SCORE), "CAND": make("CAND", cand_score, **kw)}


TARGET = {"C": {"baseline": "BASE", "candidate": "CAND", "interventions": ["memory"], "default": True,
                "primary": {"metric": "score", "margin": 0.05},
                "guardrails": [{"metric": "prompt_tokens", "statistic": "relative_mean", "tolerance": 0.30},
                               {"metric": "latency_s", "statistic": "relative_p95", "tolerance": 0.30}]}}


def test_clear_regression_blocks():
    r = run_gate(spec(TARGET, ["BASE", "CAND"], MEM), two_arms(independent(0.4, 2)))
    assert r.targets["C"].state == State.BLOCK
    assert r.targets["C"].components[0].state == State.BLOCK


def test_no_change_passes_with_guardrails():
    r = run_gate(spec(TARGET, ["BASE", "CAND"], MEM), two_arms(perturbed(BASE_SCORE, 3)))
    v = r.targets["C"]
    assert v.state == State.PASS, v.reasons


def test_token_cost_violation_blocks():
    r = run_gate(spec(TARGET, ["BASE", "CAND"], MEM),
                 two_arms(perturbed(BASE_SCORE, 3), tokens=lambda i, r: int(base_tokens(i, r) * 1.8) + (i % 7) * 13))
    v = r.targets["C"]
    assert v.state == State.BLOCK
    assert v.components[0].state == State.PASS and v.components[1].state == State.BLOCK


def test_undeclared_config_difference_is_invalid():
    arms = two_arms(perturbed(BASE_SCORE, 3), cfg={"memory_pool": POOL, "max_steps": 30})
    r = run_gate(spec(TARGET, ["BASE", "CAND"], MEM), arms)
    assert r.targets["C"].state == State.INVALID
    assert any("undeclared" in x for x in r.targets["C"].reasons)


def test_file_hash_mismatch_is_invalid():
    bad = {"memory": {"config_patch": {"memory_pool": POOL}, "files_sha256": {"memory_pool": "sha-other"}}}
    r = run_gate(spec(TARGET, ["BASE", "CAND"], bad), two_arms(perturbed(BASE_SCORE, 3)))
    assert r.targets["C"].state == State.INVALID


def test_missing_candidate_cost_cannot_hide_in_guardrails():
    # round-4 counterexample (a): one missing cell -> guardrails INCONCLUSIVE, best verdict INCONCLUSIVE
    r = run_gate(spec(TARGET, ["BASE", "CAND"], MEM), two_arms(perturbed(BASE_SCORE, 3), skip=[(5, 1)]))
    v = r.targets["C"]
    assert v.state == State.INCONCLUSIVE
    assert all(c.state == State.INCONCLUSIVE for c in v.components[1:])


def test_targets_are_independent():
    # round-3: a BLOCK on one target must not veto another target
    arms = two_arms(independent(0.4, 2))
    arms["FIX"] = make("FIX", perturbed(BASE_SCORE, 4))
    targets = {"C": {**TARGET["C"], "default": False}, "FIXED": {**TARGET["C"], "candidate": "FIX", "default": True}}
    r = run_gate(spec(targets, ["BASE", "CAND", "FIX"], MEM), arms)
    assert r.targets["C"].state == State.BLOCK
    assert r.targets["FIXED"].state == State.PASS, r.targets["FIXED"].reasons
    name, v = r.selected(None)
    assert name == "FIXED" and v.exit_code == 0


def test_default_target_must_be_unique_and_explicit():
    r = run_gate(spec({"A": TARGET["C"], "B": TARGET["C"]}, ["BASE", "CAND"], MEM), two_arms(perturbed(BASE_SCORE, 3)))
    assert r.selected(None)[1].state == State.INVALID
    assert r.selected("nope")[1].state == State.INVALID
    lone = run_gate(spec({"A": {**TARGET["C"], "default": False}}, ["BASE", "CAND"], MEM), two_arms(perturbed(BASE_SCORE, 3)))
    assert lone.selected(None)[1].state == State.INVALID          # a sole target still needs default: true


def test_legacy_logs_give_descriptive_not_pass():
    b = arm("BASE", episodes("BASE", N, BASE_SCORE, legacy=True), N, grid=False)
    c = arm("CAND", episodes("CAND", N, perturbed(BASE_SCORE, 3), legacy=True), N, grid=False)
    t = {"C": {**TARGET["C"], "guardrails": []}}
    r = run_gate(spec(t, ["BASE", "CAND"], MEM), {"BASE": b, "CAND": c})
    assert r.targets["C"].state == State.DESCRIPTIVE


def test_joint_claim_needs_every_part():
    arms = {"BASE": make("BASE", independent(0.4, 5)), "CAND": make("CAND", independent(0.6, 6)),
            "CAND2": make("CAND2", perturbed(independent(0.4, 5), 7))}
    contrasts = {"L1": {"candidate": "CAND", "baseline": "BASE", "role": "diagnostic", "claim": "greater"},
                 "L2": {"candidate": "CAND2", "baseline": "BASE", "role": "diagnostic", "claim": "greater"}}
    claims = {"loc": {"all_of": ["L1", "L2"]}}
    r = run_gate(spec(TARGET, ["BASE", "CAND", "CAND2"], MEM, contrasts, claims), arms)
    assert r.contrasts["L1"]["supported"] is True
    assert r.contrasts["L2"]["supported"] is False
    assert r.claims["loc"]["evaluable"] is True and r.claims["loc"]["holds"] is False


def test_report_has_behaviour_power_blocks_and_replays():
    r = run_gate(spec(TARGET, ["BASE", "CAND"], MEM), two_arms(independent(0.4, 2)))
    d = r.target_details["C"]
    assert d["power"]["n_tasks"] == N
    assert d["behaviour"]["actions_per_episode"]
    assert len(d["replays"]) >= 1
    assert set(d["blocks"]["per_block"]) == {"1", "2"}
