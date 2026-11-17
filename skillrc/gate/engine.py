"""Orchestration: spec -> arms -> paired tables -> evidence -> verdicts, diagnostics, behaviour, replays.

Nodes (design v1 §3, v3/v4 contract):
  G1 ingest (ingest/*)  -> G2 spec + intervention check (check_interventions) -> G4 pairing (pairing.py)
  -> G5 statistics (stats.py, here) -> G6 behaviour (behaviour.py) -> G7 verdict (verdict.py) -> G8 report (report/).
G3 (injection audit) runs inside the producer: a release-eligible manifest must carry the audit record; the check
here is that the declared files match the manifest's sha256 values.
"""
from __future__ import annotations

import glob
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import __version__, ingest
from .behaviour import behaviour_shift
from .pairing import ArmData, PairedTable, metric_value
from .schema import fingerprint, fingerprint_diff
from .spec import Contrast, GateSpec, Guardrail, Target
from .stats import Estimate, PairedBootstrap, UndefinedStatistic, mde
from .verdict import GuardrailEvidence, PrimaryEvidence, State, TargetVerdict, decide, primary_component

BOUNDS = {"score": (0.0, 1.0), "success": (0.0, 1.0)}
UNCERTAINTY_NOTE = ("Intervals describe task-sampling uncertainty conditional on the observed execution blocks. "
                    "They do not quantify variability across future execution windows. Two blocks are insufficient "
                    "to estimate that component reliably.")
ERROR_BUDGET_NOTE = ("Nominal error budget per target: one-sided 0.025 for the primary test and 0.025 for the "
                     "guardrail family (Holm). This is not a finite-sample guarantee; percentile-bootstrap error "
                     "rates can exceed nominal levels, notably for rare-event mixtures and small strata.")


def _expand(base: Path, pattern: str) -> List[Path]:
    p = pattern if Path(pattern).is_absolute() else str(base / pattern)
    hits = sorted(glob.glob(p))
    if not hits:
        raise FileNotFoundError(f"no log file matches {p}")
    return [Path(h) for h in hits]


def load_arms(spec: GateSpec) -> Dict[str, ArmData]:
    arms: Dict[str, ArmData] = {}
    base = spec.root
    for name, a in spec.arms.items():
        try:
            paths = [p for pat in a.logs for p in _expand(base, pat)]
            man = (base / a.manifest) if a.manifest and not Path(a.manifest).is_absolute() else (
                Path(a.manifest) if a.manifest else None)
            eps, manifest = ingest.load(a.format, paths, name, man, a.options)
            arms[name] = ArmData(name, eps, manifest, a.grid, base, spec.max_missing_frac, spec.max_infra_frac)
        except Exception as e:  # noqa: BLE001 - a broken arm invalidates only the targets that use it
            arms[name] = ArmData(name, [], None, None, base, spec.max_missing_frac, spec.max_infra_frac)
            arms[name].invalid.append(f"arm {name} failed to load: {type(e).__name__}: {e}")
    return arms


def check_interventions(spec: GateSpec, target: Target, cand: ArmData, base: ArmData) -> List[str]:
    """G2: the arms may differ only in the declared interventions (config keys and the files they name).
    Everything else in the fingerprint - other config keys, every other file hash, model, scorer, the scientific
    part of serving and env - must be identical."""
    if cand.manifest is None or base.manifest is None:
        return []
    declared: Dict[str, Any] = {}
    files: Dict[str, Optional[str]] = {}
    for iname in target.interventions:
        iv = spec.interventions[iname]
        declared.update(iv.config_patch)
        files.update(iv.files_sha256)
    vol = set(spec.volatile_config_keys)
    ign_cfg = vol | set(declared)
    ign_files = set(files) | set(declared)
    fc = fingerprint(cand.manifest, ign_cfg, ign_files)
    fb = fingerprint(base.manifest, ign_cfg, ign_files)
    out = []
    undeclared = fingerprint_diff(fc, fb)
    if undeclared:
        out.append(f"undeclared differences between {cand.name} and {base.name}: {undeclared}")
    cc, bc = cand.manifest.config, base.manifest.config
    for k, v in declared.items():
        if k in ("model", "model_revision"):
            continue
        if cc.get(k) == bc.get(k):
            out.append(f"declared intervention key {k} is identical in both arms")
        if v is not None and cc.get(k) != v:
            out.append(f"candidate {k}={cc.get(k)!r}, intervention declares {v!r}")
    for k, sha in files.items():
        got = cand.manifest.files_sha256.get(k)
        if got is None:
            out.append(f"candidate manifest has no producer-time hash for {k}")
        elif sha is not None and got != sha:
            out.append(f"file for {k}: manifest sha256 {got} != declared {sha}")
    return out


def _guardrail_estimate(boot: PairedBootstrap, pair: PairedTable, g: Guardrail,
                        rows: Optional[np.ndarray] = None) -> Estimate:
    """Guardrail statistic on all tasks, or on the task rows given (completed-cell display)."""
    _require_recorded(pair, g.metric)
    sel = (lambda a: a) if rows is None else (lambda a: a[rows])  # noqa: E731
    if g.statistic == "relative_mean":
        return boot.relative_mean(sel(pair.task_values("candidate", g.metric)),
                                  sel(pair.task_values("baseline", g.metric)), name=g.name)
    q = {"relative_p50": 0.5, "relative_p95": 0.95}[g.statistic]
    return boot.relative_quantile(sel(pair.raw("candidate", g.metric)), sel(pair.raw("baseline", g.metric)), q,
                                  name=g.name)


def _complete_rows(pair: PairedTable) -> np.ndarray:
    """Tasks whose cells are all present in both arms."""
    miss = {c[0] for arm in (pair.candidate, pair.baseline) for c in arm.missing}
    return np.array([u not in miss for u in pair.tasks])


def _require_recorded(pair: PairedTable, metric: str):
    tasks = set(pair.tasks)
    for arm in (pair.candidate, pair.baseline):
        for ep in arm.scored.values():
            if ep.task_uid in tasks and not np.isfinite(metric_value(ep, metric)):
                raise ValueError(f"{metric} is not recorded for every scored attempt")


def _power_table(d: np.ndarray, margin: float, alpha: float) -> Dict[str, Any]:
    n = d.size
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    z = NormalDist().inv_cdf(1 - alpha)
    rows = []
    for theta in (margin / 2, 0.0, -margin / 2, -margin, -2 * margin):
        if se > 0:
            p_pass = NormalDist().cdf((theta + margin) / se - z)
            p_block = NormalDist().cdf((-margin - theta) / se - z)
        else:
            p_pass = p_block = float("nan")
        rows.append({"true_effect": theta, "p_pass": p_pass, "p_block": p_block})
    return {"n_tasks": n, "paired_sd": sd, "se": se,
            "mde_80_two_sided_0.05": mde(sd, n) if n > 1 else float("nan"),
            "rows": rows,
            "note": "normal approximation for one execution of this design at the observed paired SD; "
                    "true effect is assumed, sidedness one-sided at alpha for PASS/BLOCK"}


def _subset_mean_diff(pair: PairedTable, d: np.ndarray, spec: GateSpec, name: str) -> Optional[dict]:
    mask = np.isfinite(d)
    if mask.sum() < 2:
        return None
    boot = PairedBootstrap(pair.strata[mask], spec.bootstrap_draws, spec.seed, spec.alpha, spec.min_distinct)
    e = boot.mean_diff(d[mask], name=name)
    return {**e.to_dict(), "n_tasks": int(mask.sum())}


def _nanmean_rows(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=1)


def _per_block(pair: PairedTable, metric: str, spec: GateSpec) -> Dict[str, Any]:
    rc, rb = pair.raw("candidate", metric), pair.raw("baseline", metric)
    bc, bb = pair.blocks("candidate"), pair.blocks("baseline")
    blocks = sorted(set(bc[bc > 0].tolist()) & set(bb[bb > 0].tolist()))
    out: Dict[str, Any] = {"blocks": blocks, "per_block": {}}
    per_task = {}
    for k in blocks:
        ck = np.where(bc == k, rc, np.nan)
        bk = np.where(bb == k, rb, np.nan)
        d = _nanmean_rows(ck) - _nanmean_rows(bk)
        per_task[k] = d
        out["per_block"][str(k)] = _subset_mean_diff(pair, d, spec, f"delta_block{k}")
    if len(blocks) >= 2:
        k1, k2 = blocks[0], blocks[1]
        out["block_difference"] = _subset_mean_diff(pair, per_task[k2] - per_task[k1], spec,
                                                    f"delta_block{k2}-delta_block{k1}")
    out["note"] = "comparable observations only: tasks with both arms observed in that block"
    return out


def _task_order_key(uid: str):
    tid = uid.split(":")[1] if uid.count(":") >= 2 else uid
    return (0, int(tid), uid) if tid.isdigit() else (1, 0, uid)


def _replays(pair: PairedTable, metric: str, spec: GateSpec) -> List[Dict[str, Any]]:
    tv_c, tv_b = pair.task_values("candidate", metric), pair.task_values("baseline", metric)
    order = sorted(range(pair.n), key=lambda i: _task_order_key(pair.tasks[i]))
    worse = [i for i in order if np.isfinite(tv_b[i]) and np.isfinite(tv_c[i]) and tv_b[i] > tv_c[i]]
    chosen = worse[:spec.report.replays_worse]
    rest = [i for i in order if i not in chosen]
    rng = np.random.default_rng(spec.report.replay_seed)
    k = min(spec.report.replays_random, len(rest))
    rnd = sorted(rng.choice(len(rest), size=k, replace=False).tolist()) if k else []
    out = []
    for why, idxs in (("baseline scored higher (first by task id)", chosen),
                      ("random pair (fixed seed)", [rest[j] for j in rnd])):
        for i in idxs:
            ec, eb = pair.episode("candidate", i), pair.episode("baseline", i)
            if ec is None or eb is None:
                continue
            ac = [s.action for s in ec.trajectory or []]
            ab = [s.action for s in eb.trajectory or []]
            div = next((j for j in range(min(len(ac), len(ab))) if ac[j] != ab[j]), min(len(ac), len(ab)))
            out.append({
                "why": why, "task_uid": pair.tasks[i], "task_id": ec.task_id, "task_type": ec.task_type,
                "first_divergent_step": div + 1,
                "candidate": _ep_view(ec), "baseline": _ep_view(eb),
            })
    return out


def _ep_view(ep) -> Dict[str, Any]:
    return {"run_id": ep.run_id, "repetition": ep.repetition, "score": ep.score, "steps": ep.steps,
            "prompt_tokens": ep.prompt_tokens, "latency_s": ep.latency_s,
            "trajectory": [{"step": s.step, "action": s.action, "obs": s.obs[:1500]} for s in ep.trajectory or []]}


def evaluate_target(spec: GateSpec, name: str, target: Target,
                    arms: Dict[str, ArmData]) -> Tuple[TargetVerdict, Dict[str, Any]]:
    c, b = arms[target.candidate], arms[target.baseline]
    pair = PairedTable(c, b)
    invalid = list(dict.fromkeys(c.invalid + b.invalid + pair.invalid + check_interventions(spec, target, c, b)))
    descriptive = list(dict.fromkeys(c.descriptive + b.descriptive))
    any_missing = pair.any_missing()
    details: Dict[str, Any] = {"candidate": c.name, "baseline": b.name, "n_tasks": pair.n,
                               "strata": {s: int((pair.strata == s).sum()) for s in sorted(set(pair.strata))},
                               "pairing": target.pairing, "notes": list(pair.notes), "any_missing": any_missing,
                               "interventions": target.interventions}
    if pair.n == 0:
        invalid.append("no shared tasks")
    m = target.primary.metric
    if not invalid and any_missing and (target.pairing == "same_block" or m not in BOUNDS):
        invalid.append("missing cells cannot be bounded for this primary metric / pairing")
    if invalid:
        return decide(name, None, [], spec.alpha, invalid), details
    boot = PairedBootstrap(pair.strata, spec.bootstrap_draws, spec.seed, spec.alpha, spec.min_distinct)
    try:
        if any_missing:
            lo, hi = BOUNDS[m]
            pess = boot.mean_diff(pair.diff(m, "average", fill_cand=lo, fill_base=hi), "primary_pessimistic")
            opt = boot.mean_diff(pair.diff(m, "average", fill_cand=hi, fill_base=lo), "primary_optimistic")
            d_obs = pair.diff(m, "average")
        else:
            d_obs = pair.diff(m, target.pairing)
            if not np.all(np.isfinite(d_obs)):
                raise ValueError(f"{m}: some tasks have no comparable pair")
            pess = opt = boot.mean_diff(d_obs, "primary")
    except (ValueError, UndefinedStatistic) as e:
        return decide(name, None, [], spec.alpha, [f"primary: {e}"]), details
    guards: List[GuardrailEvidence] = []
    rows = _complete_rows(pair) if any_missing else None
    sub = (PairedBootstrap(pair.strata[rows], spec.bootstrap_draws, spec.seed, spec.alpha, spec.min_distinct)
           if any_missing and rows.sum() > 1 else None)
    for g in target.guardrails:
        try:
            if any_missing:
                est = _guardrail_estimate(sub, pair, g, rows) if sub is not None else None
                note = "completed attempts only (missing cells): shown, not used for the verdict"
            else:
                est, note = _guardrail_estimate(boot, pair, g), ""
            guards.append(GuardrailEvidence(g.name, g.tolerance, est, note))
        except UndefinedStatistic as e:
            invalid.append(f"guardrail {g.name}: {e}")
        except ValueError as e:
            if descriptive:
                guards.append(GuardrailEvidence(g.name, g.tolerance, None, str(e)))
            else:
                invalid.append(f"guardrail {g.name}: {e}")
    if invalid:
        return decide(name, None, [], spec.alpha, invalid), details
    verdict = decide(name, PrimaryEvidence(m, target.primary.margin, pess, opt), guards, spec.alpha,
                     [], descriptive, any_missing)
    fin = d_obs[np.isfinite(d_obs)]
    margins = sorted({0.0, 0.02, 0.05, float(target.primary.margin)})
    sensitivity = [{"margin": d, "primary_component": primary_component(PrimaryEvidence(m, d, pess, opt)).state.value}
                   for d in margins]
    details.update({
        "primary": {"metric": m, "margin": target.primary.margin, "margin_sensitivity": sensitivity,
                    "candidate_mean": float(np.nanmean(pair.task_values("candidate", m))),
                    "baseline_mean": float(np.nanmean(pair.task_values("baseline", m))),
                    "pessimistic": pess.to_dict(), "optimistic": opt.to_dict()},
        "guardrails": [{"name": g.name, "tolerance": g.tolerance,
                        "estimate": g.estimate.to_dict() if g.estimate else None, "note": g.note} for g in guards],
        "power": _power_table(fin, target.primary.margin, spec.alpha),
        "blocks": _per_block(pair, m, spec),
        "behaviour": behaviour_shift(pair, boot),
        "replays": _replays(pair, m, spec),
    })
    return verdict, details


def evaluate_contrast(spec: GateSpec, name: str, con: Contrast, arms: Dict[str, ArmData]) -> Dict[str, Any]:
    """Diagnostic / descriptive contrast. With a list of baselines, the per-task estimate is the mean of the
    candidate-minus-baseline differences over the listed arms (each computed with the contrast's pairing)."""
    c = arms[con.candidate]
    bases = [con.baseline] if isinstance(con.baseline, str) else list(con.baseline)
    pairs = [PairedTable(c, arms[b]) for b in bases]
    invalid = list(dict.fromkeys(c.invalid + [r for b in bases for r in arms[b].invalid] +
                                 [r for p in pairs for r in p.invalid]))
    descriptive = list(dict.fromkeys(c.descriptive + [r for b in bases for r in arms[b].descriptive]))
    out: Dict[str, Any] = {"candidate": c.name, "baseline": con.baseline, "metric": con.metric, "role": con.role,
                           "claim": con.claim, "threshold": con.threshold, "pairing": con.pairing,
                           "description": con.description, "notes": [n for p in pairs for n in p.notes]}
    if not invalid and len({tuple(p.tasks) for p in pairs}) > 1:
        invalid.append("the listed baselines cover different tasks")
    if invalid or pairs[0].n == 0:
        out.update({"evaluable": False, "reasons": invalid or ["no shared tasks"]})
        return out
    pair = pairs[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        d = np.nanmean(np.vstack([p.diff(con.metric, con.pairing) for p in pairs]), axis=0) if len(pairs) > 1 \
            else pair.diff(con.metric, con.pairing)
        if len(pairs) > 1:
            allfin = np.all(np.isfinite(np.vstack([p.diff(con.metric, con.pairing) for p in pairs])), axis=0)
            d = np.where(allfin, d, np.nan)
    minus_missing = False
    if con.minus is not None:
        mp = PairedTable(arms[con.minus.candidate], arms[con.minus.baseline])
        minus_missing = mp.any_missing()
        extra_invalid = mp.invalid + arms[con.minus.candidate].invalid + arms[con.minus.baseline].invalid
        if extra_invalid or mp.tasks != pair.tasks:
            out.update({"evaluable": False, "reasons": extra_invalid or ["the subtracted contrast covers other tasks"]})
            return out
        d = d - mp.diff(con.metric, con.pairing)
        out["minus"] = {"candidate": con.minus.candidate, "baseline": con.minus.baseline}
        descriptive += [r for a in (con.minus.candidate, con.minus.baseline) for r in arms[a].descriptive]
    mask = np.isfinite(d)
    if not mask.all():
        out["notes"].append(f"complete pairs only: {int((~mask).sum())} tasks without a comparable pair dropped")
    boot = PairedBootstrap(pair.strata[mask], spec.bootstrap_draws, spec.seed, spec.alpha, spec.min_distinct)
    e = boot.mean_diff(d[mask], name=name)
    out.update({"n_tasks": int(mask.sum()), "estimate": e.to_dict(),
                "candidate_mean": float(np.nanmean(pair.task_values("candidate", con.metric))),
                "baseline_mean": float(np.nanmean(np.vstack([p.task_values("baseline", con.metric) for p in pairs])))})
    reasons = []
    if any(p.any_missing() for p in pairs) or minus_missing or not mask.all():
        reasons.append("missing cells or unpaired tasks (in any of the arms the contrast uses)")
    if descriptive:
        reasons.append("descriptive arms: " + "; ".join(descriptive))
    if e.degenerate:
        reasons.append("degenerate bootstrap distribution")
    if con.claim != "none":
        p = e.p_greater(con.threshold) if con.claim == "greater" else e.p_less(con.threshold)
        sup = e.supports_greater(con.threshold) if con.claim == "greater" else e.supports_less(con.threshold)
        out.update({"p_value": p, "supported": bool(sup)})
    out["evaluable"] = not reasons
    out["reasons"] = reasons
    return out


@dataclass
class GateResult:
    spec: GateSpec
    targets: Dict[str, TargetVerdict]
    target_details: Dict[str, Dict[str, Any]]
    contrasts: Dict[str, Dict[str, Any]]
    claims: Dict[str, Dict[str, Any]]
    coverage: Dict[str, Dict[str, Any]]
    notes: List[str] = field(default_factory=list)

    def selected(self, target: Optional[str]) -> Tuple[Optional[str], TargetVerdict]:
        name = target or self.spec.default_target()
        if name is None:
            return None, TargetVerdict("<default>", State.INVALID,
                                       ["--target omitted and the spec does not have exactly one default target"])
        if name not in self.targets:
            return name, TargetVerdict(name, State.INVALID, [f"unknown target {name!r}"])
        return name, self.targets[name]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": {"name": "skillrc-gate", "version": __version__},
            "spec": {"name": self.spec.name, "description": self.spec.description, "seed": self.spec.seed,
                     "bootstrap_draws": self.spec.bootstrap_draws, "alpha": self.spec.alpha,
                     "default_target": self.spec.default_target()},
            "targets": {n: {**v.to_dict(), "details": self.target_details.get(n, {}),
                            "description": self.spec.targets[n].description}
                        for n, v in self.targets.items()},
            "contrasts": self.contrasts,
            "claims": self.claims,
            "coverage": self.coverage,
            "notes": self.notes,
        }


def run_gate(spec: GateSpec, arms: Optional[Dict[str, ArmData]] = None) -> GateResult:
    arms = arms if arms is not None else load_arms(spec)
    targets, details = {}, {}
    for n, t in spec.targets.items():
        targets[n], details[n] = evaluate_target(spec, n, t, arms)
    contrasts = {n: evaluate_contrast(spec, n, c, arms) for n, c in spec.contrasts.items()}
    claims = {}
    for n, j in spec.claims.items():
        parts = {c: contrasts[c] for c in j.all_of}
        evaluable = all(p.get("evaluable") for p in parts.values())
        holds = evaluable and all(p.get("supported") for p in parts.values())
        claims[n] = {"description": j.description, "all_of": j.all_of, "evaluable": evaluable,
                     "holds": bool(holds),
                     "rule": "intersection-union: every listed contrast must support its claim; no adjustment"}
    coverage = {n: {"attempts": a.n_attempts, "expected_cells": len(a.expected) if a.expected is not None else None,
                    "scored_cells": len(a.scored), "missing_cells": len(a.missing),
                    "infra_failed_cells": len(a.infra_cells), "invalid": a.invalid, "descriptive": a.descriptive,
                    "manifest": a.manifest.model_dump() if a.manifest else None}
                for n, a in arms.items()}
    return GateResult(spec, targets, details, contrasts, claims, coverage,
                      [UNCERTAINTY_NOTE, ERROR_BUDGET_NOTE])
