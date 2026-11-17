"""Cells, completeness and task pairing (contract v3 §4, v4 §4–5).

A cell is (task_uid, seed, repetition) within an arm. Its scored attempt is the first attempt without an
infrastructure failure. Cells are grouped into tasks (task_uid), the bootstrap resampling unit.

INVALID conditions found here: duplicate attempt records, two scored attempts in one cell, cells outside the
declared grid, missing cells above the cap, cells with an infrastructure failure above the cap, task identity
mismatch (same task_id, different task_uid), task-type mismatch, different task sets between paired arms.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .schema import Episode, RunManifest, provenance_gaps
from .spec import Grid

Cell = Tuple[str, int, int]          # (task_uid, seed, repetition)


def metric_value(ep: Optional[Episode], metric: str) -> float:
    if ep is None:
        return float("nan")
    if metric == "success":
        v = ep.success
        return float("nan") if v is None else float(v)
    if metric == "total_tokens":
        if ep.prompt_tokens is None or ep.completion_tokens is None:
            return float("nan")
        return float(ep.prompt_tokens + ep.completion_tokens)
    v = getattr(ep, metric)
    return float("nan") if v is None else float(v)


@dataclass
class ArmData:
    name: str
    episodes: List[Episode]
    manifest: Optional[RunManifest] = None
    grid: Optional[Grid] = None
    base_dir: Path = Path(".")
    max_missing_frac: float = 0.005
    max_infra_frac: float = 0.02

    scored: Dict[Cell, Episode] = field(default_factory=dict, init=False)
    infra_cells: Set[Cell] = field(default_factory=set, init=False)
    expected: Optional[Set[Cell]] = field(default=None, init=False)
    missing: Set[Cell] = field(default_factory=set, init=False)
    uid_of_id: Dict[str, str] = field(default_factory=dict, init=False)
    type_of_uid: Dict[str, str] = field(default_factory=dict, init=False)
    invalid: List[str] = field(default_factory=list, init=False)
    descriptive: List[str] = field(default_factory=list, init=False)
    n_attempts: int = field(default=0, init=False)

    def __post_init__(self):
        self._build()

    def _build(self):
        attempts: Dict[Cell, Dict[int, Episode]] = defaultdict(dict)
        legacy = False
        registry = self._load_registry()
        for ep in self.episodes:
            self.n_attempts += 1
            if ep.repetition is None:
                legacy = True
            if ep.task_uid.startswith("unknown:"):
                # an attempt that failed before its task text was seen: identity from the registry, else a
                # placeholder that pairing resolves from the other arm; never registered as a second identity
                if registry is not None and ep.task_id in registry:
                    uid, ttype = registry[ep.task_id]
                    ep = ep.model_copy(update={"task_uid": uid,
                                               "task_type": ttype if ep.task_type in ("unknown", "all") else ep.task_type})
                else:
                    ep = ep.model_copy(update={"task_uid": f"missing-task:{ep.task_id}"})
            cell = (ep.task_uid, ep.seed, ep.repetition or 1)
            if ep.attempt in attempts[cell]:
                self.invalid.append(f"arm {self.name}: duplicate record for cell {cell} attempt {ep.attempt}")
                continue
            attempts[cell][ep.attempt] = ep
            if ep.task_uid.startswith("missing-task:"):
                continue
            prev = self.uid_of_id.setdefault(ep.task_id, ep.task_uid)
            if prev != ep.task_uid:
                self.invalid.append(f"arm {self.name}: task_id {ep.task_id} maps to two task_uids")
            t = self.type_of_uid.setdefault(ep.task_uid, ep.task_type)
            if t != ep.task_type:
                self.invalid.append(f"arm {self.name}: task {ep.task_uid} has two task types")
        for cell, by_attempt in attempts.items():
            scored = [a for a in sorted(by_attempt) if by_attempt[a].scored]
            if any(not e.scored for e in by_attempt.values()):
                self.infra_cells.add(cell)
            if scored:
                first = scored[0]
                if any(a > first for a in scored):
                    self.invalid.append(f"arm {self.name}: cell {cell} has more than one scored attempt")
                self.scored[cell] = by_attempt[first]
        if legacy:
            self.descriptive.append(f"arm {self.name}: no repetition ids (legacy log)")
        if self.manifest is None:
            self.descriptive.append(f"arm {self.name}: no run manifest")
        else:
            gaps = provenance_gaps(self.manifest)
            if gaps:
                self.descriptive.append(f"arm {self.name}: manifest lacks required provenance {gaps}")
        if self.grid is None:
            self.descriptive.append(f"arm {self.name}: no declared grid")
            self.missing = {c for c in attempts if c not in self.scored}
            return
        ids = self.grid.resolve_task_ids(self.base_dir)
        if registry is not None:
            for tid, (uid, ttype) in registry.items():
                seen = self.uid_of_id.get(tid)
                if seen is not None and seen != uid:
                    self.invalid.append(f"arm {self.name}: task {tid} has identity {seen}, registry says {uid}")
                    break
                seen_t = self.type_of_uid.get(uid)
                if seen_t is not None and seen_t != ttype:
                    self.invalid.append(f"arm {self.name}: task {tid} has type {seen_t}, registry says {ttype}")
                    break
            missing_ids = [t for t in ids if t not in registry]
            if missing_ids:
                self.invalid.append(f"arm {self.name}: {len(missing_ids)} grid tasks are not in the registry")
            for tid, (uid, ttype) in registry.items():
                self.uid_of_id.setdefault(tid, uid)
                self.type_of_uid.setdefault(uid, ttype)
        self.expected = set()
        for tid in ids:
            uid = self.uid_of_id.get(tid, f"missing-task:{tid}")
            for s in self.grid.seeds:
                for r in self.grid.repetitions:
                    self.expected.add((uid, s, r))
        extra = set(attempts) - self.expected
        if extra:
            self.invalid.append(f"arm {self.name}: {len(extra)} cells outside the declared grid, e.g. {sorted(extra)[0]}")
        self.missing = self.expected - set(self.scored)
        n = max(1, len(self.expected))
        if len(self.missing) / n > self.max_missing_frac:
            self.invalid.append(f"arm {self.name}: {len(self.missing)}/{n} cells missing "
                                f"(> {self.max_missing_frac:.1%})")
        if len(self.infra_cells) / n > self.max_infra_frac:
            self.invalid.append(f"arm {self.name}: {len(self.infra_cells)}/{n} cells had an infrastructure failure "
                                f"(> {self.max_infra_frac:.1%})")

    def _load_registry(self) -> Optional[Dict[str, Tuple[str, str]]]:
        if self.grid is None or not self.grid.registry:
            return None
        import json
        p = self.base_dir / self.grid.registry
        out = {}
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[str(r["task_id"])] = (r["task_uid"], r.get("task_type", "all"))
        return out

    def resolve_placeholders(self, other: "ArmData") -> None:
        """A task absent from this arm (all its cells missing) takes the identity the other arm observed."""
        if self.expected is None:
            return
        fix = {}
        for cell in self.expected:
            if cell[0].startswith("missing-task:"):
                tid = cell[0].split(":", 1)[1]
                if tid in other.uid_of_id:
                    fix[cell[0]] = other.uid_of_id[tid]
                    self.type_of_uid.setdefault(other.uid_of_id[tid], other.type_of_uid.get(other.uid_of_id[tid], "all"))
        if fix:
            ren = lambda c: (fix.get(c[0], c[0]), c[1], c[2])  # noqa: E731
            self.expected = {ren(c) for c in self.expected}
            self.missing = {ren(c) for c in self.missing}

    @property
    def cells(self) -> Set[Cell]:
        return self.expected if self.expected is not None else set(self.scored) | self.missing

    def tasks(self) -> Set[str]:
        return {c[0] for c in self.cells}


@dataclass
class PairedTable:
    """Two arms aligned on the same tasks. Rows = tasks (sorted task_uid), columns = that arm's cells."""

    candidate: ArmData
    baseline: ArmData
    invalid: List[str] = field(default_factory=list, init=False)
    notes: List[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        c, b = self.candidate, self.baseline
        c.resolve_placeholders(b)
        b.resolve_placeholders(c)
        for tid, uid in c.uid_of_id.items():
            if tid in b.uid_of_id and b.uid_of_id[tid] != uid:
                self.invalid.append(f"task identity mismatch for task_id {tid}: {c.name} has {uid}, "
                                    f"{b.name} has {b.uid_of_id[tid]}")
                break
        ct, bt = c.tasks(), b.tasks()
        if c.expected is not None and b.expected is not None:
            if ct != bt:
                self.invalid.append(f"{c.name} and {b.name} declare different task sets "
                                    f"({len(ct - bt)} only in {c.name}, {len(bt - ct)} only in {b.name})")
            tasks = ct & bt
        else:
            tasks = ct & bt
            if ct != bt:
                self.notes.append(f"descriptive pairing on {len(tasks)} shared tasks "
                                  f"({len(ct - bt)} only in {c.name}, {len(bt - ct)} only in {b.name})")
        for uid in tasks:
            tc, tb = c.type_of_uid.get(uid), b.type_of_uid.get(uid)
            if tc is not None and tb is not None and tc != tb:
                self.invalid.append(f"task {uid} has type {tc} in {c.name} and {tb} in {b.name}")
                break
        self.tasks: List[str] = sorted(tasks)
        self.strata = np.array([c.type_of_uid.get(u) or b.type_of_uid.get(u) or "all" for u in self.tasks])
        self._cols = {arm.name: self._columns(arm) for arm in (c, b)}

    def _columns(self, arm: ArmData) -> List[List[Cell]]:
        by_task: Dict[str, List[Cell]] = defaultdict(list)
        for cell in arm.cells:
            by_task[cell[0]].append(cell)
        return [sorted(by_task.get(u, [])) for u in self.tasks]

    @property
    def n(self) -> int:
        return len(self.tasks)

    def any_missing(self) -> bool:
        tasks = set(self.tasks)
        return any(cell[0] in tasks for arm in (self.candidate, self.baseline) for cell in arm.missing)

    def _arm(self, which: str) -> ArmData:
        return self.candidate if which == "candidate" else self.baseline

    def raw(self, which: str, metric: str, fill_missing: Optional[float] = None) -> np.ndarray:
        """(n_tasks, k) metric values of the arm's cells; NaN (or fill_missing) for missing cells."""
        arm = self._arm(which)
        cols = self._cols[arm.name]
        k = max((len(cs) for cs in cols), default=0)
        out = np.full((self.n, max(k, 1)), np.nan)
        for i, cs in enumerate(cols):
            for j, cell in enumerate(cs):
                ep = arm.scored.get(cell)
                out[i, j] = metric_value(ep, metric) if ep is not None else (
                    fill_missing if fill_missing is not None else np.nan)
        return out

    def blocks(self, which: str) -> np.ndarray:
        arm = self._arm(which)
        cols = self._cols[arm.name]
        k = max((len(cs) for cs in cols), default=1)
        out = np.zeros((self.n, max(k, 1)), dtype=int)
        for i, cs in enumerate(cols):
            for j, cell in enumerate(cs):
                ep = arm.scored.get(cell)
                out[i, j] = (ep.block or 0) if ep is not None else -1
        return out

    def task_values(self, which: str, metric: str, fill_missing: Optional[float] = None) -> np.ndarray:
        r = self.raw(which, metric, fill_missing)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)   # all-missing rows -> NaN
            return np.nanmean(r, axis=1) if r.size else np.full(self.n, np.nan)

    def diff(self, metric: str, pairing: str = "average", fill_cand: Optional[float] = None,
             fill_base: Optional[float] = None) -> np.ndarray:
        """Task-level candidate - baseline differences."""
        if pairing == "average":
            return self.task_values("candidate", metric, fill_cand) - self.task_values("baseline", metric, fill_base)
        if pairing != "same_block":
            raise ValueError(pairing)
        if fill_cand is not None or fill_base is not None:
            raise ValueError("same-block pairing does not impute missing cells; the caller must not request it")
        rc, rb = self.raw("candidate", metric), self.raw("baseline", metric)
        bc, bb = self.blocks("candidate"), self.blocks("baseline")
        out = np.full(self.n, np.nan)
        for i in range(self.n):
            ds = []
            for j in range(rc.shape[1]):
                if bc[i, j] <= 0 or np.isnan(rc[i, j]):
                    continue
                match = [rb[i, jj] for jj in range(rb.shape[1]) if bb[i, jj] == bc[i, j] and not np.isnan(rb[i, jj])]
                if match:
                    ds.append(rc[i, j] - float(np.mean(match)))
            if ds:
                out[i] = float(np.mean(ds))
        return out

    def episodes(self, which: str, task_index: int) -> List[Episode]:
        """Scored episodes of one task in one arm, in cell order."""
        arm = self._arm(which)
        return [arm.scored[c] for c in self._cols[arm.name][task_index] if c in arm.scored]

    def episode(self, which: str, task_index: int, col: int = 0) -> Optional[Episode]:
        arm = self._arm(which)
        cs = self._cols[arm.name][task_index]
        return arm.scored.get(cs[col]) if col < len(cs) else None
