"""Behaviour shifts between two arms: action categories and outcome flags per episode (design v1 G6, first release).

Item-level attribution (which memory item caused which action) is deliberately not automated; the report shows
category and outcome shifts with paired intervals, and links to paired replays.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from .pairing import PairedTable
from .schema import Episode
from .stats import PairedBootstrap

_ASIN = re.compile(r"^b[0-9a-z]{9}$")
_ALF_VERBS = ("go", "take", "put", "open", "close", "toggle", "heat", "cool", "clean", "slice", "use",
              "examine", "look", "inventory", "think")
_SUBPAGES = {"description", "features", "reviews", "attributes"}


def webshop_category(action: str) -> str:
    a = action.strip().lower()
    if a.startswith("search["):
        return "search"
    m = re.match(r"click\[(.*)\]$", a)
    if not m:
        return "unparsed"
    arg = m.group(1).strip()
    if arg in ("< prev", "next >"):
        return "paginate"
    if arg == "back to search":
        return "back_to_search"
    if arg == "buy now":
        return "buy"
    if arg in _SUBPAGES:
        return "subpage"
    if _ASIN.match(arg):
        return "open_product"
    return "choose_option"


def alfworld_category(action: str) -> str:
    w = action.strip().lower().split(" ", 1)[0] if action.strip() else ""
    return w if w in _ALF_VERBS else "other"


def generic_category(action: str) -> str:
    """Tool name for function-call actions (`name(args)` or `name[args]`), else the first word."""
    a = action.strip().lower()
    for sep in ("(", "["):
        if sep in a:
            return a.split(sep, 1)[0].strip() or "other"
    return a.split(" ", 1)[0] or "other"


def categorizer_for(task_uid: str):
    env = task_uid.split(":", 1)[0]
    return {"webshop": webshop_category, "alfworld": alfworld_category}.get(env, generic_category)


def outcome_flags(ep: Episode) -> Dict[str, float]:
    acts = [s.action.strip().lower() for s in (ep.trajectory or [])]
    loop = any(acts[i] == acts[i + 1] == acts[i + 2] for i in range(len(acts) - 2))
    flags = {"success": float(bool(ep.success)) if ep.success is not None else float(ep.score == 1.0),
             "loop_3x": float(loop)}
    if ep.task_uid.startswith("webshop:"):
        bought = any(a == "click[buy now]" for a in acts)
        flags["purchased"] = float(bought)
        flags["no_purchase"] = float(not bought)
    return flags


def _per_task(table: PairedTable, which: str, fn) -> Dict[str, np.ndarray]:
    """Per task: mean over that arm's scored cells of fn(episode) -> {key: value}."""
    rows: List[Dict[str, float]] = []
    keys = set()
    for i in range(table.n):
        acc: Counter = Counter()
        eps = [ep for ep in table.episodes(which, i) if ep.trajectory is not None]
        for ep in eps:
            for k, v in fn(ep).items():
                acc[k] += v
        n = len(eps)
        row = {k: v / n for k, v in acc.items()} if n else None
        keys |= set(row or {})
        rows.append(row)
    return {k: np.array([np.nan if r is None else r.get(k, 0.0) for r in rows]) for k in sorted(keys)}


def behaviour_shift(table: PairedTable, boot: PairedBootstrap) -> Optional[dict]:
    """Category counts and outcome flags, candidate - baseline, ranked by |Δ|. None if no trajectories."""
    def counts(ep: Episode) -> Dict[str, float]:
        cat = categorizer_for(ep.task_uid)
        return dict(Counter(cat(s.action) for s in ep.trajectory or []))

    c_counts, b_counts = _per_task(table, "candidate", counts), _per_task(table, "baseline", counts)
    if not c_counts and not b_counts:
        return None
    if any(np.isnan(v).any() for d in (c_counts, b_counts) for v in d.values()):
        return {"note": "some tasks have no trajectories in one arm; behaviour shift skipped"}
    c_flags, b_flags = _per_task(table, "candidate", outcome_flags), _per_task(table, "baseline", outcome_flags)
    out = {"actions_per_episode": [], "outcomes": []}
    for name, (cd, bd) in (("actions_per_episode", (c_counts, b_counts)), ("outcomes", (c_flags, b_flags))):
        for k in sorted(set(cd) | set(bd)):
            c = cd.get(k, np.zeros(table.n))
            b = bd.get(k, np.zeros(table.n))
            e = boot.mean_diff(c - b, name=f"{name}:{k}")
            out[name].append({"key": k, "candidate": float(c.mean()), "baseline": float(b.mean()),
                              **{f"diff_{x}": v for x, v in (("estimate", e.estimate), ("lower", e.lower),
                                                             ("upper", e.upper))}})
        out[name].sort(key=lambda r: -abs(r["diff_estimate"]))
    out["note"] = "descriptive: paired 95% intervals, no multiplicity adjustment; not part of any verdict"
    return out
