"""Report writers: result.json (machine), summary.md (CI comment), report.html (self-contained page)."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATES = Path(__file__).parent / "templates"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _clean(o):
    """Replace NaN/inf by None so the JSON is standard."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def fmt(x: Optional[float], d: int = 3, sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def interval_svg(rows: List[Dict[str, Any]], refs: List[Dict[str, Any]], width: int = 640) -> str:
    """Horizontal interval chart. rows: label, est, lo, hi, tone. refs: x, label, tone."""
    if not rows:
        return ""
    xs = [v for r in rows for v in (r["lo"], r["hi"], r["est"]) if v is not None] + [r["x"] for r in refs] + [0.0]
    lo, hi = min(xs), max(xs)
    pad = (hi - lo) * 0.08 or 0.05
    lo, hi = lo - pad, hi + pad
    left, right, rh = 190, 24, 30
    h = rh * len(rows) + 34
    sx = lambda v: left + (v - lo) / (hi - lo) * (width - left - right)  # noqa: E731
    out = [f'<svg class="iv" viewBox="0 0 {width} {h}" role="img" aria-label="intervals">']
    for rf in refs:
        x = sx(rf["x"])
        out.append(f'<line x1="{x:.1f}" y1="6" x2="{x:.1f}" y2="{h - 22}" class="ref {rf.get("tone", "")}"/>')
        out.append(f'<text x="{x:.1f}" y="{h - 8}" class="reflab" text-anchor="middle">{html.escape(rf["label"])}</text>')
    for i, r in enumerate(rows):
        y = 18 + i * rh
        out.append(f'<text x="{left - 10}" y="{y + 4}" class="lab" text-anchor="end">{html.escape(r["label"])}</text>')
        if r["lo"] is not None and r["hi"] is not None:
            out.append(f'<line x1="{sx(r["lo"]):.1f}" y1="{y}" x2="{sx(r["hi"]):.1f}" y2="{y}" class="bar {r.get("tone", "")}"/>')
        if r["est"] is not None:
            out.append(f'<circle cx="{sx(r["est"]):.1f}" cy="{y}" r="4.5" class="dot {r.get("tone", "")}"/>')
    out.append("</svg>")
    return "".join(out)


def _target_chart(t: Dict[str, Any]) -> str:
    det = t.get("details", {})
    p = det.get("primary")
    if not p:
        return ""
    rows = [{"label": f"{p['metric']} (candidate − baseline)", "est": p["pessimistic"]["estimate"],
             "lo": p["pessimistic"]["lower"], "hi": p["optimistic"]["upper"], "tone": t["state"].lower()}]
    for k, v in sorted((det.get("blocks", {}).get("per_block") or {}).items()):
        if v:
            rows.append({"label": f"block {k} only", "est": v["estimate"], "lo": v["lower"], "hi": v["upper"],
                         "tone": "muted"})
    refs = [{"x": -p["margin"], "label": f"−δ = {-p['margin']:+.2f}", "tone": "margin"},
            {"x": 0.0, "label": "0", "tone": "zero"}]
    return interval_svg(rows, refs)


def _guard_chart(t: Dict[str, Any]) -> str:
    gs = [g for g in t.get("details", {}).get("guardrails", []) if g.get("estimate")]
    if not gs:
        return ""
    rows = [{"label": g["name"], "est": g["estimate"]["estimate"], "lo": g["estimate"]["lower"],
             "hi": g["estimate"]["upper"], "tone": "muted"} for g in gs]
    refs = [{"x": g["tolerance"], "label": f"τ = {g['tolerance']:+.2f}", "tone": "margin"} for g in gs[:1]]
    refs.append({"x": 0.0, "label": "0", "tone": "zero"})
    return interval_svg(rows, refs)


def _contrast_chart(contrasts: Dict[str, Dict[str, Any]]) -> str:
    rows, refs = [], [{"x": 0.0, "label": "0", "tone": "zero"}]
    for n, c in contrasts.items():
        e = c.get("estimate")
        if not e:
            continue
        tone = "pass" if c.get("supported") else ("muted" if c.get("claim") == "none" else "inconclusive")
        rows.append({"label": n, "est": e["estimate"], "lo": e["lower"], "hi": e["upper"], "tone": tone})
    return interval_svg(rows, refs)


def render_markdown(res: Dict[str, Any], selected: Optional[str] = None) -> str:
    L = [f"## skillrc-gate: {res['spec']['name']}", ""]
    L.append("| target | verdict | primary Δ [95% interval] | margin | guardrails |")
    L.append("|---|---|---|---|---|")
    for n, t in res["targets"].items():
        det = t.get("details", {})
        p = det.get("primary")
        iv = (f"{fmt(p['pessimistic']['estimate'])} [{fmt(p['pessimistic']['lower'])}, {fmt(p['optimistic']['upper'])}]"
              if p else "–")
        gs = "; ".join(f"{g['name']} {fmt(g['estimate']['estimate'] if g.get('estimate') else None)}"
                       for g in det.get("guardrails", [])) or "–"
        mark = " ◀" if n == selected else ""
        L.append(f"| {n}{mark} | **{t['state']}** | {iv} | {p['margin'] if p else '–'} | {gs} |")
    L.append("")
    for n, t in res["targets"].items():
        L.append(f"**{n}** — {t.get('description') or ''}")
        for r in t["reasons"]:
            L.append(f"- {r}")
        for f in t.get("flags", []):
            L.append(f"- ⚑ {f}")
        L.append("")
    if res.get("claims"):
        L.append("| claim | evaluable | holds |")
        L.append("|---|---|---|")
        for n, c in res["claims"].items():
            L.append(f"| {n} | {c['evaluable']} | {c['holds']} |")
        L.append("")
    L.extend(f"> {x}" for x in res.get("notes", []))
    return "\n".join(L) + "\n"


def render_html(res: Dict[str, Any], selected: Optional[str] = None) -> str:
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES)), autoescape=select_autoescape(["html", "j2"]))
    env.filters["f"] = fmt
    env.filters["pct"] = lambda x: "–" if x is None else f"{100 * x:.1f}%"
    tpl = env.get_template("report.html.j2")
    charts = {n: {"primary": _target_chart(t), "guard": _guard_chart(t)} for n, t in res["targets"].items()}
    return tpl.render(res=res, charts=charts, selected=selected, contrast_chart=_contrast_chart(res.get("contrasts", {})))


def write_reports(result_dict: Dict[str, Any], out_dir: Path, selected: Optional[str] = None,
                  formats=("json", "md", "html")) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = _clean(json.loads(json.dumps(result_dict, default=_json_default)))
    paths = {}
    if "json" in formats:
        p = out_dir / "result.json"
        p.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
        paths["json"] = p
    if "md" in formats:
        p = out_dir / "summary.md"
        p.write_text(render_markdown(res, selected))
        paths["md"] = p
    if "html" in formats:
        p = out_dir / "report.html"
        p.write_text(render_html(res, selected))
        paths["html"] = p
    return paths
