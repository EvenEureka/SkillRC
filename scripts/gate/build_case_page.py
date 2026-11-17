#!/usr/bin/env python3
"""Build the case page docs/gate/index.html from the gate's own result.json files (no hand-copied numbers).

Inputs (any may be missing; a missing release check is shown as running):
  examples/webshop_bundled_release/report/result.json     historical bundled release (DESCRIPTIVE)
  examples/webshop_release_check/report/result.json       pre-registered release check, Qwen3-8B
  examples/webshop_release_check_32b/report/result.json   Qwen3-32B extension
  examples/inspect_agentdojo/report/result.json           Inspect AI producer
Copies each report.html to docs/gate/reports/<case>.html and links it.
"""
from __future__ import annotations

import html
import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def stacked_svg(rows, width=480, rh=50):
    """Compact interval chart for phones and desktops: label above each bar, value beside the dot, zero line."""
    if not rows:
        return ""
    lo = min([0.0] + [r["lo"] for r in rows]); hi = max([0.0] + [r["hi"] for r in rows])
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    left, right = 10, 70
    h = rh * len(rows) + 34
    sx = lambda v: left + (v - lo) / (hi - lo) * (width - left - right)  # noqa: E731
    z = sx(0.0)
    out = [f'<svg class="st" viewBox="0 0 {width} {h}" role="img" aria-label="score change by change">',
           f'<text x="{z:.1f}" y="{h - 8}" class="reflab" text-anchor="middle">┆ 0 = no change</text>']
    for i, r in enumerate(rows):
        y = 16 + i * rh
        out.append(f'<text x="{left}" y="{y}" class="lab">{html.escape(r["label"])}</text>')
        yb = y + 18
        out.append(f'<line x1="{z:.1f}" y1="{yb - 11}" x2="{z:.1f}" y2="{yb + 11}" class="zero"/>')
        out.append(f'<line x1="{sx(r["lo"]):.1f}" y1="{yb}" x2="{sx(r["hi"]):.1f}" y2="{yb}" class="bar {r["tone"]}"/>')
        out.append(f'<circle cx="{sx(r["est"]):.1f}" cy="{yb}" r="6" class="dot {r["tone"]}"/>')
        out.append(f'<text x="{sx(r["hi"]) + 8:.1f}" y="{yb + 5}" class="val {r["tone"]}">{f(r["est"])}</text>')
    out.append("</svg>")
    return "".join(out)


def interval_svg(rows, refs, width=820, left=330, right=24, rh=32):
    """Case-page interval chart (own copy, so the frozen report renderer is untouched): full labels on the left,
    reference labels placed beside their lines so they never overlap."""
    if not rows:
        return ""
    xs = [v for r in rows for v in (r["lo"], r["hi"], r["est"]) if v is not None] + [r["x"] for r in refs]
    lo, hi = min(xs), max(xs)
    pad = (hi - lo) * 0.06 or 0.05
    lo, hi = lo - pad, hi + pad
    h = rh * len(rows) + 40
    sx = lambda v: left + (v - lo) / (hi - lo) * (width - left - right)  # noqa: E731
    out = [f'<svg class="iv" viewBox="0 0 {width} {h}" role="img" aria-label="score change intervals by release target">']
    for rf in refs:
        x = sx(rf["x"])
        anchor = "end" if rf.get("tone") == "margin" else "start"
        dx = -4 if anchor == "end" else 4
        out.append(f'<line x1="{x:.1f}" y1="6" x2="{x:.1f}" y2="{h - 24}" class="ref {rf.get("tone", "")}"/>')
        out.append(f'<text x="{x + dx:.1f}" y="{h - 9}" class="reflab" text-anchor="{anchor}">{html.escape(rf["label"])}</text>')
    for i, r in enumerate(rows):
        y = 20 + i * rh
        out.append(f'<text x="{left - 12}" y="{y + 4}" class="lab" text-anchor="end">{html.escape(r["label"])}</text>')
        out.append(f'<line x1="{sx(r["lo"]):.1f}" y1="{y}" x2="{sx(r["hi"]):.1f}" y2="{y}" class="bar {r["tone"]}"/>')
        out.append(f'<circle cx="{sx(r["est"]):.1f}" cy="{y}" r="5" class="dot {r["tone"]}"/>')
    out.append("</svg>")
    return "".join(out)


OUT = ROOT / "docs/gate"
REPO = "https://github.com/EvenEureka/SkillRC"
CASES = {
    "historical": "examples/webshop_bundled_release",
    "release": "examples/webshop_release_check",
    "release32": "examples/webshop_release_check_32b",
    "inspect": "examples/inspect_agentdojo",
}


def load(case):
    p = ROOT / CASES[case] / "report/result.json"
    return json.loads(p.read_text()) if p.exists() else None


def f(x, d=3):
    return "–" if x is None else f"{x:+.{d}f}".replace("-", "−")


def pct(x, d=0):
    return "–" if x is None else f"{100 * x:+.{d}f}%".replace("-", "−")


def prim(t):
    p = t["details"]["primary"]
    return p["pessimistic"]["estimate"], p["pessimistic"]["lower"], p["optimistic"]["upper"], p


def guard(t, metric):
    for g in t["details"].get("guardrails", []):
        if g["name"].startswith(metric) and g.get("estimate"):
            return g["estimate"]["estimate"], g["estimate"]["lower"], g["estimate"]["upper"]
    return None, None, None


def badge(state):
    return f'<span class="badge {state}">{state}</span>'


def chart(res):
    rows = []
    for case, label, target in (
        ("historical", "historical: label + memory vs old release", "bundle_vs_old"),
        ("historical", "historical: page label alone", "label_only"),
        ("historical", "historical: memory, given the label", "memory_given_label"),
        ("release", "pre-registered C1: ship the memory", "C1_ship_pub"),
        ("release", "pre-registered C2: ship the repaired memory", "C2_ship_repaired"),
        ("release32", "Qwen3-32B C1: ship the memory", "C1_32_ship_pub"),
        ("inspect", "Inspect AgentDojo: add a tips block", "add_tips"),
    ):
        r = res.get(case)
        if not r or target not in r["targets"] or not r["targets"][target]["details"].get("primary"):
            continue
        t = r["targets"][target]
        est, lo, hi, _ = prim(t)
        rows.append({"label": label, "est": est, "lo": lo, "hi": hi, "tone": t["state"].lower()})
    return interval_svg(rows, [{"x": -0.05, "label": "−δ = −0.05", "tone": "margin"}, {"x": 0.0, "label": "0", "tone": "zero"}])


def beh(sh, key, which):
    """Look up one behaviour row (actions or outcomes) by key."""
    for part in ("actions_per_episode", "outcomes"):
        for row in (sh or {}).get(part, []):
            if row["key"] == key:
                return row[which]
    return None


def iv(e):
    return f"{f(e['estimate'])} [{f(e['lower'])}, {f(e['upper'])}]"


def release_view(r, extras):
    """Everything the release-check findings need, read from result.json and extras.json."""
    if r is None:
        return None
    v = {}
    for key, name in (("c1", "C1_ship_pub"), ("c2", "C2_ship_repaired")):
        t = r["targets"].get(name)
        if not t or not t["details"].get("primary"):
            continue
        est, lo, hi, p = prim(t)
        comp = {c["kind"]: c["state"] for c in t.get("components", []) if c["kind"] == "primary"}
        v[key] = {"state": t["state"], "est": est, "lo": lo, "hi": hi, "cand": p["candidate_mean"],
                  "base": p["baseline_mean"], "tok": guard(t, "prompt_tokens"), "lat": guard(t, "latency_s"),
                  "primary_state": comp.get("primary"),
                  "sens": {m["margin"]: m["primary_component"] for m in p.get("margin_sensitivity", [])},
                  "beh": t["details"].get("behaviour")}
    loc = r["claims"].get("localization")
    if loc:
        names = dict(LOC_ROWS)
        v["loc"] = {"evaluable": loc["evaluable"], "holds": loc["holds"],
                    "failed": [names.get(c, c) for c in loc["all_of"] if not r["contrasts"][c].get("supported")]}
    v["con"] = {k: c for k, c in r["contrasts"].items() if c.get("estimate")}
    v["extras"] = extras
    return v


LOC_ROWS = (("L1_abl_vs_true", "repair vs published memory"),
            ("L2_abl_vs_comp1", "repair vs comparison edit 1"),
            ("L2_abl_vs_comp2", "repair vs comparison edit 2"),
            ("L2_abl_vs_comp3", "repair vs comparison edit 3"),
            ("L2_abl_vs_mean_comp", "repair vs mean of the three edits"))


def localization_chart(r):
    if r is None:
        return ""
    rows = []
    for name, label in LOC_ROWS:
        c = r["contrasts"].get(name)
        if not c or not c.get("estimate"):
            continue
        e = c["estimate"]
        tone = "pass" if c.get("supported") else ("block" if e["upper"] < 0 else "neutral")
        rows.append({"label": label, "est": e["estimate"], "lo": e["lower"], "hi": e["upper"], "tone": tone})
    return interval_svg(rows, [{"x": -0.1, "label": "−0.1", "tone": "zero"}, {"x": 0.0, "label": "0", "tone": "zero"},
                               {"x": 0.1, "label": "+0.1", "tone": "zero"}])


def replay(r, target):
    if not r or target not in r["targets"]:
        return None
    reps = r["targets"][target]["details"].get("replays") or []
    if not reps:
        return None
    rp = reps[0]
    return {"task": rp["task_id"], "type": rp["task_type"], "why": rp["why"], "div": rp["first_divergent_step"],
            "base": rp["baseline"], "cand": rp["candidate"]}


PLAIN = {"PASS": "ship", "BLOCK": "don't ship", "INCONCLUSIVE": "not enough evidence, not approved",
         "INVALID": "experiment broken", "DESCRIPTIVE": "numbers only, no decision"}


def verdict(state):
    return f'<span class="badge {state}">{state}</span> <span class="plain">{PLAIN.get(state, "")}</span>'


def hidden_chart(hist):
    rows = []
    for key, label in (("bundle_vs_old", "label + memory, shipped together"),
                       ("label_only", "page label alone"),
                       ("memory_given_label", "memory, added on top of the label")):
        h = hist.get(key)
        if h:
            rows.append({"label": label, "est": h["est"], "lo": h["lo"], "hi": h["hi"],
                         "tone": "pass" if h["est"] > 0 else "block"})
    return stacked_svg(rows)


def main():
    res = {k: load(k) for k in CASES}
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    links = {}
    for k, d in CASES.items():
        src = ROOT / d / "report/report.html"
        if src.exists():
            shutil.copy(src, OUT / "reports" / f"{k}.html")
            links[k] = f"reports/{k}.html"
    h = res["historical"]["targets"] if res["historical"] else {}
    hist = {}
    for n in ("bundle_vs_old", "label_only", "memory_given_label"):
        if n in h:
            est, lo, hi, _ = prim(h[n])
            hist[n] = {"est": est, "lo": lo, "hi": hi, "tok": guard(h[n], "prompt_tokens")[0]}
    ins = res["inspect"]["targets"]["add_tips"] if res["inspect"] else None
    ins_d = None
    if ins:
        est, lo, hi, p = prim(ins)
        ins_d = {"state": ins["state"], "est": est, "lo": lo, "hi": hi, "tok": guard(ins, "prompt_tokens"),
                 "lat": guard(ins, "latency_s"), "n": ins["details"]["n_tasks"], "mde": ins["details"]["power"]["mde_80_two_sided_0.05"],
                 "cand": p["candidate_mean"], "base": p["baseline_mean"]}
    ex_path = ROOT / CASES["release"] / "extras.json"
    extras = json.loads(ex_path.read_text()) if ex_path.exists() else None
    from jinja2 import Template
    page = Template(TEMPLATE).render(
        hidden=hidden_chart(hist), verdict=verdict,
        chart=chart(res), loc_chart=localization_chart(res["release"]), hist=hist, ins=ins_d,
        rel=release_view(res["release"], extras), rel32=release_view(res["release32"], None),
        beh=beh, iv=iv,
        rep=replay(res["release"], "C1_ship_pub") or replay(res["historical"], "memory_given_label"),
        links=links, repo=REPO, today=date.today().isoformat(), f=f, pct=pct, badge=badge)
    (OUT / "index.html").write_text(page)
    print("wrote", OUT / "index.html", "reports:", sorted(links))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>skillrc-gate | Should this agent change ship?</title>
<meta name="description" content="A release check for agent memory, prompt and harness changes, computed from episode logs: paired intervals, cost and latency guardrails, provenance checks, and paired replays.">
<style>
  :root { color-scheme: light; --bg:#fdfcfc; --surface:#fff; --card:#f4eeef; --rule:#e6dfe1; --text:#1a1718; --text-2:#4f4a4c;
    --text-3:#7f7a7c; --accent:#5b0a18; --link:#2e7fa3; --code-bg:#f7f3f4; --pass:#1f7a4d; --pass-bg:#e6f3ec; --block:#b3261e;
    --block-bg:#fbe9e7; --inc:#8a6100; --inc-bg:#fbf1dc; --desc:#3f5f8a; --desc-bg:#e8eef7; --inv:#5b3f99; --inv-bg:#efe9fa;
    --serif:"Libertinus Serif","Linux Libertine O","Iowan Old Style",Palatino,Georgia,serif;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --bg:#161314; --surface:#1f1b1c;
    --card:#2a2022; --rule:#3a3234; --text:#f3eeef; --text-2:#c9c1c3; --text-3:#9d9497; --accent:#e8a1ad; --link:#7cc3df;
    --code-bg:#262022; --pass:#6fcf97; --pass-bg:#173326; --block:#f2847c; --block-bg:#3a1d1a; --inc:#e5b85c; --inc-bg:#3a2f15;
    --desc:#92b4e3; --desc-bg:#1c2738; --inv:#b9a3ee; --inv-bg:#2b2340; } }
  :root[data-theme="dark"] { color-scheme: dark; --bg:#161314; --surface:#1f1b1c; --card:#2a2022; --rule:#3a3234; --text:#f3eeef;
    --text-2:#c9c1c3; --text-3:#9d9497; --accent:#e8a1ad; --link:#7cc3df; --code-bg:#262022; --pass:#6fcf97; --pass-bg:#173326;
    --block:#f2847c; --block-bg:#3a1d1a; --inc:#e5b85c; --inc-bg:#3a2f15; --desc:#92b4e3; --desc-bg:#1c2738; --inv:#b9a3ee; --inv-bg:#2b2340; }
  * { box-sizing: border-box; } html { -webkit-text-size-adjust: 100%; }
  body { margin:0; background:var(--bg); color:var(--text); font-family:var(--serif); font-size:18px; line-height:1.6; }
  main { max-width: 860px; margin: 0 auto; padding: 0 16px 64px; }
  a { color: var(--link); }
  header { text-align:center; padding: 44px 0 16px; }
  h1 { font-size: clamp(26px,4.6vw,38px); line-height:1.2; margin:0 auto 6px; color:var(--accent); text-wrap: balance; }
  .subtitle { font-size: clamp(18px,3vw,21px); color: var(--text-2); margin: 0 0 16px; text-wrap: balance; }
  .links { display:flex; flex-wrap:wrap; gap:10px; justify-content:center; font-family:var(--sans); }
  .btn { display:inline-block; padding:8px 16px; border-radius:999px; border:1px solid var(--accent); color:var(--accent); text-decoration:none; font-size:15px; font-weight:600; }
  .btn.primary { background: var(--accent); color: var(--bg); }
  h2 { font-size: 24px; color: var(--accent); margin: 38px 0 10px; }
  p { margin: 0 0 14px; }
  .question { border-left: 3px solid var(--accent); padding-left: 12px; font-size: 19px; }
  ol.findings { padding-left: 1.3em; } ol.findings li { margin: 0 0 14px; }
  .badge { font-family: var(--sans); font-size: 13px; font-weight: 700; padding: 2px 9px; border-radius: 999px; letter-spacing: .03em; }
  .PASS{color:var(--pass);background:var(--pass-bg)} .BLOCK{color:var(--block);background:var(--block-bg)}
  .INCONCLUSIVE{color:var(--inc);background:var(--inc-bg)} .DESCRIPTIVE{color:var(--desc);background:var(--desc-bg)}
  .INVALID{color:var(--inv);background:var(--inv-bg)}
  figure { margin: 22px 0 8px; } .frame { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 10px; overflow-x: auto; }
  figcaption { font-size: 15px; color: var(--text-2); margin-top: 8px; line-height: 1.5; }
  svg.iv { width:100%; min-width: 640px; height:auto; font-family: var(--sans); } svg.iv .lab, svg.iv .reflab { fill: var(--text-2); font-size: 12px; }
  svg.iv .bar { stroke: var(--text); stroke-width: 3; stroke-linecap: round } svg.iv .dot { fill: var(--text) }
  svg.iv .bar.block{stroke:var(--block)} svg.iv .dot.block{fill:var(--block)} svg.iv .bar.pass{stroke:var(--pass)} svg.iv .dot.pass{fill:var(--pass)}
  svg.iv .bar.inconclusive{stroke:var(--inc)} svg.iv .dot.inconclusive{fill:var(--inc)} svg.iv .bar.descriptive{stroke:var(--desc)} svg.iv .dot.descriptive{fill:var(--desc)}
  svg.iv .bar.invalid{stroke:var(--inv)} svg.iv .dot.invalid{fill:var(--inv)}
  svg.iv .bar.neutral{stroke:var(--text-3)} svg.iv .dot.neutral{fill:var(--text-3)}
  svg.iv .ref { stroke: var(--rule); stroke-width: 1.5 } svg.iv .ref.margin { stroke: var(--block); stroke-dasharray: 4 3 }
  table { border-collapse: collapse; width: 100%; font-family: var(--sans); font-size: 14.5px; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--rule); vertical-align: top; } th { color: var(--text-3); font-weight: 600; }
  .scroll { overflow-x: auto; }
  .replay { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; } @media (max-width: 680px) { .replay { grid-template-columns: 1fr; } }
  .traj { font-family: var(--mono); font-size: 12.5px; line-height: 1.45; background: var(--code-bg); border: 1px solid var(--rule); border-radius: 6px; padding: 8px 10px; overflow-x: auto; }
  .traj div { white-space: pre-wrap; word-break: break-word; border-bottom: 1px dashed var(--rule); padding: 1px 0; } .traj .div { background: var(--inc-bg); }
  .traj .n { color: var(--text-3); display: inline-block; width: 2.2em; }
  pre { background: var(--code-bg); border: 1px solid var(--rule); border-radius: 6px; padding: 12px; overflow-x: auto; font-family: var(--mono); font-size: 13px; }
  .pending { background: var(--card); border-radius: 8px; padding: 12px 14px; font-family: var(--sans); font-size: 15px; }
  .note { font-size: 16px; color: var(--text-2); }
  footer { margin-top: 44px; padding-top: 14px; border-top: 1px solid var(--rule); font-size: 14px; color: var(--text-3); text-align: center; }
  svg.st { display: block; width: 100%; max-width: 560px; height: auto; margin: 0 auto; font-family: var(--sans); }
  svg.st .lab { fill: var(--text); font-size: 14px; } svg.st .reflab { fill: var(--text-3); font-size: 12px; }
  svg.st .zero { stroke: var(--text-3); stroke-width: 1.2; stroke-dasharray: 3 3; }
  svg.st .bar { stroke-width: 4; stroke-linecap: round; } svg.st .val { font-size: 14px; font-weight: 700; }
  svg.st .pass { stroke: var(--pass); fill: var(--pass); } svg.st .block { stroke: var(--block); fill: var(--block); }
  svg.st text.pass, svg.st text.block { stroke: none; }
  ul.plainlist { padding-left: 1.2em; } ul.plainlist li { margin: 0 0 6px; }
  .plain { font-family: var(--sans); font-size: 14px; color: var(--text-2); }
  table.vt td:first-child { width: 34%; } table.vt td:last-child { white-space: nowrap; }
  @media (max-width: 680px) { table.vt td:last-child { white-space: normal; } }
  details { background: var(--card); border-radius: 8px; padding: 10px 14px; margin: 8px 0 0; }
  details summary { cursor: pointer; font-family: var(--sans); font-size: 15px; font-weight: 600; color: var(--accent); }
  details ul { font-size: 16px; color: var(--text-2); padding-left: 1.2em; } details li { margin: 0 0 8px; }
  h3 { font-size: 20px; margin: 26px 0 8px; }
</style>
</head>
<body>
<main>
<header>
  <h1>skillrc-gate</h1>
  <p class="subtitle">Should this agent change ship? A release check that answers from the run logs.</p>
  <div class="links">
    <a class="btn primary" href="{{ repo }}/tree/main/skillrc/gate">Code</a>
    <a class="btn" href="{{ repo }}/blob/main/docs/gate/README.md">Docs</a>
    {% if links.release %}<a class="btn" href="{{ links.release }}">Full technical report</a>{% endif %}
  </div>
</header>

<h2>The problem</h2>
<p>Teams that build LLM agents change them all the time: they add a memory of lessons from past runs, rewrite the
prompt, or switch on a new feature. Before each release someone has to decide whether the new version is actually
better. The usual method is to run both versions on a test set and compare the average score. That method fails in
predictable ways:</p>
<ul class="plainlist">
  <li><strong>Luck.</strong> A small difference can come from which tasks happened to be in the test set.</li>
  <li><strong>Bundling.</strong> When two changes ship together, a good one can hide a bad one.</li>
  <li><strong>Unfair comparison.</strong> Something else changed between the two runs: the model version, a setting, the task list.</li>
  <li><strong>Dropped failures.</strong> Tasks that crashed are quietly left out, which shifts the average.</li>
  <li><strong>Hidden cost.</strong> The score holds, but every task now uses far more tokens or time.</li>
</ul>

{% if hist %}
<h2>A real case: a good release hid a bad change</h2>
<p>In our SkillRC project, a shopping agent (Qwen3-8B on the WebShop benchmark, 500 tasks) received two changes at
once: a short label that tells it which page it is on, and a memory of 30 lessons distilled from earlier runs. Together
they raised the average task score (0 to 1) by {{ f(hist.bundle_vs_old.est) }}, so the release looked good. Tested one at
a time, the label was worth {{ f(hist.label_only.est) }} and the memory {{ f(hist.memory_given_label.est) }}. The memory
made the agent worse, and the average hid it.</p>
<figure>
  <div class="frame">{{ hidden | safe }}</div>
  <figcaption>Change in average task score versus the version without that change, on the same 500 tasks. Dot and
  number: the measured change. Bar: the range the true change lies in with 95% confidence.</figcaption>
</figure>
{% endif %}

<h2>What skillrc-gate does</h2>
<p>You give it the logs of the old and the new version on the same tasks, plus a short file that names each change.
For every change separately, it answers with one of these:</p>
<div class="scroll"><table>
<tr><td>{{ verdict("PASS") }}</td><td>{{ verdict("BLOCK") }}</td></tr>
<tr><td>{{ verdict("INCONCLUSIVE") }}</td><td>{{ verdict("INVALID") }}</td></tr>
</table></div>
<p>It is a command-line tool, and its exit code is the answer, so a CI pipeline can stop a release the way a failing
unit test stops a merge. Before it answers, it checks four things:</p>
<ol>
  <li><strong>Same conditions.</strong> The two versions may differ only in the declared change. The model version,
  settings, task list and a fingerprint (hash) of every input file are compared; any other difference means
  "experiment broken".</li>
  <li><strong>Nothing dropped.</strong> Every planned task needs a result. Crashed tasks are rerun and counted, never
  removed.</li>
  <li><strong>Real or luck.</strong> It resamples the tasks thousands of times to find the range the true difference
  lies in. "Don't ship" needs the whole range to be worse than the allowed drop (here 0.05). "Ship" needs the whole range
  to be better than that. Anything in between is "not enough evidence".</li>
  <li><strong>Cost.</strong> Tokens and time per task must stay within a limit (here +30%).</li>
</ol>
<p>It also reports how the agent's behaviour changed and shows the runs of both versions on the same task side by
side.</p>

{% if rel and rel.c1 %}{% set c1 = rel.c1 %}{% set c2 = rel.c2 %}
<h2>Tested on a real release decision</h2>
<p>We then ran a fresh experiment and let the gate decide. The plan and the decision rules were written down and frozen
before anything ran.</p>
<div class="scroll"><table class="vt">
<tr><th>change</th><th>what happened</th><th>verdict</th></tr>
<tr><td>Ship the memory</td><td>score {{ "%.2f" | format(c1.base) }} → {{ "%.2f" | format(c1.cand) }};
  tokens per task {{ pct(c1.tok[0]) }}</td><td>{{ verdict(c1.state) }}</td></tr>
{% if c2 %}<tr><td>Ship a "fixed" memory (six suspicious lessons replaced)</td><td>score still {{ "%.3f" | format(-c2.est) }} lower than
  with no memory; tokens per task {{ pct(c2.tok[0]) }}</td><td>{{ verdict(c2.state) }}</td></tr>{% endif %}
{% if ins %}<tr><td>Add six tips to the prompt (another benchmark, AgentDojo, run through the Inspect AI framework)</td>
  <td>score {{ f(ins.est) }}, too few tasks to tell</td><td>{{ verdict(ins.state) }}</td></tr>{% endif %}
</table></div>

<h3>What we learned</h3>
<ol class="findings">
  <li><strong>The memory makes the agent keep searching instead of buying.</strong> With the memory, the agent went back
  to the search page {{ "%.2f" | format(beh(c1.beh, "back_to_search", "candidate")) }} times per task instead of
  {{ "%.2f" | format(beh(c1.beh, "back_to_search", "baseline")) }}, and ended
  {{ "%.0f" | format(100 * beh(c1.beh, "no_purchase", "candidate")) }}% of tasks without buying anything instead of
  {{ "%.0f" | format(100 * beh(c1.beh, "no_purchase", "baseline")) }}%. It also used {{ "%.0f" | format(100 * c1.tok[0]) }}% more tokens.
  The gate blocks it on both score and cost.</li>
  {% if rel.extras and rel.loc %}{% set x = rel.extras.behaviour %}
  <li><strong>Removing the obvious culprit did not fix it.</strong> Six of the 30 lessons tell the agent to keep
  searching. On 500 new tasks, replacing those six with neutral text of the same length stopped the extra searching
  ({{ "%.2f" | format(beh(x["D_ABL-D_TRUE"], "back_to_search", "baseline")) }} →
  {{ "%.2f" | format(beh(x["D_ABL-D_TRUE"], "back_to_search", "candidate")) }} returns to search per task;
  {{ "%.2f" | format(beh(x["D_TRUE-D_NONE"], "back_to_search", "baseline")) }} without any memory). But it won back
  only {{ "%.3f" | format(x["D_ABL-D_TRUE"].score_diff) }} of the {{ "%.3f" | format(-x["D_TRUE-D_NONE"].score_diff) }}
  that the memory had cost, and that gain could be zero (95% range [{{ f(rel.con.L1_abl_vs_true.estimate.lower) }},
  {{ f(rel.con.L1_abl_vs_true.estimate.upper) }}]). As a
  control, we also replaced three other, randomly chosen sets of six lessons. One of them won back more than our fix
  did ({{ "%+.3f" | format(x["D_COMP1-D_TRUE"].score_diff) }}). So the data do not support "these six lessons caused the
  loss", and the gate does not approve the fixed memory.</li>
  {% endif %}
  {% if ins %}
  <li><strong>When the evidence is thin, it says so.</strong> On {{ ins.n }} AgentDojo tasks, only a score change of
  about {{ "%.2f" | format(ins.mde) }} can be detected reliably; the tips changed the score by {{ f(ins.est) }}. The
  gate answers "not enough evidence" instead of guessing.</li>
  {% endif %}
</ol>
<p class="note">Next: the same check with a larger model (Qwen3-32B) is running{% if rel32 and rel32.c1 %}; result:
{{ verdict(rel32.c1.state) }}, score change {{ f(rel32.c1.est) }}{% else %}, and its result will be added here{% endif %}.</p>
{% endif %}

{% if rep %}
<h2>Same task, two versions</h2>
<p class="note">Task {{ rep.task }} ({{ rep.type }}), picked by a fixed rule rather than by hand: the first task, by
id, where the version without the memory scored higher. Highlighted: the first step where the two runs differ.
{% if rep.task|string == "1" %}Here the version with the memory clicked “buy now” before choosing the colour and size, so it got
only partial credit.{% endif %}</p>
<div class="replay">
  {% for side, e in (("without the memory", rep.base), ("with the memory", rep.cand)) %}
  <div><p class="note"><strong>{{ side }}</strong> · score {{ "%.2f" | format(e.score) }} · {{ e.steps }} steps</p>
  <div class="traj">{% for s in e.trajectory %}<div class="{{ 'div' if s.step == rep.div else '' }}"><span class="n">{{ s.step }}</span>{{ s.action }}</div>{% endfor %}</div></div>
  {% endfor %}
</div>
{% endif %}

<h2>Try it</h2>
<p>Runs on a laptop with the real logs from the case above; no GPU or API key needed.</p>
<pre>pip install -e .
skillrc-gate report examples/webshop_bundled_release/gate.yaml --out /tmp/gate_demo
open /tmp/gate_demo/report.html</pre>

<h2>How it is built</h2>
<ul class="plainlist">
  <li>Reads logs from the SkillRC agent runner, from the Inspect AI framework, or from a simple JSON-lines format.</li>
  <li>The runner is built for long unattended GPU jobs: every attempt is logged before it starts, a crashed run resumes
  where it stopped, and each job checks that the previous one finished correctly before it starts.</li>
  {% if rel and rel.extras %}{% set t = rel.extras.totals %}{% set rr = rel.extras.reruns %}
  <li>The release experiment ran {{ "{:,}".format(t.arm_runs_attempts) }} agent episodes with
  {{ t.infra_failed_cells }} crashes and {{ t.missing_cells }} missing results.</li>
  <li>Every verdict on this page can be reproduced from the published logs: one script restores the logs, one command
  recomputes the report.</li>
  <li>94 automated tests; the decision rule was also checked on simulated data where the true answer is known.</li>
  <li>One lesson: we ran every setting twice to measure run-to-run noise. Both runs gave identical results on all
  {{ "{:,}".format(rr.values() | sum(attribute="same_score_steps_tokens")) }} task pairs (greedy decoding, seeded
  environment). In this setup the second run added nothing; more tasks would have been a better use of the GPUs.</li>
  {% endif %}
</ul>

<h2>Details and limits</h2>
<details>
<summary>Statistics, assumptions and review record</summary>
<ul>
  <li>Ranges are 95% intervals from a paired bootstrap over tasks (20,000 resamples, stratified by product category,
  fixed seed). They describe the uncertainty from which tasks were sampled, given the runs we made; they do not cover
  future runs.</li>
  <li>Cost checks: "don't ship" only if the limit is confirmed exceeded (Holm correction across the two cost checks);
  "ship" only if the whole range stays under the limit.{% if rel and rel.c2 %} The fixed memory's token increase was
  {{ pct(rel.c2.tok[0], 1) }} with range [{{ pct(rel.c2.tok[1], 1) }}, {{ pct(rel.c2.tok[2], 1) }}], straddling the
  +30% limit, so its cost check was also "not enough evidence". Its score range was
  [{{ f(rel.c2.lo) }}, {{ f(rel.c2.hi) }}]: below zero, but not wholly below −0.05. With an allowed drop of 0.02 it
  would have been blocked.{% endif %}</li>
  <li>The allowed drop (0.05) and the cost limit (+30%) are example policies fixed before the runs, not
  recommendations. In simulation, the chance of "ship" when the true change sat exactly at the allowed drop was
  0.018–0.041, against a target of 0.025.</li>
  <li>The "fix" replaces the six lessons with unrelated text of the same length, so it tests replacement, not
  deletion. The localization test required the fix to beat the original memory and each of the three random
  replacements{% if rel and rel.loc %}; it failed for {{ rel.loc.failed | join("; ") }}{% endif %}. It covers those
  three replacements only.</li>
  <li>Behaviour numbers (returns to search, tasks without a purchase) and the "won back" amounts for the random
  replacements are descriptive and not part of any verdict.</li>
  <li>The two repeat runs matched in score, steps and tokens, not in latency; this does not prove determinism in other
  setups.</li>
  <li>WebShop here is the 1,000-product version with generated shopping goals and a 15-step limit; most tasks are
  fashion items. The AgentDojo check used 96 tasks, two attempts each.</li>
  <li>The bundled-release case comes from an older run without the provenance records the gate requires, so the gate
  shows its numbers without a decision ({{ verdict("DESCRIPTIVE") }}).</li>
  <li>The design and implementation went through eleven rounds of review by an independent model reviewer (Codex)
  before launch: <a href="{{ repo }}/tree/main/engineering">design, pre-registration and reviews</a>,
  <a href="{{ repo }}/blob/main/docs/gate/calibration.md">calibration</a>{% if links.historical %},
  <a href="{{ links.historical }}">historical report</a>{% endif %}{% if links.inspect %},
  <a href="{{ links.inspect }}">AgentDojo report</a>{% endif %}.</li>
</ul>
</details>

<footer>
  <p>skillrc-gate is part of <a href="{{ repo }}">SkillRC</a> · page generated {{ today }} from the gate's result files by
  <code>scripts/gate/build_case_page.py</code></p>
</footer>
</main>
</body>
</html>
"""

if __name__ == "__main__":
    main()
