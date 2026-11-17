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
    return "–" if x is None else f"{x:+.{d}f}"


def pct(x):
    return "–" if x is None else f"{100 * x:+.0f}%"


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


def release_findings(r):
    """Sentences chosen by state; every number from result.json."""
    if r is None:
        return None
    out = []
    t1, t2 = r["targets"].get("C1_ship_pub"), r["targets"].get("C2_ship_repaired")
    for t, what in ((t1, "Shipping the published memory (panel T, sessions 0–499)"),
                    (t2, "Shipping the repaired memory (panel D, sessions 520–1019)")):
        if not t or not t["details"].get("primary"):
            continue
        est, lo, hi, p = prim(t)
        tok = guard(t, "prompt_tokens")[0]
        lat = guard(t, "latency_s")[0]
        comp = {c["kind"] + ":" + c["name"]: c["state"] for c in t.get("components", [])}
        out.append(f"<strong>{what}: {badge(t['state'])}.</strong> Mean score {p['candidate_mean']:.3f} with the memory "
                   f"vs {p['baseline_mean']:.3f} without (Δ {f(est)} [{f(lo)}, {f(hi)}]); prompt tokens per episode "
                   f"{pct(tok)}, p95 episode latency {pct(lat)}. Components: "
                   + ", ".join(f"{html.escape(k)} {v}" for k, v in comp.items()) + ".")
    loc = r["claims"].get("localization")
    if loc:
        parts = [r["contrasts"][c] for c in loc["all_of"]]
        desc = "; ".join(f"{c} {f(r['contrasts'][c]['estimate']['estimate'])} "
                         f"[{f(r['contrasts'][c]['estimate']['lower'])}, {f(r['contrasts'][c]['estimate']['upper'])}]"
                         for c in loc["all_of"] if r["contrasts"][c].get("estimate"))
        if not loc["evaluable"]:
            verdict = "could not be evaluated (" + "; ".join(sorted({x for p_ in parts for x in p_.get("reasons", [])})) + ")"
        elif loc["holds"]:
            verdict = ("<strong>holds</strong>: replacing the six “keep searching” rules beats the published memory and "
                       "each of three matched edits of other rules")
        else:
            failed = [c for c in loc["all_of"] if not r["contrasts"][c].get("supported")]
            verdict = f"<strong>does not hold</strong>: not supported for {', '.join(failed)}"
        out.append(f"<strong>Localization</strong> (joint claim, every part must hold): {verdict}. {desc}.")
    return out


def replay(r, target):
    if not r or target not in r["targets"]:
        return None
    reps = r["targets"][target]["details"].get("replays") or []
    if not reps:
        return None
    rp = reps[0]
    return {"task": rp["task_id"], "type": rp["task_type"], "why": rp["why"], "div": rp["first_divergent_step"],
            "base": rp["baseline"], "cand": rp["candidate"]}


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
                 "n": ins["details"]["n_tasks"], "mde": ins["details"]["power"]["mde_80_two_sided_0.05"],
                 "cand": p["candidate_mean"], "base": p["baseline_mean"]}
    from jinja2 import Template
    page = Template(TEMPLATE).render(
        chart=chart(res), hist=hist, ins=ins_d, rel=release_findings(res["release"]),
        rel32=release_findings(res["release32"]) if res["release32"] else None,
        rep=replay(res["release"], "C1_ship_pub") or replay(res["historical"], "memory_given_label"),
        links=links, repo=REPO, today=date.today().isoformat(), f=f, pct=pct, badge=badge)
    (OUT / "index.html").write_text(page)
    print("wrote", OUT / "index.html", "reports:", sorted(links))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>skillrc-gate | Release checks for agent changes</title>
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
  figure { margin: 22px 0 8px; } .frame { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 10px; }
  figcaption { font-size: 15px; color: var(--text-2); margin-top: 8px; line-height: 1.5; }
  svg.iv { width:100%; height:auto; font-family: var(--sans); } svg.iv .lab, svg.iv .reflab { fill: var(--text-2); font-size: 12px; }
  svg.iv .bar { stroke: var(--text); stroke-width: 3; stroke-linecap: round } svg.iv .dot { fill: var(--text) }
  svg.iv .bar.block{stroke:var(--block)} svg.iv .dot.block{fill:var(--block)} svg.iv .bar.pass{stroke:var(--pass)} svg.iv .dot.pass{fill:var(--pass)}
  svg.iv .bar.inconclusive{stroke:var(--inc)} svg.iv .dot.inconclusive{fill:var(--inc)} svg.iv .bar.descriptive{stroke:var(--desc)} svg.iv .dot.descriptive{fill:var(--desc)}
  svg.iv .bar.invalid{stroke:var(--inv)} svg.iv .dot.invalid{fill:var(--inv)}
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
</style>
</head>
<body>
<main>
<header>
  <h1>skillrc-gate</h1>
  <p class="subtitle">A release check for agent changes, computed from episode logs</p>
  <div class="links">
    <a class="btn primary" href="{{ repo }}/blob/main/docs/gate/README.md">Docs</a>
    <a class="btn" href="{{ repo }}/tree/main/skillrc/gate">Code</a>
    {% if links.release %}<a class="btn" href="{{ links.release }}">Release-check report</a>{% endif %}
    {% if links.historical %}<a class="btn" href="{{ links.historical }}">Historical report</a>{% endif %}
    {% if links.inspect %}<a class="btn" href="{{ links.inspect }}">Inspect report</a>{% endif %}
  </div>
</header>

<h2>The question</h2>
<p class="question">An agent team adds a distilled memory, edits a prompt, or turns on a harness feature. Before shipping: is the
new version better or worse on a fixed task set, by how much, at what cost, and if it is worse, which part of the change
is responsible? A single before-and-after number cannot answer that, especially when several changes ship together.</p>
<p>The gate reads the logs of both versions on the same tasks and returns one verdict per declared change:
{{ badge("PASS") }} {{ badge("BLOCK") }} {{ badge("INCONCLUSIVE") }} {{ badge("INVALID") }} {{ badge("DESCRIPTIVE") }}.
The command's exit code is the verdict, so a CI job can stop a release.</p>

<figure>
  <div class="frame">{{ chart | safe }}</div>
  <figcaption>Change in mean score (candidate minus baseline) with 95% paired intervals, coloured by verdict. The dashed
  line is the margin: a change passes only if its whole interval clears −0.05, and is blocked if its whole interval lies
  below it. Rows are release targets from the three cases below.</figcaption>
</figure>

<h2>What the gate found</h2>
<ol class="findings">
  {% if hist %}
  <li><strong>A bundled release hid a regression.</strong> On 500 WebShop sessions with Qwen3-8B, shipping a page label and a
  distilled memory together looked like a gain over the old release (Δ {{ f(hist.bundle_vs_old.est) }}
  [{{ f(hist.bundle_vs_old.lo) }}, {{ f(hist.bundle_vs_old.hi) }}]). Split into declared interventions, the label alone
  gave {{ f(hist.label_only.est) }} and the memory, given the label, {{ f(hist.memory_given_label.est) }}
  [{{ f(hist.memory_given_label.lo) }}, {{ f(hist.memory_given_label.hi) }}] while adding {{ pct(hist.memory_given_label.tok) }}
  prompt tokens per episode. These logs predate the provenance contract, so the gate reports them as
  {{ badge("DESCRIPTIVE") }}: real statistics, no release decision.</li>
  {% endif %}
  <li><strong>The pre-registered release check.</strong>
  {% if rel %}{% for s in rel %}<br>{{ s | safe }}{% endfor %}
  {% else %}<span class="pending">Running now: 13 arm-runs in two execution blocks on two GPUs (published memory vs none on
  sessions 0–499; repaired memory, the published memory and three matched comparator edits on held-out sessions
  520–1019). This section is generated from the gate's result file when the runs finish.</span>{% endif %}
  {% if rel32 %}<br><strong>Qwen3-32B extension:</strong>{% for s in rel32 %}<br>{{ s | safe }}{% endfor %}{% endif %}</li>
  {% if ins %}
  <li><strong>Not enough evidence is also a verdict.</strong> Through Inspect AI on 96 AgentDojo tasks (two epochs per arm),
  a block of six tips in the system prompt scored {{ "%.3f" | format(ins.cand) }} vs {{ "%.3f" | format(ins.base) }}
  (Δ {{ f(ins.est) }} [{{ f(ins.lo) }}, {{ f(ins.hi) }}]): non-inferior within the margin, but its token cost
  ({{ pct(ins.tok[0]) }}, interval up to {{ pct(ins.tok[2]) }}) could neither be shown under the +30% tolerance nor shown to
  exceed it. The gate returns {{ badge(ins.state) }} and says why; at {{ ins.n }} tasks the smallest detectable score change is
  about {{ "%.2f" | format(ins.mde) }}.</li>
  {% endif %}
</ol>

{% if rep %}
<h2>From verdict to trajectory</h2>
<p class="note">Task {{ rep.task }} ({{ rep.type }}), chosen by a fixed rule ({{ rep.why }}), not by hand. Highlighted: the
first step where the two versions act differently.</p>
<div class="replay">
  {% for side, e in (("without the memory", rep.base), ("with the memory", rep.cand)) %}
  <div><p class="note"><strong>{{ side }}</strong> · score {{ "%.2f" | format(e.score) }} · {{ e.steps }} steps</p>
  <div class="traj">{% for s in e.trajectory %}<div class="{{ 'div' if s.step == rep.div else '' }}"><span class="n">{{ s.step }}</span>{{ s.action }}</div>{% endfor %}</div></div>
  {% endfor %}
</div>
{% endif %}

<h2>How it works</h2>
<p>The gate does not run agents; it reads their logs (SkillRC runner, Inspect AI, or a plain JSONL schema). Each stage
passes one thing to the next:</p>
<div class="scroll"><table>
<tr><th>stage</th><th>what it does</th><th>passes on</th></tr>
<tr><td>ingest</td><td>parses each arm's logs into episodes and a run manifest; all runs of an arm must share one fingerprint</td><td>arms (a broken arm fails only its own targets)</td></tr>
<tr><td>fingerprint check</td><td>the two versions may differ only in the declared interventions; model revision, scorer, serving, environment and every other file hash must match; missing provenance means no decision</td><td>undeclared differences (any → INVALID)</td></tr>
<tr><td>pairing</td><td>builds the expected task × repetition grid from a frozen task registry, picks each cell's scored attempt, counts missing and failed cells</td><td>task-aligned table</td></tr>
<tr><td>statistics</td><td>task-clustered, stratified bootstrap; order-statistic intervals that agree exactly with the p-values; Holm across guardrails; missing cells bound the score</td><td>estimates and intervals</td></tr>
<tr><td>verdict</td><td>BLOCK if the score interval lies below −δ or a cost limit is confirmed broken; PASS if the score clears −δ and every guardrail stays under its limit; else INCONCLUSIVE</td><td>one verdict per change, with reasons</td></tr>
<tr><td>report</td><td>JSON for machines, Markdown for CI, HTML with behaviour shifts and paired replays</td><td>files and exit code</td></tr>
</table></div>
<pre>pip install -e .
skillrc-gate report examples/webshop_bundled_release/gate.yaml --out /tmp/gate_demo   # real logs, no GPU
open /tmp/gate_demo/report.html</pre>

<h2>Engineering</h2>
<p>The producer side is built for unattended GPU runs: every attempt is journaled before it starts; a run resumes only
under the same fingerprint; interrupted or truncated records are recovered; a run stops itself after repeated
infrastructure failures so its servers can be restarted; each job in a chain checks that its predecessors succeeded
(hash-verified markers); state files are published atomically. The contract was reviewed in eleven rounds with an
independent reviewer before launch (<a href="{{ repo }}/tree/main/engineering">design, pre-registration and review record</a>);
94 tests and a calibration study (<a href="{{ repo }}/blob/main/docs/gate/calibration.md">calibration.md</a>) back the
decision rules.</p>

<h2>Limits</h2>
<p class="note">WebShop here is the 1,000-product version with generated goals, fashion-heavy test panels and a 15-step
budget; readers are open Qwen3 models. Intervals describe task-sampling uncertainty given the observed execution
windows, not future windows. The error budget is nominal: in simulation, the chance of a PASS when the true change sits
exactly at the margin was 0.018–0.041 against a nominal 0.025. The margin (0.05) and the cost tolerances (+30%) are
illustrative policies fixed before the runs, not recommendations. Which rule caused a loss is tested only through
declared ablation arms, never inferred automatically.</p>

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
