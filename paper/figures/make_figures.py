#!/usr/bin/env python3
"""Render the result figures from the frozen summary statistics.

Every number below is copied from the frozen result records of the study
(complete prompt-axis result, order-placebo gate, task-transfer taxonomy,
representation matrix, and the WebShop comparison). Per-episode logs are not
part of the public release, so the figures plot reported estimates and counts;
nothing is re-estimated here.  Run:  python paper/figures/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = Path(__file__).resolve().parent

# Pastel palette in the style of the MemoHarness figures.
LAV, LAV_D = "#C9C5E8", "#7F78BE"      # baseline / no memory
BLUE, BLUE_D = "#BDCFEA", "#5F86C2"    # GPT-4o-mini
MINT, MINT_D = "#BFE3D9", "#4FA58C"    # Qwen3-8B
SALM, SALM_D = "#F4C4B5", "#D9735A"    # SkillRC memory / highlight
GREY_BAND = "#F3F3F3"
INK, MUTED, GRID = "#3A3A3A", "#7A7A7A", "#E6E6E6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Droid Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": "#BDBDBD",
    "axes.linewidth": 0.7,
    "xtick.color": INK,
    "ytick.color": MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8,
    "axes.labelcolor": INK,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def _style(ax, ygrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", length=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _title(fig, text, legend=None, y=0.995, below=False):
    fig.text(0.01, y, text, ha="left", va="top", fontsize=11, color=INK)
    if legend:
        anchor, loc = ((0.0, y - 0.075), "upper left") if below else ((0.995, y + 0.012), "upper right")
        fig.legend(handles=[Patch(facecolor=c, label=lab) for lab, c in legend], loc=loc,
                   bbox_to_anchor=anchor, ncol=len(legend), frameon=False, fontsize=8,
                   handlelength=0.9, handleheight=0.9, columnspacing=1.1, handletextpad=0.4)


def _bands(ax, n, start=0):
    for i in range(n):
        if (i + start) % 2 == 1:
            ax.axvspan(i - 0.5, i + 0.5, color=GREY_BAND, zorder=0, lw=0)


def _r3(v, signed=False):
    """Round half up to three decimals, as in the paper text (0.1045 -> 0.105)."""
    from decimal import ROUND_HALF_UP, Decimal
    d = Decimal(str(v)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return (f"+{d}" if signed and d > 0 else str(d)).replace("-", "−")


def _label(ax, x, v, color, dy=0.006, signed=False):
    txt = _r3(v, signed)
    ax.text(x, v + (dy if v >= 0 else -dy), txt, ha="center", va="bottom" if v >= 0 else "top",
            fontsize=7, color=color)


def fig_success():
    """RQ1: equal-pair ALFWorld success without and with the experience memory."""
    groups = ["GPT-4o-mini", "Qwen3-8B"]
    none, mem = [0.5871, 0.7040], [0.7139, 0.6828]
    fig, ax = plt.subplots(figsize=(3.1, 2.05))
    _bands(ax, 2, start=1)
    w = 0.32
    for i in range(2):
        ax.bar(i - w / 2 - 0.02, none[i], w, color=LAV, zorder=2)
        ax.bar(i + w / 2 + 0.02, mem[i], w, color=SALM, zorder=2)
        _label(ax, i - w / 2 - 0.02, none[i], LAV_D)
        _label(ax, i + w / 2 + 0.02, mem[i], SALM_D)
    ax.set_xticks(range(2))
    ax.set_xticklabels(groups, fontweight="bold")
    ax.set_ylim(0.5, 0.76)
    ax.set_xlim(-0.55, 1.55)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    _style(ax)
    _title(fig, "ALFWorld Success (134 tasks)", [("No memory", LAV), ("SkillRC memory", SALM)], y=1.17, below=True)
    _save(fig, "fig_success")


def fig_pairs():
    """RQ2: memory gain on each unique prompt pair and the equal-pair estimate."""
    labels = ["Pair 01", "Pair 02", "Pair 12", "Mean"]
    gpt = [0.1119, 0.0970, 0.1716, 0.1269]
    qwen = [0.0075, -0.0821, 0.0112, -0.0211]
    cis = [(0.0721, 0.1841, BLUE_D), (-0.0622, 0.0199, MINT_D)]
    fig, ax = plt.subplots(figsize=(3.9, 2.05))
    _bands(ax, 4)
    w = 0.34
    xs = []
    for i in range(4):
        xg, xq = i - w / 2 - 0.02, i + w / 2 + 0.02
        ax.bar(xg, gpt[i], w, color=BLUE, zorder=2)
        ax.bar(xq, qwen[i], w, color=MINT, zorder=2)
        _label(ax, xg, gpt[i], BLUE_D, signed=True, dy=(0.1841 - gpt[i] + 0.006) if i == 3 else 0.006)
        _label(ax, xq, qwen[i], MINT_D, signed=True, dy=(qwen[i] + 0.0622 + 0.006) if i == 3 else 0.006)
        xs = [xg, xq]
    for x, (lo, hi, c) in zip(xs, cis):
        ax.plot([x, x], [lo, hi], color=c, lw=1.3, zorder=3)
        for yv in (lo, hi):
            ax.plot([x - 0.05, x + 0.05], [yv, yv], color=c, lw=1.3, zorder=3)
    ax.axhline(0, color="#9E9E9E", lw=0.8, zorder=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(-0.11, 0.215)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylabel("Paired gain", fontsize=8.5)
    _style(ax)
    _title(fig, "Memory Gain by Prompt Pair", [("GPT-4o-mini", BLUE), ("Qwen3-8B", MINT)], y=1.17, below=True)
    _save(fig, "fig_pairs")


def fig_placebo():
    """RQ3: no memory / order placebo / true memory on the 60-task gate, plus increments."""
    fig = plt.figure(figsize=(6.6, 2.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.62, 1.55], wspace=0.3)
    data = {"GPT-4o-mini": (0.5444, 0.5444 + 0.0889, 0.6806),
            "Qwen3-8B": (0.7278, 0.7278 - 0.1083, 0.6833)}
    for k, (name, vals) in enumerate(data.items()):
        ax = fig.add_subplot(gs[0, k])
        for i, (v, c, cd) in enumerate(zip(vals, [LAV, BLUE, SALM], [LAV_D, BLUE_D, SALM_D])):
            ax.bar(i, v, 0.64, color=c, zorder=2)
            _label(ax, i, v, cd)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["None", "Placebo", "True"], fontsize=8)
        ax.set_ylim(0.5, 0.77)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(name, fontsize=9, color=INK, fontweight="bold", pad=4)
        _style(ax)
    ax = fig.add_subplot(gs[0, 3])
    rows = [("GPT: true − none", 0.1361, 0.0750, 0.2000, BLUE, BLUE_D),
            ("GPT: placebo − none", 0.0889, 0.0250, 0.1528, BLUE, BLUE_D),
            ("GPT: true − placebo", 0.0472, -0.0194, 0.1167, BLUE, BLUE_D),
            ("Qwen: true − none", -0.0444, -0.1000, 0.0056, MINT, MINT_D),
            ("Qwen: placebo − none", -0.1083, -0.1639, -0.0528, MINT, MINT_D),
            ("Qwen: true − placebo", 0.0639, 0.0000, 0.1278, MINT, MINT_D),
            ("Gap (Qwen − GPT)", 0.0167, -0.0750, 0.1083, SALM, SALM_D)]
    ax.axvspan(-0.05, 0.05, color=GREY_BAND, zorder=0, lw=0)
    for j, (lab, est, lo, hi, c, cd) in enumerate(rows):
        y = len(rows) - 1 - j
        ax.barh(y, est, 0.62, color=c, zorder=2)
        ax.plot([lo, hi], [y, y], color=cd, lw=1.2, zorder=3)
        ax.text(0.30, y, _r3(est, True), va="center", ha="right", fontsize=7, color=cd)
    ax.axvline(0, color="#9E9E9E", lw=0.8, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.5)
    ax.set_xlim(-0.2, 0.30)
    ax.set_title("Paired effects (95% CI)", fontsize=9, color=INK, fontweight="bold", pad=4)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(fig, "Order-Placebo Gate (60 frozen tasks)",
           [("No memory", LAV), ("Order placebo", BLUE), ("True memory", SALM)], y=1.11)
    _save(fig, "fig_placebo")


def fig_rerun():
    """RQ3, same-window rerun (2026-10-10): none / scrambled placebo / coherent irrelevant / true memory on 60 tasks,
    plus the pre-registered increments. Every number is read from results/placebo/rerun_2026-10-10_results.json
    (copied to data/summaries/rerun_2026-10-10/)."""
    import json
    src = OUT.parents[1] / "data" / "summaries" / "rerun_2026-10-10" / "rerun_2026-10-10_results.json"
    R = json.loads(src.read_text())
    m, e = R["means"], R["estimands"]
    TEAL, TEAL_D = "#E6DDB8", "#A08A3C"   # coherent irrelevant control
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 1.3, 0.55, 1.6], wspace=0.3)
    for k, (name, ex) in enumerate((("GPT-4o-mini", "gpt"), ("Qwen3-8B", "qwen"))):
        ax = fig.add_subplot(gs[0, k])
        vals = [m[f"{ex}/{c}"] for c in ("none", "placebo", "irrelevant", "true")]
        for i, (v, c, cd) in enumerate(zip(vals, [LAV, BLUE, TEAL, SALM], [LAV_D, BLUE_D, TEAL_D, SALM_D])):
            ax.bar(i, v, 0.66, color=c, zorder=2)
            _label(ax, i, v, cd)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["None", "Scram-\nbled", "Irrel-\nevant", "True"], fontsize=7.2, linespacing=0.95)
        ax.set_ylim(0.5, 0.77)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(name, fontsize=9, color=INK, fontweight="bold", pad=4)
        _style(ax)
    ax = fig.add_subplot(gs[0, 3])
    rows = []
    for ex, c, cd, tag in (("gpt", BLUE, BLUE_D, "GPT"), ("qwen", MINT, MINT_D, "Qwen")):
        for key, lab in (("T", "true − none"), ("S", "true − scrambled"), ("R", "true − irrelevant"), ("C", "irrelevant − scrambled")):
            x = e[f"{key}_{ex}"]
            rows.append((f"{tag}: {lab}", x["mean"], x["ci95"][0], x["ci95"][1], c, cd))
    ax.axvspan(-0.05, 0.05, color=GREY_BAND, zorder=0, lw=0)
    for j, (lab, est, lo, hi, c, cd) in enumerate(rows):
        y = len(rows) - 1 - j
        ax.barh(y, est, 0.62, color=c, zorder=2)
        ax.plot([lo, hi], [y, y], color=cd, lw=1.2, zorder=3)
        ax.text(0.30, y, _r3(est, True), va="center", ha="right", fontsize=7, color=cd)
    ax.axvline(0, color="#9E9E9E", lw=0.8, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.3)
    ax.set_xlim(-0.16, 0.30)
    ax.set_title("Paired effects (95% CI)", fontsize=9, color=INK, fontweight="bold", pad=4)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    _title(fig, "Rerun, 60 tasks",
           [("No memory", LAV), ("Scrambled placebo", BLUE), ("Coherent, irrelevant", TEAL), ("True memory", SALM)], y=1.11)
    _save(fig, "fig_rerun")


def fig_transport():
    """RQ4: task-level transport map and GPT gain by task type."""
    fig = plt.figure(figsize=(6.2, 2.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    labels = ["hurt", "neutral", "help"]
    counts = {("hurt", "hurt"): 6, ("hurt", "neutral"): 11, ("hurt", "help"): 14,
              ("neutral", "hurt"): 11, ("neutral", "neutral"): 35, ("neutral", "help"): 31,
              ("help", "hurt"): 5, ("help", "neutral"): 12, ("help", "help"): 9}
    assert sum(counts.values()) == 134
    grid = [[counts[(q, g)] for g in labels] for q in labels]
    cmap = LinearSegmentedColormap.from_list("mauve", ["#FBF8F9", "#F4E3E7", "#E3C2CA", "#C98E9C"])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=38, origin="lower")
    for i, q in enumerate(labels):
        for j, g in enumerate(labels):
            ax.text(j, i, str(counts[(q, g)]), ha="center", va="center", fontsize=9.5, color=INK,
                    fontweight="bold" if g == "help" else "normal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels)
    ax.set_xticks([0.5, 1.5], minor=True)
    ax.set_yticks([0.5, 1.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="both", length=0)
    ax.set_xlabel("GPT-4o-mini memory effect", fontsize=8.5)
    ax.set_ylabel("Qwen3-8B memory effect", fontsize=8.5)
    ax.set_title("Tasks by sign of paired effect", fontsize=9, color=INK, fontweight="bold", pad=4)

    ax = fig.add_subplot(gs[0, 1])
    types = ["puttwo", "heat", "clean", "examine", "put", "cool"]
    gains = [0.2549, 0.2464, 0.2151, 0.1852, 0.0833, -0.1111]
    _bands(ax, len(types))
    for i, g in enumerate(gains):
        ax.bar(i, g, 0.6, color=SALM if g > 0 else LAV, zorder=2)
        _label(ax, i, g, SALM_D if g > 0 else LAV_D, signed=True)
    ax.axhline(0, color="#9E9E9E", lw=0.8)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, fontsize=8)
    ax.set_ylim(-0.16, 0.31)
    ax.set_xlim(-0.55, len(types) - 0.45)
    ax.set_title("GPT-4o-mini gain by task type", fontsize=9, color=INK, fontweight="bold", pad=4)
    _style(ax)
    _title(fig, "Task-Level Transfer (134 ALFWorld tasks)", None, y=1.09)
    _save(fig, "fig_transport")


def fig_webshop():
    """RQ5: train-disjoint WebShop comparison with local Qwen3-8B."""
    fig = plt.figure(figsize=(6.2, 2.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    vals = [(0.1944, 0.1045), (0.09, 0.07)]
    _bands(ax, 2, start=1)
    w = 0.32
    for i, (a, b) in enumerate(vals):
        ax.bar(i - w / 2 - 0.02, a, w, color=LAV, zorder=2)
        ax.bar(i + w / 2 + 0.02, b, w, color=SALM, zorder=2)
        _label(ax, i - w / 2 - 0.02, a, LAV_D, dy=0.004)
        _label(ax, i + w / 2 + 0.02, b, SALM_D, dy=0.004)
    ax.set_xticks(range(2))
    ax.set_xticklabels(["Reward", "Exact success"], fontweight="bold")
    ax.set_ylim(0, 0.24)
    ax.set_xlim(-0.55, 1.55)
    _style(ax)
    ax = fig.add_subplot(gs[0, 1])
    left = 0
    for name, n, c, cd in [("Declined", 20, SALM, SALM_D), ("Tied", 72, "#E8E8E8", MUTED),
                           ("Improved", 8, MINT, MINT_D)]:
        ax.barh(0, n, left=left, color=c, height=0.5, edgecolor="white", linewidth=2)
        ax.text(left + n / 2, 0, str(n), ha="center", va="center", fontsize=8.5, color=INK)
        ax.text(left + n / 2, 0.36, name, ha="center", va="bottom", fontsize=7.5, color=cd)
        left += n
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.4, 0.75)
    ax.set_yticks([])
    ax.set_xlabel("Sessions (paired reward change, memory vs. none)", fontsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    _title(fig, "WebShop (Qwen3-8B, 100 train-disjoint sessions)",
           [("No memory", LAV), ("SkillRC memory", SALM)], y=1.12)
    _save(fig, "fig_webshop")


if __name__ == "__main__":
    for f in (fig_success, fig_pairs, fig_placebo, fig_rerun, fig_transport, fig_webshop):
        f()
    print("wrote", sorted(p.name for p in OUT.glob("fig_*.pdf")))
