#!/usr/bin/env python3
"""Render the two result figures of the manuscript from the released result files.

Every number is read from data/summaries/ (the JSON files written by the analysis scripts);
nothing is re-estimated here.

  fig_readers   Reading cost I and content value R per reader: one pool and five readers
                (five-reader study), and 18 pools and three Qwen3 readers (crossover study).
  fig_contrasts The protocol contrasts of the crossover and generic-rule studies, each with
                its prediction and outcome.

Run:  python paper/figures/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SUM = ROOT / "data/summaries"

INK, MUTED, GRID = "#2F2F2F", "#6B6B6B", "#E4E4E4"
COST = "#8A8A8A"          # reading cost
VALUE = "#C2552F"         # content value
VALUE_LIGHT = "#EDB59F"   # one pool's content value
BAND = "#F1F1F1"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.edgecolor": "#BDBDBD",
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": MUTED,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(rel):
    return json.loads((SUM / rel).read_text())


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=240)
    plt.close(fig)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", length=2, width=0.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def check_ticklabel_gaps(fig, axes, min_gap_pt=3.0):
    """Fail if neighbouring x tick labels come closer than min_gap_pt (in points at the saved size)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in axes:
        boxes = sorted((lab.get_window_extent(r) for lab in ax.get_xticklabels() if lab.get_text()), key=lambda b: b.x0)
        for a, b in zip(boxes, boxes[1:]):
            gap = (b.x0 - a.x1) * 72 / fig.dpi
            assert gap >= min_gap_pt, f"tick labels {gap:.1f} pt apart"


def errpoint(ax, x, est, ci, color, filled, size=4.2):
    ax.plot([x, x], ci, color=color, lw=1.1, solid_capstyle="butt", zorder=3)
    ax.plot([x], [est], marker="o", ms=size, color=color, mfc=color if filled else "white",
            mew=1.1, zorder=4, ls="none")


def fig_readers():
    p2 = load("phase2_2026-10-10/phase2_2026-10-10_results.json")
    w1a = load("w1a_2026-10-13/w1a_2026-10-13_results.json")

    # Five-reader study: one pool (the focal pool), single runs (GPT-4o-mini: run 1, as in the protocol).
    b = p2["blocks"]
    blocks = {"Llama-3.1-8B": b["C134/LLAMA8B"], "GPT-4o-mini": b["run1_134/GPT"], "Qwen3-8B": b["run1_134/QWEN"],
              "GPT-4.1-mini": b["C134/GPT41MINI"], "Qwen3-14B": b["C134/QWEN14B"]}
    five = sorted(blocks.items(), key=lambda kv: kv[1]["means"]["none"])

    # Crossover study: 18 balanced pools, three Qwen3 readers, hierarchical intervals.
    readers = [("Qwen3-8B", "Q8"), ("Qwen3-14B", "Q14"), ("Qwen3-32B", "Q32")]
    bal = [p for p in w1a["pools"] if p not in ("GPTREF", "GREF")]
    assert len(bal) == 18

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.45), sharey=True,
                                 gridspec_kw={"width_ratios": [5.6, 3.0], "wspace": 0.03})
    off = 0.14
    for ax in (a1, a2):
        style(ax)
        ax.axhspan(-0.04, 0.04, color=BAND, zorder=0, lw=0)
        ax.axhline(0, color="#9A9A9A", lw=0.7, zorder=1)

    for i, (name, b) in enumerate(five):
        e = b["estimands"]
        errpoint(a1, i - off, e["I"]["mean"], e["I"]["ci95"], COST, False)
        errpoint(a1, i + off, e["R"]["mean"], e["R"]["ci95"], VALUE, True)
    a1.set_xticks(range(len(five)))
    a1.set_xticklabels([f"{n}\n{b['means']['none']:.2f}" for n, b in five], fontsize=7)
    a1.set_xlim(-0.5, len(five) - 0.5)
    a1.set_title("One pool, five readers", fontsize=8.5, color=INK, loc="left", pad=4)
    a1.set_ylabel("Change in success rate", fontsize=8)

    for i, (name, rd) in enumerate(readers):
        pooled = w1a["pooled"]
        rs = sorted(w1a["pools"][p][rd]["R"]["mean"] for p in bal)
        xs = [i + off + 0.20 + 0.012 * (k % 3) for k in range(len(rs))]
        a2.scatter(xs, rs, s=7, color=VALUE_LIGHT, edgecolor="none", zorder=2)
        errpoint(a2, i - off, pooled[f"I_bar_{rd}"]["mean"], pooled[f"I_bar_{rd}"]["ci95"], COST, False)
        errpoint(a2, i + off, pooled[f"R_bar_{rd}"]["mean"], pooled[f"R_bar_{rd}"]["ci95"], VALUE, True)
    a2.set_xticks(range(len(readers)))
    a2.set_xticklabels([f"{n}\n{w1a['none_mean'][rd]:.2f}" for n, rd in readers], fontsize=7)
    a2.set_xlim(-0.5, len(readers) - 0.5)
    a2.set_title("18 pools, three Qwen3 readers", fontsize=8.5, color=INK, loc="left", pad=4)
    a2.tick_params(axis="y", length=0)

    a1.set_ylim(-0.10, 0.155)
    a1.set_yticks([-0.08, -0.04, 0, 0.04, 0.08, 0.12])
    a1.set_yticklabels(["−0.08", "−0.04", "0", "+0.04", "+0.08", "+0.12"])
    handles = [Line2D([], [], marker="o", color=COST, mfc="white", mew=1.1, ms=4.2, lw=1.1,
                      label=r"Reading cost ($I$)"),
               Line2D([], [], marker="o", color=VALUE, ms=4.2, lw=1.1,
                      label=r"Content value ($R$)"),
               Line2D([], [], marker="o", color=VALUE_LIGHT, ms=3.2, lw=0, label=r"$R$ of one pool")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False,
               handletextpad=0.4, columnspacing=1.4)
    fig.text(0.005, -0.04, "Below each reader: its success rate without memory.", fontsize=7, color=MUTED)
    check_ticklabel_gaps(fig, (a1, a2))
    save(fig, "fig_readers")


def fig_contrasts():
    w1a = load("w1a_2026-10-13/w1a_2026-10-13_results.json")["pooled"]
    w1b = load("w1b_2026-10-13/w1b_2026-10-13_results.json")["pooled"]
    pr = load("poolrep_2026-10-12/poolrep_2026-10-12_results.json")
    rows = [
        ("What the content is", None, None, None),
        ("Experience rules − generic rules", w1b["E_bar"], "> 0", "holds"),
        ("Generic rules − unrelated text", w1b["G_bar"], "> 0", "fails"),
        ("Whose experience", None, None, None),
        ("Source: Qwen3-8B − GPT-4o-mini", w1a["src_Q8_minus_GPT"], "includes 0", "holds"),
        ("Distiller: Qwen3-32B − Qwen3-8B", w1a["D32_minus_D8"], "two-sided", "none detected"),
        ("Own-source advantage", w1a["Delta_own"], "includes 0, < +0.05", "unresolved"),
        ("Who reads", None, None, None),
        ("Reader: Qwen3-8B − Qwen3-32B", w1a["R_Q8_minus_R_Q32"], "> 0", "fails"),
        ("Which unrelated text", None, None, None),
        ("Astronomy − gardening control", w1b["dI_bar"], "|Δ| < 0.03, incl. 0", "fails"),
        ("Which pool (pool replication)", None, None, None),
        ("Mean of three replicate pools", pr["replicates"]["R_bar"], "> 0", "fails"),
        (f"Replicate 1 − replicate 3", pr["replicates"]["pairwise"]["Q32R1-Q32R3"], "includes 0", "fails"),
    ]
    n = len(rows)
    fig = plt.figure(figsize=(6.3, 3.05))
    ax = fig.add_axes([0.37, 0.10, 0.33, 0.84])
    style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.axvline(0, color="#9A9A9A", lw=0.7, zorder=1)
    ax.set_ylim(n - 0.4, -0.7)
    ax.set_xlim(-0.14, 0.095)
    ax.set_xticks([-0.12, -0.09, -0.06, -0.03, 0, 0.03, 0.06, 0.09])
    ax.set_xticklabels(["−0.12", "", "−0.06", "", "0", "", "+0.06", ""])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Difference in success rate (95% interval)", fontsize=7.5)
    tr = ax.get_yaxis_transform()
    fig.text(0.71, 0.955, "Predicted", fontsize=7.5, color=MUTED, ha="left")
    fig.text(0.885, 0.955, "Outcome", fontsize=7.5, color=MUTED, ha="left")
    for y, (label, est, pred, outcome) in enumerate(rows):
        if est is None:
            ax.text(-0.02, y, label, transform=tr, ha="right", va="center", fontsize=7.5,
                    color=INK, fontweight="bold")
            continue
        ax.text(-0.02, y, label, transform=tr, ha="right", va="center", fontsize=7.5, color=INK)
        lo, hi = est["ci95"]
        excl = lo > 0 or hi < 0
        col = VALUE if excl else INK
        ax.plot([lo, hi], [y, y], color=col, lw=1.2, zorder=3, solid_capstyle="butt")
        ax.plot([est["mean"]], [y], marker="o", ms=4, color=col, mfc=col if excl else "white", mew=1.1, zorder=4)
        ax.text(1.03, y, pred, transform=tr, ha="left", va="center", fontsize=7.5, color=INK)
        ax.text(1.56, y, outcome, transform=tr, ha="left", va="center", fontsize=7.5, color=INK)
    save(fig, "fig_contrasts")


if __name__ == "__main__":
    fig_readers()
    fig_contrasts()
    print("wrote", OUT / "fig_readers.pdf", OUT / "fig_contrasts.pdf")
