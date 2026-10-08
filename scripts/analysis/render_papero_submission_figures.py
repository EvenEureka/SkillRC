#!/usr/bin/env python3
"""Render publication-sized vector figures for Paper O from raw paired data."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ASSETS = ROOT.parent / "ARR" / "PaperO_assets"
OUT = ROOT.parent / "ARR" / "PaperO_submission_figures"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#6B7280"
LIGHT = "#D1D5DB"
INK = "#1F2937"


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans 3", "DejaVu Sans"],
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "axes.edgecolor": "#9CA3AF",
        "axes.linewidth": 0.7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.4,
        "figure.dpi": 180,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_jsonl(run_id: str, seeds: tuple[int, ...] | None = None) -> dict[tuple[int, int], dict]:
    rows = {}
    for line in (RESULTS / f"{run_id}.episodes.jsonl").read_text().splitlines():
        row = json.loads(line)
        seed = int(row["seed"])
        if seeds is not None and seed not in seeds:
            continue
        rows[(seed, int(row["task_id"]))] = row
    return rows


def stack_conditions(stack: str) -> tuple[dict, dict]:
    if stack == "GPT-4o-mini":
        base = load_jsonl("R008_nomem_s3", (0, 1, 2)) | load_jsonl("R013_nomem_seed5", (5,))
        true = load_jsonl("R012_expel_bm25", (0, 1, 2)) | load_jsonl("R013_expel_bm25_seed5", (5,))
    else:
        base = load_jsonl("Q_nomem_s3", (0, 1, 2)) | load_jsonl("Q_nomem_seed5", (5,))
        true = load_jsonl("Q_expel_bm25_s3", (0, 1, 2)) | load_jsonl("Q_expel_bm25_seed5", (5,))
    return base, true


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color="#E5E7EB", linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.11, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", color=INK)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def figure_interaction() -> None:
    pair_seeds = (("01", (5,)), ("02", (0,)), ("12", (1, 2)))
    fig = plt.figure(figsize=(6.9, 2.72), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.25])

    aggregate = {}
    for col, (stack, color) in enumerate((("GPT-4o-mini", BLUE), ("Qwen3-8B", ORANGE))):
        ax = fig.add_subplot(gs[0, col])
        base, true = stack_conditions(stack)
        pair_values = []
        label_offsets = {"01": 0.006, "02": 0.003, "12": -0.006}
        for pair, seeds in pair_seeds:
            task_ids = sorted(t for s, t in base if s == seeds[0])
            b = np.mean([base[(seed, t)]["success"] for seed in seeds for t in task_ids])
            m = np.mean([true[(seed, t)]["success"] for seed in seeds for t in task_ids])
            pair_values.append((b, m))
            ax.plot([0, 1], [b, m], color=LIGHT, linewidth=1.2, zorder=2)
            ax.scatter([0, 1], [b, m], s=19, facecolor="white", edgecolor=color,
                       linewidth=0.9, zorder=3)
            ax.text(1.055, m + label_offsets[pair], pair, va="center", color=GRAY, fontsize=6.8)
        means = np.mean(np.asarray(pair_values), axis=0)
        aggregate[stack] = means
        ax.plot([0, 1], means, color=color, linewidth=2.8, zorder=4)
        ax.scatter([0, 1], means, s=43, color=color, edgecolor="white", linewidth=0.8, zorder=5)
        ax.text(0, means[0] + 0.025, f"{means[0]:.3f}", ha="center", color=color,
                fontweight="bold")
        ax.text(1, means[1] + 0.025, f"{means[1]:.3f}", ha="center", color=color,
                fontweight="bold")
        ax.set_xticks([0, 1], ["No memory", "True skills"])
        ax.set_ylim(0.50, 0.79)
        ax.set_yticks(np.arange(0.5, 0.81, 0.1))
        ax.set_ylabel("Success rate" if col == 0 else "")
        ax.set_title(stack, loc="left", fontweight="semibold", color=INK, pad=5)
        clean_axis(ax)
        if col == 1:
            ax.tick_params(labelleft=False)
        panel_label(ax, "a" if col == 0 else "")

    ax = fig.add_subplot(gs[0, 2])
    with (ASSETS / "table7_complete_prompt_axis.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    labels = ["GPT skill effect", "Qwen skill effect", "Qwen - GPT interaction"]
    selected = [rows[0], rows[1], rows[2]]
    y = np.array([2, 1, 0])
    est = np.array([float(row["estimate"]) for row in selected])
    lo = np.array([float(row["task_boot_ci95_lo"]) for row in selected])
    hi = np.array([float(row["task_boot_ci95_hi"]) for row in selected])
    colors = [BLUE, ORANGE, INK]
    ax.axvline(0, color="#9CA3AF", linewidth=0.9, zorder=1)
    for yi, e, l, h, color in zip(y, est, lo, hi, colors):
        ax.plot([l, h], [yi, yi], color=color, linewidth=2.0, zorder=2)
        ax.scatter(e, yi, s=38, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(h + 0.012, yi, f"{e:+.3f}", va="center", color=color, fontsize=7.2)
    ax.set_yticks(y, labels)
    ax.set_xlim(-0.25, 0.24)
    ax.set_xlabel("Paired success difference (95% task-bootstrap CI)")
    ax.set_title("Interaction-first estimate", loc="left", fontweight="semibold", color=INK, pad=5)
    clean_axis(ax, "x")
    panel_label(ax, "b")
    save(fig, "fig1_executor_interaction")


def figure_task_transport() -> None:
    stacks = {stack: stack_conditions(stack) for stack in ("GPT-4o-mini", "Qwen3-8B")}
    task_ids = sorted({task_id for _, task_id in stacks["GPT-4o-mini"][0]})
    task_types = {task_id: stacks["GPT-4o-mini"][0][(0, task_id)]["task_type"] for task_id in task_ids}
    effects = {}
    for stack, (base, true) in stacks.items():
        effects[stack] = {
            task_id: np.mean([
                np.mean([true[(seed, task_id)]["success"] - base[(seed, task_id)]["success"]
                         for seed in seeds])
                for seeds in ((5,), (0,), (1, 2))
            ])
            for task_id in task_ids
        }
    positions = Counter((round(effects["GPT-4o-mini"][t], 4), round(effects["Qwen3-8B"][t], 4))
                        for t in task_ids)

    fig, ax = plt.subplots(figsize=(3.25, 3.15), constrained_layout=True)
    ax.axhline(0, color="#9CA3AF", linewidth=0.85, zorder=1)
    ax.axvline(0, color="#9CA3AF", linewidth=0.85, zorder=1)
    max_count = max(positions.values())
    for (x, y), count in sorted(positions.items()):
        size = 34 + 260 * np.sqrt(count / max_count)
        if x > 0 and y <= 0:
            color = BLUE
        elif y > 0 and x <= 0:
            color = ORANGE
        elif x > 0 and y > 0:
            color = GREEN
        elif x < 0 and y < 0:
            color = "#A04444"
        else:
            color = GRAY
        ax.scatter(x, y, s=size, color=color, alpha=0.82, edgecolor="white", linewidth=1.0, zorder=3)
        ax.text(x, y, str(count), color="white", ha="center", va="center",
                fontsize=6.7, fontweight="bold", zorder=4)

    category_counts = {
        "both help": sum(effects["GPT-4o-mini"][t] > 0 and effects["Qwen3-8B"][t] > 0 for t in task_ids),
        "GPT only/opposed": sum(effects["GPT-4o-mini"][t] > 0 and effects["Qwen3-8B"][t] <= 0 for t in task_ids),
        "Qwen only/opposed": sum(effects["Qwen3-8B"][t] > 0 and effects["GPT-4o-mini"][t] <= 0 for t in task_ids),
        "both hurt": sum(effects["GPT-4o-mini"][t] < 0 and effects["Qwen3-8B"][t] < 0 for t in task_ids),
    }
    quadrant_font = 5.8
    ax.text(0.97, 0.97, f"both +  n={category_counts['both help']}", transform=ax.transAxes,
            ha="right", va="top", color=GREEN, fontsize=quadrant_font)
    ax.text(0.03, 0.97, f"Qwen +  n={category_counts['Qwen only/opposed']}",
            transform=ax.transAxes, ha="left", va="top", color=ORANGE, fontsize=quadrant_font)
    ax.text(0.97, 0.03, f"GPT +  n={category_counts['GPT only/opposed']}",
            transform=ax.transAxes, ha="right", va="bottom", color=BLUE, fontsize=quadrant_font)
    ax.text(0.03, 0.03, f"both -  n={category_counts['both hurt']}", transform=ax.transAxes,
            ha="left", va="bottom", color="#A04444", fontsize=quadrant_font)
    ticks = np.array([-1, -2/3, -1/3, 0, 1/3, 2/3, 1])
    tick_labels = ["-1", "-2/3", "-1/3", "0", "1/3", "2/3", "1"]
    ax.set_xticks(ticks, tick_labels)
    ax.set_yticks(ticks, tick_labels)
    ax.tick_params(axis="x", labelsize=5.8)
    ax.set_xlim(-1.11, 1.11)
    ax.set_ylim(-1.11, 1.11)
    ax.set_aspect("equal")
    ax.set_xlabel("GPT-4o-mini true-memory effect")
    ax.set_ylabel("Qwen3-8B true-memory effect")
    ax.grid(color="#EEF0F3", linewidth=0.65)
    ax.set_axisbelow(True)
    panel_label(ax, "")
    save(fig, "fig2_task_transport_map")


def figure_webshop() -> None:
    base = load_jsonl("PAPERO_WS_Q_nomem_eval100")
    true = load_jsonl("PAPERO_WS_Q_expel_bm25_eval100")
    task_ids = sorted(task for _, task in base)
    delta = np.array([float(true[(0, task)]["reward"]) - float(base[(0, task)]["reward"])
                      for task in task_ids])
    ordered = np.sort(delta)
    colors = np.where(ordered < 0, ORANGE, np.where(ordered > 0, BLUE, LIGHT))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.55),
                                  gridspec_kw={"width_ratios": [2.2, 1]}, constrained_layout=True)
    ax.axhline(0, color="#9CA3AF", linewidth=0.9)
    ax.vlines(np.arange(len(ordered)), 0, ordered, color=colors, linewidth=0.8, alpha=0.8)
    ax.scatter(np.arange(len(ordered)), ordered, s=12, color=colors, edgecolor="none", zorder=3)
    ax.set_xlabel("Held-out WebShop tasks, ordered by paired change")
    ax.set_ylabel("Memory - no-memory reward")
    ax.set_xlim(-2, len(ordered) + 1)
    ax.set_xticks([0, 24, 49, 74, 99], [1, 25, 50, 75, 100])
    clean_axis(ax)
    panel_label(ax, "a")

    with (ASSETS / "table12_webshop_crossenv.csv").open() as handle:
        reward = next(csv.DictReader(handle))
    mean = float(reward["paired_delta"])
    lo = float(reward["task_boot_ci95_lo"])
    hi = float(reward["task_boot_ci95_hi"])
    ax2.axvline(0, color="#9CA3AF", linewidth=0.9)
    ax2.plot([lo, hi], [0, 0], color=ORANGE, linewidth=2.3)
    ax2.scatter(mean, 0, s=48, color=ORANGE, edgecolor="white", linewidth=0.8, zorder=3)
    ax2.set_xlim(-0.19, 0.07)
    ax2.set_ylim(-0.65, 0.65)
    ax2.set_yticks([])
    ax2.set_xlabel("Mean paired reward change\n(95% task-bootstrap CI)")
    ax2.text(mean, 0.22, f"{mean:+.3f}\n[{lo:+.3f}, {hi:+.3f}]", ha="center",
             va="bottom", color=ORANGE, fontweight="semibold")
    ax2.text(0.03, 0.96, "8 improve / 20 decline / 72 tie", transform=ax2.transAxes,
             ha="left", va="top", color=INK, fontsize=7.2)
    clean_axis(ax2, "x")
    panel_label(ax2, "b")
    save(fig, "fig4_webshop_paired_shift")


def figure_placebo_if_available() -> bool:
    path = ASSETS / "semantic_placebo_estimates.csv"
    if not path.exists():
        return False
    decision_path = ASSETS / "semantic_placebo_decision.json"
    decision = json.loads(decision_path.read_text())
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    semantic = [row for row in rows if row["contrast"] in
                ("semantic_increment", "semantic_increment_interaction")]
    labels = ["GPT: true - placebo", "Qwen: true - placebo", "Qwen - GPT interaction"]
    semantic.sort(key=lambda row: {"gpt": 0, "qwen": 1, "qwen_minus_gpt": 2}[row["executor"]])
    est = np.array([float(row["estimate"]) for row in semantic])
    lo = np.array([float(row["ci95_lo"]) for row in semantic])
    hi = np.array([float(row["ci95_hi"]) for row in semantic])

    fig = plt.figure(figsize=(6.9, 2.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.3])
    ax0 = fig.add_subplot(gs[0, 0])
    condition_order = ("baseline", "placebo", "true")
    condition_labels = ("No memory", "Order\nplacebo", "True skills")
    x = np.arange(3)
    for executor, label, color in (("gpt", "GPT-4o-mini", BLUE), ("qwen", "Qwen3-8B", ORANGE)):
        pair_means = decision["prompt_pair_means"][executor]
        for pair in ("01", "02", "12"):
            values = [pair_means[condition][pair] for condition in condition_order]
            ax0.plot(x, values, color=LIGHT, linewidth=1.0, zorder=1)
            ax0.scatter(x, values, s=15, facecolor="white", edgecolor=color,
                        linewidth=0.7, zorder=2)
        means = [decision["condition_means"][executor][condition] for condition in condition_order]
        ax0.plot(x, means, color=color, linewidth=2.4, label=label, zorder=3)
        ax0.scatter(x, means, s=38, color=color, edgecolor="white", linewidth=0.7, zorder=4)
        offset = 0.014 if executor == "gpt" else -0.022
        for xi, mean in zip(x, means):
            ax0.text(xi, mean + offset, f"{mean:.3f}", color=color, ha="center",
                     va="center", fontsize=6.6, fontweight="semibold")
    ax0.set_xticks(x, condition_labels)
    ax0.set_xlim(-0.18, 2.36)
    ax0.set_ylim(0.45, 0.82)
    ax0.set_ylabel("Success rate on balanced 60-task gate")
    ax0.legend(frameon=False, loc="lower left")
    ax0.set_title("Placebo reproduces the direction", loc="left", fontweight="semibold",
                  color=INK, pad=5)
    clean_axis(ax0)
    panel_label(ax0, "a")

    ax = fig.add_subplot(gs[0, 1])
    y = np.array([2, 1, 0])
    ax.axvspan(-0.05, 0.05, color="#F3F4F6", zorder=0)
    ax.axvline(0, color="#9CA3AF", linewidth=0.9)
    for yi, e, l, h, color in zip(y, est, lo, hi, (BLUE, ORANGE, INK)):
        ax.plot([l, h], [yi, yi], color=color, linewidth=2.2)
        ax.scatter(e, yi, s=42, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(h + 0.006, yi, f"{e:+.3f}", va="center", color=color, fontsize=7.0)
    ax.set_yticks(y, labels)
    ax.set_xlim(-0.085, 0.155)
    ax.set_xlabel("True skills - order placebo\n(95% stratified task-bootstrap CI)")
    ax.set_title("Coherent-order increments do not differ by stack", loc="left",
                 fontweight="semibold", color=INK, pad=5)
    clean_axis(ax, "x")
    panel_label(ax, "b")
    save(fig, "fig3_semantic_placebo")
    return True


def main() -> None:
    configure()
    figure_interaction()
    figure_task_transport()
    figure_webshop()
    has_placebo = figure_placebo_if_available()
    print(json.dumps({"output": str(OUT), "placebo_figure": has_placebo}, indent=2))


if __name__ == "__main__":
    main()
