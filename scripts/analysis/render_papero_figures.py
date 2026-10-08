#!/usr/bin/env python3
"""Render dependency-free SVG figures and analysis tables for Paper O.

The renderer deliberately uses only the Python standard library.  It consumes
the CSV assets exported by ``export_papero_assets.py`` and the episode JSONL
files, so figures can be regenerated without another model/API call.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ASSETS = ROOT.parent / "ARR" / "PaperO_assets"
OUT = ROOT.parent / "ARR" / "PaperO_figures"

COLORS = {
    "navy": "#17324D",
    "blue": "#2878B5",
    "orange": "#D95F02",
    "teal": "#2A9D8F",
    "red": "#C44747",
    "gold": "#C7921E",
    "gray": "#65727E",
    "light": "#E9EEF2",
    "grid": "#D5DDE4",
    "ink": "#17212B",
    "white": "#FFFFFF",
}

VARIANT_LABELS = {
    "expel_bm25": "Insight + BM25",
    "raw_dense": "Raw + dense",
    "skillos_dense": "Markdown + dense",
    "raw_bm25": "Raw + BM25",
    "expel_dense": "Insight + dense",
    "skillos_bm25": "Markdown + BM25",
    "nomem": "No memory",
}

VARIANT_COLORS = {
    "expel_bm25": COLORS["orange"],
    "raw_dense": COLORS["blue"],
    "skillos_dense": COLORS["teal"],
    "raw_bm25": "#6A9BC2",
    "expel_dense": "#E38C4A",
    "skillos_bm25": "#65B8AD",
    "nomem": COLORS["navy"],
}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def svg_start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="64" y="54" font-family="DejaVu Sans, sans-serif" font-size="26" font-weight="700" fill="{COLORS["ink"]}">{esc(title)}</text>',
        f'<text x="64" y="82" font-family="DejaVu Sans, sans-serif" font-size="15" fill="{COLORS["gray"]}">{esc(subtitle)}</text>',
    ]


def svg_end(lines: list[str]) -> str:
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def text(lines, x, y, value, size=14, fill=None, anchor="start", weight="400", rotate=None):
    fill = fill or COLORS["ink"]
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    lines.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{transform}>{esc(value)}</text>'
    )


def line(lines, x1, y1, x2, y2, stroke, width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>')


def circle(lines, x, y, radius, fill, stroke=None, width=1):
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}"{stroke_attr}/>')


def rect(lines, x, y, width, height, fill, stroke=None, radius=0, opacity=None):
    stroke_attr = f' stroke="{stroke}"' if stroke else ""
    opacity_attr = f' opacity="{opacity}"' if opacity is not None else ""
    lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="{radius}" fill="{fill}"{stroke_attr}{opacity_attr}/>')


def write_svg(name: str, lines: list[str]):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(svg_end(lines))


def read_csv(name: str) -> list[dict[str, str]]:
    with (ASSETS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def render_forest():
    rows = read_csv("figure1_paired_vs_nomem.csv")
    width, height = 1180, 700
    left, right, top, bottom = 310, 86, 140, 88
    x0, x1 = left, width - right
    y0, y1 = top, height - bottom
    xmin, xmax = -0.02, 0.22
    sx = lambda value: x0 + (float(value) - xmin) / (xmax - xmin) * (x1 - x0)
    sy = lambda idx: y0 + idx * (y1 - y0) / (len(rows) - 1)
    lines = svg_start(width, height, "Figure 1. Paired gains versus no memory", "ALFWorld unseen split; 402 paired episodes per comparison; points are means and bars are 95% CIs")
    for tick in [0.00, 0.05, 0.10, 0.15, 0.20]:
        xx = sx(tick)
        line(lines, xx, y0 - 25, xx, y1 + 8, COLORS["grid"], 1)
        text(lines, xx, y1 + 38, f"{tick:+.2f}", 13, COLORS["gray"], "middle")
    line(lines, sx(0), y0 - 25, sx(0), y1 + 8, COLORS["ink"], 1.8)
    for idx, row in enumerate(rows):
        yy = sy(idx)
        label = VARIANT_LABELS[row["variant_key"]]
        lo = float(row["delta_success_ci95_lo"])
        hi = float(row["delta_success_ci95_hi"])
        value = float(row["delta_success_mean"])
        color = VARIANT_COLORS[row["variant_key"]]
        line(lines, sx(lo), yy, sx(hi), yy, color, 5)
        circle(lines, sx(value), yy, 7, color, COLORS["white"], 2)
        text(lines, left - 26, yy + 5, label, 15, COLORS["ink"], "end")
        text(lines, sx(hi) + 14, yy + 5, f"{value:+.3f}", 13, color, "start", "700")
    text(lines, (x0 + x1) / 2, height - 28, "Paired success-rate change", 14, COLORS["gray"], "middle")
    write_svg("figure1_paired_gains.svg", lines)


def render_task_types():
    rows = read_csv("figure2_tasktype_best_vs_nomem.csv")
    order = ["clean", "cool", "examine", "heat", "put", "puttwo"]
    rows.sort(key=lambda row: order.index(row["task_type"]))
    width, height = 1180, 700
    left, right, top, bottom = 220, 100, 140, 90
    x0, x1 = left, width - right
    y0, y1 = top, height - bottom
    xmin, xmax = -0.16, 0.42
    sx = lambda value: x0 + (float(value) - xmin) / (xmax - xmin) * (x1 - x0)
    sy = lambda idx: y0 + idx * (y1 - y0) / (len(rows) - 1)
    lines = svg_start(width, height, "Figure 2. Task-type conditionality", "Insight + BM25 versus no memory; labels at right show no-memory baseline success")
    for tick in [-0.10, 0.00, 0.10, 0.20, 0.30, 0.40]:
        xx = sx(tick)
        line(lines, xx, y0 - 25, xx, y1 + 8, COLORS["grid"], 1)
        text(lines, xx, y1 + 38, f"{tick:+.2f}", 13, COLORS["gray"], "middle")
    line(lines, sx(0), y0 - 25, sx(0), y1 + 8, COLORS["ink"], 1.8)
    for idx, row in enumerate(rows):
        yy = sy(idx)
        value = float(row["delta_success_mean"])
        lo = float(row["delta_success_ci95_lo"])
        hi = float(row["delta_success_ci95_hi"])
        color = COLORS["red"] if value < 0 else COLORS["orange"]
        line(lines, sx(lo), yy, sx(hi), yy, color, 5)
        circle(lines, sx(value), yy, 7, color, COLORS["white"], 2)
        text(lines, left - 24, yy + 5, row["task_type"], 15, COLORS["ink"], "end")
        text(lines, sx(hi) + 14, yy + 5, f"baseline {float(row['baseline_success_rate']):.2f}", 13, COLORS["gray"])
    text(lines, (x0 + x1) / 2, height - 28, "Paired success-rate change", 14, COLORS["gray"], "middle")
    write_svg("figure2_tasktype_conditionality.svg", lines)


def render_frontier():
    rows = read_csv("table1_main_results.csv")
    width, height = 1240, 780
    left, right, top, bottom = 150, 180, 140, 110
    x0, x1 = left, width - right
    y0, y1 = height - bottom, top
    xmin, xmax = 50_000, 110_000
    ymin, ymax = 0.50, 0.72
    sx = lambda value: x0 + (float(value) - xmin) / (xmax - xmin) * (x1 - x0)
    sy = lambda value: y0 - (float(value) - ymin) / (ymax - ymin) * (y0 - y1)
    lines = svg_start(width, height, "Figure 3. Utility-cost frontier", "Point size encodes realized memory tokens; the highlighted line is the current success/token frontier")
    for tick in range(50_000, 110_001, 10_000):
        xx = sx(tick)
        line(lines, xx, y1 - 8, xx, y0, COLORS["grid"], 1)
        text(lines, xx, y0 + 34, f"{tick // 1000}k", 13, COLORS["gray"], "middle")
    for tick in [0.50, 0.55, 0.60, 0.65, 0.70]:
        yy = sy(tick)
        line(lines, x0, yy, x1 + 8, yy, COLORS["grid"], 1)
        text(lines, x0 - 18, yy + 5, f"{tick:.2f}", 13, COLORS["gray"], "end")
    line(lines, x0, y0, x1 + 8, y0, COLORS["ink"], 1.5)
    line(lines, x0, y1 - 8, x0, y0, COLORS["ink"], 1.5)
    frontier = []
    for row in rows:
        x = float(row["mean_prompt_tokens"])
        y = float(row["success_rate"])
        radius = 7 + min(12, float(row["realized_memory_tokens"]) / 250)
        color = VARIANT_COLORS[row["variant_key"]]
        circle(lines, sx(x), sy(y), radius, color, COLORS["white"], 2)
        label = VARIANT_LABELS[row["variant_key"]]
        dx, dy = (14, -12)
        if row["variant_key"] == "skillos_bm25":
            dy = 22
        text(lines, sx(x) + dx, sy(y) + dy, label, 13, color, "start", "700" if row["variant_key"] == "expel_bm25" else "400")
        if row["variant_key"] == "nomem":
            frontier.append((x, y))
        if row["variant_key"] == "expel_bm25":
            frontier.append((x, y))
    frontier.sort()
    points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in frontier)
    lines.append(f'<polyline points="{points}" fill="none" stroke="{COLORS["orange"]}" stroke-width="2.5" stroke-dasharray="8 6" opacity="0.8"/>')
    text(lines, (x0 + x1) / 2, height - 38, "Mean prompt tokens per episode", 14, COLORS["gray"], "middle")
    text(lines, 32, (y0 + y1) / 2, "Success rate", 14, COLORS["gray"], "middle", rotate=-90)
    legend_x, legend_y = width - 162, 180
    rect(lines, legend_x - 18, legend_y - 26, 158, 145, COLORS["white"], COLORS["grid"], 5)
    text(lines, legend_x, legend_y, "Memory footprint", 13, COLORS["gray"], "start", "700")
    for idx, value in enumerate([0, 650, 1800]):
        yy = legend_y + 30 + idx * 30
        circle(lines, legend_x + 12, yy, 7 + min(12, value / 250), COLORS["gray"], COLORS["white"], 1)
        text(lines, legend_x + 38, yy + 5, f"{value} tokens", 12, COLORS["gray"])
    write_svg("figure3_utility_cost_frontier.svg", lines)


def render_reallocation():
    rows = read_csv("appendix_task_reallocation_types12.csv")
    columns = [
        ("PILOT_eqfp_nomem_s0_types12_success", "No memory"),
        ("PILOT_eqfp_expel_bm25_s0_types12_success", "Insight"),
        ("PILOT_eqfp_raw_bm25_s0_types12_success", "Raw"),
        ("PILOT_eqfp_skillos_bm25_s0_types12_success", "Markdown"),
    ]
    width, height = 1120, 760
    left, top = 230, 145
    cell_w, cell_h = 170, 38
    lines = svg_start(width, height, "Appendix Figure A1. Matched-footprint task reallocation", "12 balanced tasks; green means success and pale red means failure")
    for col_idx, (_, label) in enumerate(columns):
        text(lines, left + col_idx * cell_w + cell_w / 2, top - 28, label, 14, COLORS["ink"], "middle", "700")
    for row_idx, row in enumerate(rows):
        yy = top + row_idx * cell_h
        text(lines, left - 20, yy + 25, f"{row['task_id']} / {row['task_type']}", 13, COLORS["ink"], "end")
        for col_idx, (field, _) in enumerate(columns):
            xx = left + col_idx * cell_w
            success = int(row[field])
            fill = "#B9E2D5" if success else "#F4D4D0"
            rect(lines, xx + 6, yy + 3, cell_w - 12, cell_h - 6, fill, COLORS["white"], 4)
            text(lines, xx + cell_w / 2, yy + 26, "success" if success else "failure", 12, COLORS["ink"], "middle", "700")
    text(lines, left + 2 * cell_w, height - 35, "All four conditions finish at 5/12 overall successes; the identity of successful tasks changes.", 13, COLORS["gray"], "middle")
    write_svg("appendix_figureA1_task_reallocation.svg", lines)


def render_executor_transfer():
    rows = [
        row for row in read_csv("table3_executor_transfer.csv")
        if row["representation"] != "none"
    ]
    order = [
        ("expel", "bm25"),
        ("expel", "dense"),
        ("skillos", "bm25"),
        ("skillos", "dense"),
    ]
    labels = {
        ("expel", "bm25"): "Insight + BM25",
        ("expel", "dense"): "Insight + dense",
        ("skillos", "bm25"): "Markdown + BM25",
        ("skillos", "dense"): "Markdown + dense",
    }
    executor_colors = {
        "gpt-4o-mini": COLORS["orange"],
        "qwen3-8b": COLORS["blue"],
    }
    executor_labels = {
        "gpt-4o-mini": "GPT-4o-mini",
        "qwen3-8b": "Qwen3-8B",
    }
    width, height = 1260, 800
    left, right, top, bottom = 340, 110, 145, 90
    x0, x1 = left, width - right
    y0, y1 = top, height - bottom
    xmin, xmax = -0.14, 0.22
    sx = lambda value: x0 + (float(value) - xmin) / (xmax - xmin) * (x1 - x0)
    lines = svg_start(
        width,
        height,
        "Figure 4. Executor dependence of memory gains",
        "Paired success-rate change versus the executor-specific no-memory baseline; missing Qwen BM25 rows are added after completion",
    )
    for tick in [-0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20]:
        xx = sx(tick)
        line(lines, xx, y0 - 25, xx, y1 + 8, COLORS["grid"], 1)
        text(lines, xx, y1 + 38, f"{tick:+.2f}", 13, COLORS["gray"], "middle")
    line(lines, sx(0), y0 - 25, sx(0), y1 + 8, COLORS["ink"], 1.8)

    slot = (y1 - y0) / len(order)
    for group_idx, key in enumerate(order):
        center = y0 + (group_idx + 0.5) * slot
        group_rows = [
            row for row in rows
            if (row["representation"], row["retriever"]) == key
        ]
        text(lines, left - 145, center + 5, labels[key], 15, COLORS["ink"], "end", "700")
        if group_idx:
            line(lines, 60, center - slot / 2, x1, center - slot / 2, COLORS["light"], 1)
        offsets = {"gpt-4o-mini": -15, "qwen3-8b": 15}
        for row in group_rows:
            executor = row["executor"]
            yy = center + offsets[executor]
            value = float(row["paired_delta_success_mean"])
            lo = float(row["paired_delta_success_ci95_lo"])
            hi = float(row["paired_delta_success_ci95_hi"])
            color = executor_colors[executor]
            line(lines, sx(lo), yy, sx(hi), yy, color, 5)
            circle(lines, sx(value), yy, 7, color, COLORS["white"], 2)
            text(lines, left - 130, yy + 5, executor_labels[executor], 12, color, "start")
            text(lines, sx(hi) + 12, yy + 5, f"{value:+.3f}", 12, color, "start", "700")
    text(lines, (x0 + x1) / 2, height - 28, "Paired success-rate change", 14, COLORS["gray"], "middle")
    write_svg("figure4_executor_transfer.svg", lines)


def render_headroom_crossfit():
    rows = read_csv("table4_crossfit_headroom.csv")
    width, height = 1320, 800
    panel_top, panel_bottom = 155, 650
    panel_width = 520
    panel_lefts = {"gpt-4o-mini": 130, "qwen3-8b": 735}
    ymin, ymax = -0.20, 0.35
    sy = lambda value: panel_bottom - (float(value) - ymin) / (ymax - ymin) * (panel_bottom - panel_top)
    styles = {
        ("expel", "bm25"): (COLORS["orange"], "Insight + BM25"),
        ("expel", "dense"): (COLORS["gold"], "Insight + dense"),
        ("skillos", "bm25"): (COLORS["teal"], "Markdown + BM25"),
        ("skillos", "dense"): (COLORS["blue"], "Markdown + dense"),
    }
    lines = svg_start(
        width,
        height,
        "Figure 5. Cross-fitted baseline competence and memory utility",
        "Difficulty uses the other two no-memory seeds; intervals are task-cluster bootstrap 95% CIs",
    )
    for executor, left in panel_lefts.items():
        right = left + panel_width
        x_positions = [left + 85, left + panel_width / 2, right - 85]
        for tick in [-0.20, -0.10, 0.00, 0.10, 0.20, 0.30]:
            yy = sy(tick)
            line(lines, left, yy, right, yy, COLORS["grid"], 1)
            if executor == "gpt-4o-mini":
                text(lines, left - 15, yy + 5, f"{tick:+.2f}", 12, COLORS["gray"], "end")
        line(lines, left, sy(0), right, sy(0), COLORS["ink"], 1.8)
        title = "GPT-4o-mini" if executor == "gpt-4o-mini" else "Qwen3-8B"
        text(lines, (left + right) / 2, panel_top - 24, title, 18, COLORS["ink"], "middle", "700")
        for x, label in zip(x_positions, ["hard\n0/2", "mixed\n1/2", "easy\n2/2"]):
            first, second = label.split("\n")
            text(lines, x, panel_bottom + 34, first, 13, COLORS["ink"], "middle", "700")
            text(lines, x, panel_bottom + 53, second, 12, COLORS["gray"], "middle")
        for (representation, retriever), (color, _) in styles.items():
            series = sorted(
                [
                    row for row in rows
                    if row["executor"] == executor
                    and row["representation"] == representation
                    and row["retriever"] == retriever
                ],
                key=lambda row: int(row["other_seed_successes"]),
            )
            points = []
            for row, x in zip(series, x_positions):
                value = float(row["delta_success_mean"])
                lo = float(row["delta_success_cluster_boot_ci95_lo"])
                hi = float(row["delta_success_cluster_boot_ci95_hi"])
                line(lines, x, sy(lo), x, sy(hi), color, 3)
                circle(lines, x, sy(value), 6, color, COLORS["white"], 1.5)
                points.append(f"{x:.1f},{sy(value):.1f}")
            lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2" opacity="0.75"/>')
    text(lines, 34, (panel_top + panel_bottom) / 2, "Paired success-rate change", 14, COLORS["gray"], "middle", rotate=-90)
    legend_y = 735
    legend_x = 215
    for idx, (_, (color, label)) in enumerate(styles.items()):
        xx = legend_x + idx * 260
        line(lines, xx, legend_y, xx + 34, legend_y, color, 3)
        circle(lines, xx + 17, legend_y, 5, color, COLORS["white"], 1)
        text(lines, xx + 44, legend_y + 5, label, 12, COLORS["ink"])
    write_svg("figure5_crossfit_headroom.svg", lines)


def render_executor_moderation():
    rows = read_csv("table5_executor_moderation.csv")
    labels = {
        "expel_bm25": "Insight + BM25",
        "expel_dense": "Insight + dense",
        "skillos_bm25": "Markdown + BM25",
        "skillos_dense": "Markdown + dense",
        "expel": "Insight: executor x retriever",
        "skillos": "Markdown: executor x retriever",
    }
    width, height = 1240, 750
    left, right, top, bottom = 360, 100, 145, 90
    x0, x1 = left, width - right
    y0, y1 = top, height - bottom
    xmin, xmax = -0.30, 0.15
    sx = lambda value: x0 + (float(value) - xmin) / (xmax - xmin) * (x1 - x0)
    sy = lambda idx: y0 + idx * (y1 - y0) / (len(rows) - 1)
    lines = svg_start(
        width,
        height,
        "Figure 6. Executor moderation of memory utility",
        "Qwen-minus-GPT difference in paired memory gains; intervals use task-cluster bootstrap",
    )
    for tick in [-0.30, -0.20, -0.10, 0.00, 0.10]:
        xx = sx(tick)
        line(lines, xx, y0 - 25, xx, y1 + 8, COLORS["grid"], 1)
        text(lines, xx, y1 + 38, f"{tick:+.2f}", 13, COLORS["gray"], "middle")
    line(lines, sx(0), y0 - 25, sx(0), y1 + 8, COLORS["ink"], 1.8)
    for idx, row in enumerate(rows):
        yy = sy(idx)
        value = float(row["estimate"])
        lo = float(row["task_cluster_boot_ci95_lo"])
        hi = float(row["task_cluster_boot_ci95_hi"])
        color = COLORS["orange"] if row["effect_type"] == "executor_x_memory" else COLORS["blue"]
        if idx == 4:
            line(lines, 60, yy - (y1 - y0) / (len(rows) - 1) / 2, x1, yy - (y1 - y0) / (len(rows) - 1) / 2, COLORS["light"], 2)
        line(lines, sx(lo), yy, sx(hi), yy, color, 5)
        circle(lines, sx(value), yy, 7, color, COLORS["white"], 2)
        text(lines, left - 25, yy + 5, labels[row["label"]], 14, COLORS["ink"], "end")
        text(lines, sx(hi) + 12, yy + 5, f"{value:+.3f}", 12, color, "start", "700")
    text(lines, (x0 + x1) / 2, height - 28, "Difference in paired success-rate gain (Qwen minus GPT)", 14, COLORS["gray"], "middle")
    write_svg("figure6_executor_moderation.svg", lines)


def load_episodes(run_id: str) -> list[dict]:
    with (RESULTS / f"{run_id}.episodes.jsonl").open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_analysis_tables():
    main = read_csv("table1_main_results.csv")
    baseline = {
        (int(row["seed"]), row["task_id"]): row
        for row in load_episodes("R008_nomem_s3")
    }
    variant_ids = [row["run_id"] for row in main if row["variant_key"] != "nomem"]
    seed_rows = []
    for run_id in variant_ids:
        variant = {
            (int(row["seed"]), row["task_id"]): row
            for row in load_episodes(run_id)
        }
        for seed in sorted({key[0] for key in variant}):
            pairs = [
                (base, variant[key])
                for key, base in baseline.items()
                if key[0] == seed and key in variant
            ]
            seed_rows.append({
                "run_id": run_id,
                "seed": seed,
                "delta_success": f"{mean(int(candidate['success']) - int(base['success']) for base, candidate in pairs):+.4f}",
                "delta_prompt_tokens": f"{mean(float(candidate['prompt_tokens']) - float(base['prompt_tokens']) for base, candidate in pairs):+.1f}",
                "paired_n": len(pairs),
            })
    write_csv(OUT / "seed_stability.csv", ["run_id", "seed", "delta_success", "delta_prompt_tokens", "paired_n"], seed_rows)

    frontier_rows = []
    for objective in ["prompt_tokens", "total_cost_usd"]:
        for row in main:
            value = float(row["mean_prompt_tokens"] if objective == "prompt_tokens" else row["total_cost_usd"])
            success = float(row["success_rate"])
            dominated = any(
                float(other["success_rate"]) >= success
                and float(other["mean_prompt_tokens"] if objective == "prompt_tokens" else other["total_cost_usd"]) <= value
                and (float(other["success_rate"]) > success or float(other["mean_prompt_tokens"] if objective == "prompt_tokens" else other["total_cost_usd"]) < value)
                for other in main
            )
            if not dominated:
                frontier_rows.append({"objective": objective, "variant_key": row["variant_key"], "success_rate": row["success_rate"], "resource": f"{value:.4f}"})
    write_csv(OUT / "pareto_frontier.csv", ["objective", "variant_key", "success_rate", "resource"], frontier_rows)

    baseline_row = next(row for row in main if row["variant_key"] == "nomem")
    efficiency_rows = []
    for row in main:
        if row["variant_key"] == "nomem":
            continue
        delta = float(row["success_rate"]) - float(baseline_row["success_rate"])
        extra_tokens = float(row["mean_prompt_tokens"]) - float(baseline_row["mean_prompt_tokens"])
        efficiency_rows.append({
            "variant_key": row["variant_key"],
            "delta_success": f"{delta:+.4f}",
            "extra_prompt_tokens": f"{extra_tokens:+.1f}",
            "delta_success_per_1k_extra_prompt_tokens": f"{delta / extra_tokens * 1000:+.4f}",
        })
    write_csv(OUT / "efficiency_vs_nomem.csv", ["variant_key", "delta_success", "extra_prompt_tokens", "delta_success_per_1k_extra_prompt_tokens"], efficiency_rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    render_forest()
    render_task_types()
    render_frontier()
    render_reallocation()
    render_executor_transfer()
    render_headroom_crossfit()
    render_executor_moderation()
    export_analysis_tables()
    print(f"wrote Paper O figures and analysis tables to {OUT}")


if __name__ == "__main__":
    main()
