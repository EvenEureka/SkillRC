#!/usr/bin/env python3
"""Generate the TikZ case-study figures (Appendix L).

Panel (a) of each ALFWorld figure shows the real environment: a first-person AI2-THOR
rendering and the top-down map of the ALFRED trial behind the task, with the places of
the ALFWorld observation labelled by their ALFWorld names. The renders, object boxes,
and map positions come from paper/figures/cases/<case>.json, written by
scripts/figures/prepare_case_assets.py from scripts/figures/render_cases.py (AI2-THOR 2.1.0,
no model run). The WebShop figure shows screenshots of the official WebShop web app.
Panel (b) shows recorded outcomes: GPT-4o-mini seed-0 episode records, the equal-pair
task effects of the task taxonomy, and the run summaries in data/summaries/.
Nothing here is estimated.
Run:  python paper/figures/make_case_figures.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
ASSETS = OUT / "cases"
TEXPATH = "figures/cases"   # \includegraphics path, relative to paper/
FULL = 15.0                 # figure width (cm)
CHAR = 0.118                # approx. width of one \scriptsize sans character (cm)
LH = 0.36                   # label height (cm)

ROLE = {"object": "dimF", "start": "dimA", "appliance": "dimE", "dest": "liftpositive"}
ROLE_FILL = {None: ("black!4", "black!25"), "start": ("dimA!25", "dimA"), "appliance": ("dimE!22", "dimE"),
             "dest": ("liftpositive!18", "liftpositive"), "object": ("dimF!18", "dimF")}
ALF_NAME = {"SinkBasin": "sinkbasin", "Sink": "sinkbasin"}
PLURAL = {"shelf": "shelves"}
SPREAD = {"Cabinet", "Drawer", "Shelf", "StoveBurner"}   # repeated furniture labelled once per 1.6 m


def _r3(v: float) -> str:
    """Round half up to three decimals, as in the paper text (0.1045 -> 0.105)."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(str(v)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _esc(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def alf(t: str) -> str:
    return ALF_NAME.get(t, t.lower())


def plural(n: str) -> str:
    return PLURAL.get(n, n + "s")


def _header():
    return r"""\begin{tikzpicture}[x=1cm, y=1cm, font=\sffamily\footnotesize,
  ptitle/.style={anchor=west, font=\sffamily\bfseries\small, text=narrareddeep},
  textbox/.style={draw=phaseA!60, fill=phaseA!6, rounded corners=2pt, inner sep=4pt, anchor=north west,
                  align=left, font=\sffamily\scriptsize},
  lbl/.style={draw=#1, fill=white, fill opacity=0.9, text opacity=1, rounded corners=1pt, inner sep=1.6pt,
              line width=0.6pt, font=\sffamily\scriptsize},
  lblkey/.style={draw=#1, fill=white, fill opacity=0.95, text opacity=1, rounded corners=1pt, inner sep=1.8pt,
                 line width=1.0pt, font=\sffamily\scriptsize\bfseries},
  lead/.style={draw=#1, line width=0.7pt},
  path/.style={-{Latex[length=2.6mm, width=2.4mm]}, draw=black!85, line width=1.6pt},
  imgcap/.style={anchor=north, font=\sffamily\scriptsize, text=black!60},
  chip/.style={draw=black!30, fill=white, rounded corners=6pt, inner sep=2.5pt, font=\sffamily\footnotesize},
  flow/.style={-{Latex[length=1.6mm]}, draw=black!50, line width=0.6pt},
  lab/.style={anchor=east, font=\sffamily\footnotesize, text=black!75},
  txt/.style={anchor=west, font=\sffamily\footnotesize, text=black!75}]
"""


# --------------------------------------------------------------------------- labels on images
class Placer:
    """Greedy, collision-free placement of labels near their anchor points inside an image."""

    def __init__(self, x0, y0, w, h):
        self.bounds = (x0, y0 - h, x0 + w, y0)
        self.rects = []

    def _ok(self, r):
        bx0, by0, bx1, by1 = self.bounds
        if r[0] < bx0 + 0.04 or r[2] > bx1 - 0.04 or r[1] < by0 + 0.04 or r[3] > by1 - 0.04:
            return False
        return all(r[2] < q[0] - 0.05 or r[0] > q[2] + 0.05 or r[3] < q[1] - 0.04 or r[1] > q[3] + 0.04
                   for q in self.rects)

    def place(self, ax, ay, text, inside=False, required=True):
        """Return (x, y, moved) for the label centre, or None when an optional label has no free spot."""
        import math
        tw = len(re.sub(r"\\[a-z]+|[{}$]", "", text)) * CHAR + 0.2
        cands = [(0.0, 0.0)] if inside else []
        for d in (0.5, 0.8, 1.1, 1.45, 1.85, 2.3, 2.8):
            for k in range(16):
                t = 2 * math.pi * k / 16 + math.pi / 2       # start straight above, then go round
                cands.append((math.cos(t) * (d + tw / 2 * abs(math.cos(t))), math.sin(t) * d))
        for dx, dy in cands:
            cx, cy = ax + dx, ay + dy
            r = (cx - tw / 2, cy - LH / 2, cx + tw / 2, cy + LH / 2)
            if self._ok(r):
                self.rects.append(r)
                return cx, cy, (dx, dy) != (0.0, 0.0)
        if not required:
            return None
        bx0, by0, bx1, by1 = self.bounds   # fall back: clamp into the image
        cx = min(max(ax, bx0 + tw / 2 + 0.05), bx1 - tw / 2 - 0.05)
        cy = min(max(ay + 0.55, by0 + LH), by1 - LH)
        self.rects.append((cx - tw / 2, cy - LH / 2, cx + tw / 2, cy + LH / 2))
        return cx, cy, True


def role_text(role, name, obj, verb, two):
    if role == "start":
        return f"{name}: the {obj}{'s are' if two else ' is'} here"
    if role == "appliance":
        return f"{name}: {verb} it here" if verb != "use" else f"{name}: turn it on"
    if role == "dest":
        return f"{name}: put {'both' if two else 'it'} here"
    return obj


def _image(lines, path, x, y, w, size):
    h = w * size[1] / size[0]
    lines.append(rf"\node[anchor=north west, inner sep=0] at ({x:.3f},{y:.3f}) "
                 rf"{{\includegraphics[width={w:.3f}cm]{{{TEXPATH}/{path}}}}};")
    lines.append(rf"\draw[black!35, line width=0.4pt] ({x:.3f},{y:.3f}) rectangle ++({w:.3f},{-h:.3f});")
    return h


def _label(lines, placer, ax, ay, text, role, inside=False, dot=True):
    got = placer.place(ax, ay, text, inside=inside, required=role is not None)
    if got is None:      # an unlabelled minor place is better than overlapping labels
        print("  skipped label:", text)
        return
    cx, cy, moved = got
    color = ROLE.get(role, "black!55")
    style = f"lblkey={color}" if role else "lbl=black!45"
    if moved:
        lines.append(rf"\draw[lead={color}] ({ax:.3f},{ay:.3f}) -- ({cx:.3f},{cy:.3f});")
    if dot or moved:
        lines.append(rf"\fill[{color}] ({ax:.3f},{ay:.3f}) circle (0.06);")
    lines.append(rf"\node[{style}] at ({cx:.3f},{cy:.3f}) {{{text}}};")


def view_panel(lines, case, view, x, y, w, verb, two):
    """First-person render with a box and label for each key place and one label per other type."""
    W, H = view["size"]
    k = w / W
    h = _image(lines, view["file"], x, y, w, view["size"])
    placer = Placer(x, y, w, h)
    roles = case["roles"]
    obj = alf(next(o.split("|")[0] for o, r in roles.items() if r == "object"))
    to_cm = lambda u, v: (x + u * k, y - v * k)
    boxes = view["boxes"]
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    # key places and the object first, so they get the best label positions
    order = sorted(boxes, key=lambda o: (roles.get(o) is None, roles.get(o) != "object"))
    done_types = set()
    for oid in order:
        b, role, typ = boxes[oid], roles.get(oid), oid.split("|")[0]
        if role in ("start", "appliance", "dest"):
            x0, y0 = to_cm(b[0], b[1]); x1, y1 = to_cm(b[2], b[3])
            lines.append(rf"\draw[{ROLE[role]}, line width=1.1pt, rounded corners=1pt] ({x0:.3f},{y0:.3f}) rectangle ({x1:.3f},{y1:.3f});")
            ax, ay = to_cm((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            _label(lines, placer, ax, ay, role_text(role, alf(typ), obj, verb, two), role, inside=area(b) > 0.05 * W * H)
        elif role == "object":
            ax, ay = to_cm((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            r = max(0.16, 0.6 * k * max(b[2] - b[0], b[3] - b[1]))
            lines.append(rf"\draw[{ROLE['object']}, line width=1.1pt] ({ax:.3f},{ay:.3f}) circle ({r:.3f});")
            _label(lines, placer, ax, ay, obj, "object", dot=False)
    for oid in sorted(boxes, key=lambda o: -area(boxes[o])):
        b, typ = boxes[oid], oid.split("|")[0]
        if oid in roles or typ in done_types or area(b) < 0.006 * W * H:
            continue
        n = sum(1 for o, bb in boxes.items() if o.split("|")[0] == typ and o not in roles and area(bb) >= 0.002 * W * H)
        done_types.add(typ)
        ax, ay = to_cm((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        _label(lines, placer, ax, ay, plural(alf(typ)) if n > 1 else alf(typ), None, inside=area(b) > 0.04 * W * H)
    return h


def map_panel(lines, case, x, y, w, verb, two):
    """Top-down map with every place of the observation labelled and the expert path drawn."""
    mp = case["map"]
    W, H = mp["size"]
    k = w / W
    h = _image(lines, mp["file"], x, y, w, mp["size"])
    placer = Placer(x, y, w, h)
    roles = case["roles"]
    obj = alf(next(o.split("|")[0] for o, r in roles.items() if r == "object"))
    to_cm = lambda u, v: (x + u * k, y - v * k)
    objs = {o["objectId"]: o for o in mp["objects"]}

    def nearest(oid):  # roles were matched in the same scene, so ids coincide
        return objs.get(oid)

    for v in case["views"][:1]:   # where the first-person view was taken and where it looks
        cx, cy = to_cm(*v["map_uv"])
        a = 90 - v["rotation"]      # AI2-THOR rotation 0 faces +z (up on the map), 90 faces +x
        lines.append(rf"\fill[black, opacity=0.22] ({cx:.3f},{cy:.3f}) -- ++({a - 45}:1.1) arc ({a - 45}:{a + 45}:1.1) -- cycle;")
        lines.append(rf"\fill[black] ({cx:.3f},{cy:.3f}) circle (0.08);")
        placer.rects.append((cx - 0.1, cy - 0.1, cx + 0.1, cy + 0.1))
    seq = [o for r in ("start", "appliance", "dest") for o, rr in roles.items() if rr == r]
    pts = [to_cm(*nearest(o)["uv"]) for o in seq if nearest(o)]
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        dx, dy = bx - ax, by - ay
        n = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        seg = rf"({ax + dx / n * 0.2:.3f},{ay + dy / n * 0.2:.3f}) -- ({bx - dx / n * 0.25:.3f},{by - dy / n * 0.25:.3f})"
        lines.append(rf"\draw[white, line width=3.4pt, opacity=0.85] {seg};")
        lines.append(rf"\draw[path] {seg};")
    for oid in seq + [o for o, r in roles.items() if r == "object"]:
        o = nearest(oid)
        if not o:
            continue
        role = roles[oid]
        ax, ay = to_cm(*o["uv"])
        if role == "object":
            continue   # the object sits on its start place; the start label names it
        if "box" in o:
            b = o["box"]
            x0, y0 = to_cm(b[0], b[1]); x1, y1 = to_cm(b[2], b[3])
            lines.append(rf"\draw[{ROLE[role]}, line width=1.1pt, rounded corners=1pt] ({x0:.3f},{y0:.3f}) rectangle ({x1:.3f},{y1:.3f});")
        _label(lines, placer, ax, ay, role_text(role, alf(o["type"]), obj, verb, two), role)
    # other places: one label per cluster of same-type places within 1 m
    rec_types = {("SinkBasin" if t == "Sink" else t) for t in case["receptacles"]}
    rest = [o for o in mp["objects"] if o["type"] in rec_types and o["objectId"] not in roles]
    used = set()
    for o in sorted(rest, key=lambda o: o["type"]):
        if o["objectId"] in used:
            continue
        cl = [p for p in rest if p["type"] == o["type"] and p["objectId"] not in used and
              ((p["uv"][0] - o["uv"][0]) ** 2 + (p["uv"][1] - o["uv"][1]) ** 2) ** 0.5
              < mp["px_per_m"] * (1.6 if o["type"] in SPREAD else 1.0)]
        used |= {p["objectId"] for p in cl}
        u = sum(p["uv"][0] for p in cl) / len(cl)
        v = sum(p["uv"][1] for p in cl) / len(cl)
        name = alf(o["type"])
        _label(lines, placer, *to_cm(u, v), plural(name) if len(cl) > 1 else name, None)
    return h


def observation(case):
    """The ALFWorld initial observation: places sorted by name, instance numbers descending."""
    items = []
    for t in sorted(case["receptacles"], key=alf):
        items += [f"a {alf(t)} {i}" for i in range(case["receptacles"][t], 0, -1)]
    return ("You are in the middle of a room. Looking quickly around you, you see "
            + ", ".join(items[:-1]) + f", and {items[-1]}.")


def expert_steps(case):
    return [" ".join(re.sub(r"_bar_.*", "", w) for w in s.split()) for s in case["walkthrough"]]


def env_panel(lines, case, title, verb, two, y=0.0):
    lines.append(rf"\node[ptitle] at (0,{y:.3f}) {{{title}}};")
    text = (rf"\textbf{{Goal.}} Your task is to: {_esc(case['goal'])}\\[1pt]"
            rf"\textbf{{Initial observation.}} {_esc(observation(case))}")
    lines.append(rf"\node[textbox, text width={FULL - 0.3:.2f}cm] (obs) at (0,{y - 0.28:.3f}) {{{text}}};")
    lines.append(r"\begin{scope}[shift={($(obs.south west)+(0,-0.25)$)}]")   # closed by the caller
    ytop = 0.0
    views = case["views"]
    msize = case["map"]["size"]
    ar_m = msize[1] / msize[0]
    gap = 0.25
    # heights equal: 0.75 * wv * len(views)^-1 ... views side by side, then the map
    nv = len(views)
    wv = (FULL - gap * nv) / (nv + 0.75 / ar_m) if nv else 0
    wm = FULL - nv * (wv + gap)
    x = 0.0
    hs = []
    for v in views:
        hs.append(view_panel(lines, case, v, x, ytop, wv, verb, two))
        x += wv + gap
    hs.append(map_panel(lines, case, x, ytop, wm, verb, two))
    hmax = max(hs)
    for i, v in enumerate(views):
        lines.append(rf"\node[imgcap] at ({i * (wv + gap) + wv / 2:.3f},{ytop - hmax - 0.04:.3f}) {{first-person view inside the room}};")
    lines.append(rf"\node[imgcap] at ({x + wm / 2:.3f},{ytop - hmax - 0.04:.3f}) {{the room from above; cone: the left view; arrow: the expert's route}};")
    ye = ytop - hmax - 0.8
    steps = expert_steps(case)
    lines.append(rf"\node[anchor=west, font=\sffamily\bfseries\footnotesize, text=black!70] at (0,{ye:.3f}) "
                 rf"{{Expert solution ({len(steps)} steps):}};")
    xs, yy = 3.9, ye
    for k, s in enumerate(steps):
        w = len(s) * 0.142 + 0.4
        if xs + w > FULL:
            xs, yy = 3.9, yy - 0.5
        lines.append(rf"\node[chip, anchor=west] (s{k}) at ({xs:.3f},{yy:.3f}) {{{_esc(s)}}};")
        if k and xs > 3.95:
            lines.append(rf"\draw[flow] (s{k - 1}.east) -- (s{k}.west);")
        xs += w + 0.3
    return yy


# --------------------------------------------------------------------------- panel (b)s
def _steps_panel(x0, y_top, rows, note_lines):
    """Horizontal bars of environment steps against the 30-step limit."""
    out = [rf"\node[ptitle] at ({x0:.3f},{y_top:.3f}) {{(b) What happened: \texttt{{GPT-4o-mini}}, seed 0}};"]
    scale = 6.0 / 30.0
    bx = x0 + 1.75
    y = y_top - 0.6
    y_first = y
    for label, steps, color, verdict in rows:
        out.append(rf"\node[lab] at ({bx - 0.1:.3f},{y:.3f}) {{{label}}};")
        out.append(rf"\fill[{color}, rounded corners=1pt] ({bx:.3f},{y + 0.15:.3f}) rectangle ++({steps * scale:.3f},-0.3);")
        out.append(rf"\node[txt] at ({bx + steps * scale + 0.08:.3f},{y:.3f}) {{{verdict}}};")
        y -= 0.46
    lim = bx + 30 * scale
    out.append(rf"\draw[densely dashed, black!40] ({lim:.3f},{y_first + 0.25:.3f}) -- ({lim:.3f},{y + 0.2:.3f});")
    out.append(rf"\node[font=\sffamily\scriptsize, text=black!50, anchor=north] at ({lim:.3f},{y + 0.2:.3f}) {{30-step limit}};")
    ny = y_top - 0.6
    for line in note_lines:
        out.append(rf"\node[txt] at ({x0 + 10.6:.3f},{ny:.3f}) {{{line}}};")
        ny -= 0.42
    return out


def _effects_panel(x0, yt, family, tasks):
    lines = [rf"\node[ptitle] at ({x0:.3f},{yt:.3f}) {{(b) Effect of the memory on this family, 3 demonstration pairs}};",
             rf"\node[txt, text=black!65] at ({x0:.3f},{yt - 0.45:.3f}) {{{family}}};"]
    ax, half = x0 + 7.4, 2.4
    ytop = yt - 0.85
    yb = ytop - 0.15 - len(tasks) * 0.66
    lines.append(rf"\draw[black!45] ({ax:.3f},{ytop:.3f}) -- ({ax:.3f},{yb:.3f});")
    for xv, t in ((ax - half, "$-1$"), (ax, "$0$"), (ax + half, "$+1$")):
        lines.append(rf"\node[font=\sffamily\scriptsize, text=black!50] at ({xv:.3f},{yb - 0.17:.3f}) {{{t}}};")
    y = ytop - 0.12
    for name, g, q in tasks:
        lines.append(rf"\node[lab] at ({ax - half - 0.15:.3f},{y - 0.15:.3f}) {{{name}}};")
        for val, col, dy in ((g, "gptblue!55", 0.0), (q, "qwenmint!55", -0.29)):
            lines.append(rf"\fill[{col}] ({ax:.3f},{y + dy:.3f}) rectangle ++({val * half:.3f},-0.24);")
            txt = "$0$" if val == 0 else (f"$+{val:.3f}$" if val > 0 else f"$-{abs(val):.3f}$")
            anchor, xx = ("west", ax + max(val, 0) * half + 0.06) if val >= 0 else ("east", ax + val * half - 0.06)
            lines.append(rf"\node[anchor={anchor}, font=\sffamily\scriptsize, text=black!70] at ({xx:.3f},{y + dy - 0.12:.3f}) {{{txt}}};")
        y -= 0.66
    ly = yt - 1.2
    lines.append(rf"\fill[gptblue!55] ({x0:.3f},{ly:.3f}) rectangle ++(0.28,-0.18);")
    lines.append(rf"\node[txt] at ({x0 + 0.32:.3f},{ly - 0.09:.3f}) {{GPT-4o-mini}};")
    lines.append(rf"\fill[qwenmint!55] ({x0:.3f},{ly - 0.4:.3f}) rectangle ++(0.28,-0.18);")
    lines.append(rf"\node[txt] at ({x0 + 0.32:.3f},{ly - 0.49:.3f}) {{Qwen3-8B}};")
    return lines


def _hl(lines, x, y, k, box, color, text=None, at=None, anchor="center"):
    """Highlight a pixel box of a screenshot shown at (x, y) with scale k (cm per pixel);
    `at` is the label position in the same pixel coordinates (it may lie outside the shot)."""
    x0, y0, x1, y1 = x + box[0] * k, y - box[1] * k, x + box[2] * k, y - box[3] * k
    lines.append(rf"\draw[{color}, line width=1.2pt, rounded corners=1pt] ({x0:.3f},{y0:.3f}) rectangle ({x1:.3f},{y1:.3f});")
    if text:
        cx, cy = x + at[0] * k, y - at[1] * k
        lines.append(rf"\node[lblkey={color}, anchor={anchor}] (hl) at ({cx:.3f},{cy:.3f}) {{{text}}};")
        lines.append(rf"\draw[lead={color}] (hl) -- ({min(max(cx, x0), x1):.3f},{min(max(cy, y1), y0):.3f});")


def webshop_case(fname):
    """Session 0 walked through in the official WebShop app (screenshots in figures/cases/ws_*.jpg)."""
    lines = [_header()]
    lines.append(r"\node[ptitle] at (0,0) {(a) One WebShop session in the official web app (evaluation session 0)};")
    top, capy = -1.75, -1.65
    shots = {"search": (620, 245), "results": (900, 1432), "product": (900, 665), "score": (600, 325)}
    col = {"search": (0.0, 5.5), "results": (5.8, 3.95), "product": (10.05, 4.95)}
    caps = {"search": r"The start page shows the request and a search box.",
            "results": r"After \texttt{search[double sided machine washable decorative pillows 28 x 28]}: "
                       r"50 results, 10 per page.",
            "product": r"After \texttt{click[B0743JKHBV]}: the product page, with its size options and a "
                       r"\textit{Buy Now} button."}
    hs = {}
    for k, (x, w) in col.items():
        lines.append(rf"\node[anchor=south west, align=left, text width={w - 0.1:.2f}cm, font=\sffamily\scriptsize, "
                     rf"text=black!75, inner sep=1pt] at ({x:.3f},{capy:.3f}) {{{caps[k]}}};")
        hs[k] = _image(lines, f"ws_{k}.jpg", x, top, w, shots[k])
    xs, ws = col["product"]
    ys = top - hs["product"] - 1.2
    lines.append(rf"\node[anchor=south west, align=left, text width={ws - 0.1:.2f}cm, font=\sffamily\scriptsize, "
                 rf"text=black!75, inner sep=1pt] at ({xs:.3f},{ys + 0.08:.3f}) "
                 r"{After \texttt{click[28\textquotedbl{} x 28\textquotedbl{}]} and \texttt{click[Buy Now]}: "
                 r"the episode ends; the score box (green) shows reward 1.0.};")
    hs["score"] = _image(lines, "ws_score.jpg", xs, ys, ws, shots["score"])
    x, w = col["results"]
    _hl(lines, x, top, w / 900, (8, 1187, 892, 1430), "liftpositive", "the requested product (5th)", at=(450, 1505))
    x, w = col["product"]
    _hl(lines, x, top, w / 900, (155, 583, 223, 619), "dimA", r"\texttt{click[28\textquotedbl{} x 28\textquotedbl{}]}",
        at=(300, 601), anchor="west")
    _hl(lines, x, top, w / 900, (802, 204, 889, 243), "dimF", r"\texttt{click[Buy Now]}", at=(780, 168), anchor="east")
    _hl(lines, xs, ys, ws / 600, (14, 274, 586, 316), "liftpositive")
    x, w = col["search"]
    yb = top - hs["search"] - 0.3
    text = (r"\textbf{What the reward checks.} The purchase is compared with the request:\\[1pt]"
            r"\textbullet\ attributes: double sided, machine washable, printing technology\\"
            r"\textbullet\ option: size 28\textquotedbl{} x 28\textquotedbl{}\\"
            r"\textbullet\ price: lower than \$50.00\\"
            r"\textbullet\ product type: the title must match the request\\[2pt]"
            r"The reward is the share of matched attributes, options, and price, times the type match; "
            r"here everything matches, so it is 1.0. A session allows at most 15 actions, "
            r"and the store holds 1{,}000 products. The ranking shown comes from our BM25 stand-in "
            r"for the search index.")
    lines.append(rf"\node[textbox, text width={w - 0.3:.2f}cm] at ({x:.3f},{yb:.3f}) {{{text}}};")
    for a, b, yy in (("search", "results", top - 1.1), ("results", "product", top - 1.1)):
        x0 = col[a][0] + col[a][1] + 0.03
        lines.append(rf"\draw[flow, line width=1pt] ({x0:.3f},{yy:.3f}) -- ({col[b][0] - 0.03:.3f},{yy:.3f});")
    lines.append(rf"\draw[flow, line width=1pt] ({xs - 0.12:.3f},{top - hs['product'] + 0.3:.3f}) |- "
                 rf"({xs - 0.03:.3f},{ys - 0.5:.3f});")
    # panel (b)
    ybot = min(top - hs["results"] - 0.6, ys - hs["score"]) - 0.75
    lines.append(rf"\node[ptitle] at (0,{ybot:.3f}) {{(b) What happened: \texttt{{Qwen3-8B}}, 100 sessions}};")
    bx, scale = 1.75, 6.0 / 0.25
    for k, (label, v, c) in enumerate([("no memory", 0.1944, "phaseA!35"), ("memory", 0.1045, "salmon!60")]):
        y = ybot - 0.6 - k * 0.46
        lines.append(rf"\node[lab] at ({bx - 0.1:.3f},{y:.3f}) {{{label}}};")
        lines.append(rf"\fill[{c}, rounded corners=1pt] ({bx:.3f},{y + 0.15:.3f}) rectangle ++({v * scale:.3f},-0.3);")
        lines.append(rf"\node[txt] at ({bx + v * scale + 0.08:.3f},{y:.3f}) {{mean reward {_r3(v)}}};")
    sx, tot = 9.6, 5.4
    lines.append(rf"\node[txt] at ({sx:.3f},{ybot - 0.6:.3f}) {{paired change per session:}};")
    for name, n, c in [("declined", 20, "salmon!60"), ("unchanged", 72, "black!12"), ("improved", 8, "qwenmint!55")]:
        w = n / 100 * tot
        lines.append(rf"\fill[{c}] ({sx:.3f},{ybot - 0.9:.3f}) rectangle ++({w:.3f},-0.36);")
        lines.append(rf"\node[font=\sffamily\footnotesize] at ({sx + w / 2:.3f},{ybot - 1.08:.3f}) {{{n}}};")
        lines.append(rf"\node[font=\sffamily\scriptsize, text=black!65, anchor=north] at ({sx + w / 2:.3f},{ybot - 1.3:.3f}) {{{name}}};")
        sx += w
    lines.append(r"\end{tikzpicture}")
    (OUT / fname).write_text("\n".join(lines) + "\n")


def load(case):
    return json.loads((ASSETS / f"{case}.json").read_text())


def alfworld_task_case(fname, case_id, verb, two, rows, notes):
    case = load(case_id)
    lines = [_header()]
    yb = env_panel(lines, case, "(a) The task and its room (AI2-THOR rendering of the benchmark trial)", verb, two)
    lines += _steps_panel(0, yb - 0.8, rows, notes)
    lines.append(r"\end{scope}")
    lines.append(r"\end{tikzpicture}")
    (OUT / fname).write_text("\n".join(lines) + "\n")
    return case


def family_case(fname, case_id, verb, family, tasks):
    case = load(case_id)
    lines = [_header()]
    yb = env_panel(lines, case, "(a) An example task of this family and its room (AI2-THOR rendering)", verb, False)
    lines += _effects_panel(0, yb - 0.8, family, tasks)
    lines.append(r"\end{scope}")
    lines.append(r"\end{tikzpicture}")
    (OUT / fname).write_text("\n".join(lines) + "\n")
    return case


if __name__ == "__main__":
    # Task 0 (put): goal and room from the run log; rendered trial: one of the two trials whose goal text matches.
    alfworld_task_case("case_task0.tex", "task0", None, False,
                       [("expert", 4, "black!30", "4"), ("no memory", 19, "phaseA!35", "19, solved"),
                        ("memory", 6, "salmon!60", "6, solved")],
                       ["prompt tokens: 40,783 $\\rightarrow$ 17,850",
                        "effect over 3 demo pairs:", "\\quad GPT $0.000$, Qwen $-0.333$"])
    alfworld_task_case("case_task1.tex", "task1", "heat", False,
                       [("expert", 6, "black!30", "6"), ("no memory", 30, "phaseA!35", "30, not solved"),
                        ("memory", 30, "salmon!60", "30, not solved")],
                       ["prompt tokens: 81,372 $\\rightarrow$ 108,358",
                        "effect over 3 demo pairs:", "\\quad GPT $0.000$, Qwen $-0.333$"])
    alfworld_task_case("case_task2.tex", "task2", None, True,
                       [("expert", 8, "black!30", "8"), ("no memory", 30, "phaseA!35", "30, not solved"),
                        ("memory", 30, "salmon!60", "30, not solved")],
                       ["prompt tokens: 89,011 $\\rightarrow$ 109,495",
                        "effect over 3 demo pairs:", "\\quad not reported for this task"])
    family_case("case_clean.tex", "clean", "clean",
                "family success: GPT $0.452\\rightarrow0.667$, Qwen $0.656\\rightarrow0.613$",
                [("task 30", 0.667, -0.333), ("task 57", 0.0, 0.333), ("task 50", 0.0, -0.667)])
    family_case("case_cool.tex", "cool", "cool",
                "family success: GPT $0.857\\rightarrow0.746$, Qwen $0.857\\rightarrow0.857$",
                [("task 5", -0.5, -0.333), ("task 16", -0.333, 0.0), ("task 19", -0.333, 0.0), ("task 39", 0.0, 0.333)])
    family_case("case_examine.tex", "examine", "use",
                "family success: GPT $0.648\\rightarrow0.833$, Qwen $0.611\\rightarrow0.574$",
                [("task 7", 0.667, 0.333), ("task 45", 0.667, -0.667), ("task 89", -0.5, 0.0)])
    webshop_case("case_webshop.tex")
    print("wrote", sorted(p.name for p in OUT.glob("case_*.tex")))
