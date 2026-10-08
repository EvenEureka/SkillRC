#!/usr/bin/env python3
"""Turn the AI2-THOR renders of render_cases.py into the case-study figure assets.

For every case it writes paper/figures/cases/<case>_view<k>.jpg, <case>_map.jpg (the
top-down view cropped to the room) and <case>.json with the goal, room inventory, expert
walkthrough, object roles, the 2D boxes of the labelled objects in each view, and the
pixel position of every labelled object in the cropped map. Map positions are exact: the
map camera is orthographic, so a world point (x, z) lands at
u = W/2 + (x - cam_x) * s,  v = H/2 - (z - cam_z) * s,  s = (H/2) / orthographic_size.

Usage: prepare_case_assets.py RENDER_DIR [OUT_DIR]
"""
import json
import sys
from pathlib import Path

from PIL import Image

RENDER = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / "paper/figures/cases"
OUT.mkdir(parents=True, exist_ok=True)
CASES = ["task0", "task1", "task2", "clean", "cool", "examine"]
THOR_TYPE = {"Sink": "SinkBasin"}
EXTRA = {"examine": ["DeskLamp"]}      # objects (not receptacles) that the expert solution uses


def center(o):
    corners = (o.get("objectBounds") or {}).get("objectBoundsCorners")
    if corners:
        return (sum(p["x"] for p in corners) / len(corners), sum(p["z"] for p in corners) / len(corners))
    return o["position"]["x"], o["position"]["z"]


def footprint(o):
    corners = (o.get("objectBounds") or {}).get("objectBoundsCorners")
    if not corners:
        return None
    xs, zs = [p["x"] for p in corners], [p["z"] for p in corners]
    return min(xs), min(zs), max(xs), max(zs)


for case in [c for c in CASES if (RENDER / f"{c}_meta.json").exists()]:
    meta = json.loads((RENDER / f"{case}_meta.json").read_text())
    rec_types = [THOR_TYPE.get(t, t) for t in meta["receptacles"]]
    target = {o.split("|")[0] for o, r in meta["roles"].items() if r == "object"}
    keep = set(rec_types) | target | set(EXTRA.get(case, [])) | {o.split("|")[0] for o in meta["roles"]}

    views = []
    for v in meta["views"]:
        im = Image.open(RENDER / v["file"]).convert("RGB")
        name = v["file"].replace(".png", ".jpg")
        im.save(OUT / name, quality=88)
        views.append({"file": name, "size": im.size,
                      "boxes": {o: b for o, b in v["boxes"].items() if o.split("|")[0] in keep}})

    mp = meta["map"]
    W, H = mp["width"], mp["height"]
    cam, orth = mp["cameraPosition"], mp["cameraOrthSize"]
    s = (H / 2) / orth

    def proj(x, z):
        return W / 2 + (x - cam["x"]) * s, H / 2 - (z - cam["z"]) * s

    # crop to the furnished area: the extent of all object positions plus a margin
    # (AI2-THOR 2.1.0 leaves metadata["sceneBounds"] empty)
    xs = [o["position"]["x"] for o in mp["objects"]] + [f[k] for f in map(footprint, mp["objects"]) if f for k in (0, 2)]
    zs = [o["position"]["z"] for o in mp["objects"]] + [f[k] for f in map(footprint, mp["objects"]) if f for k in (1, 3)]
    margin = 0.55
    u0, v0 = proj(min(xs) - margin, max(zs) + margin)
    u1, v1 = proj(max(xs) + margin, min(zs) - margin)
    pad = 0
    box = (max(0, int(u0) - pad), max(0, int(v0) - pad), min(W, int(u1) + pad), min(H, int(v1) + pad))
    im = Image.open(RENDER / mp["file"]).convert("RGB").crop(box)
    im.save(OUT / f"{case}_map.jpg", quality=88)
    objs = []
    for o in mp["objects"]:
        if o["objectType"] not in keep:
            continue
        u, v = proj(*center(o))
        fp = footprint(o)
        item = {"objectId": o["objectId"], "type": o["objectType"], "uv": [round(u - box[0], 1), round(v - box[1], 1)]}
        if fp:
            a, b = proj(fp[0], fp[3])
            c, d = proj(fp[2], fp[1])
            item["box"] = [round(a - box[0], 1), round(b - box[1], 1), round(c - box[0], 1), round(d - box[1], 1)]
        objs.append(item)
    for v, mv in zip(views, meta["views"]):   # where each first-person view was taken, on the map
        u, w = proj(mv["position"]["x"], mv["position"]["z"])
        v["map_uv"], v["rotation"], v["horizon"] = [round(u - box[0], 1), round(w - box[1], 1)], mv["rotation"], mv["horizon"]
    out = {k: meta[k] for k in ("case", "trial", "floor_plan", "goal", "receptacles", "walkthrough", "roles")}
    out["views"] = views
    out["map"] = {"file": f"{case}_map.jpg", "size": im.size, "px_per_m": round(s, 2), "objects": objs}
    (OUT / f"{case}.json").write_text(json.dumps(out, indent=1))
    print(case, meta["floor_plan"], "views", len(views), "map crop", box, "scale px/m", round(s, 1), "objects", len(objs))
