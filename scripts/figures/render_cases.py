#!/usr/bin/env python3
"""Render the six ALFWorld case scenes with AI2-THOR 2.1.0 (the version ALFWorld uses).

For each case the ALFRED trial state (object poses, toggles, dirty/empty) is restored as in
alfworld/env/thor_env.py. A low-resolution pass searches reachable viewpoints for one (or
two) first-person views that show the receptacles of the expert solution; a full-resolution
pass then saves those views with their 2D instance boxes, plus the top-down map view with
the orthographic camera parameters and every object's 3D bounds, so map labels can be
projected exactly. No model is run.

Usage: render_cases.py SCENE_DIR OUT_DIR   (needs an X display; we use Xvfb with GLX)
"""
import json, os, re, sys

import numpy as np
from PIL import Image
import ai2thor.controller as ac

ac.Controller.check_x_display = lambda self, x: None  # xdpyinfo is not installed on the nodes
ac.Controller.lock_release = lambda self: None   # flock(LOCK_SH) on a "w" handle fails on NFS home
ac.Controller.unlock_release = lambda self: None

scene_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
CASES = ["task0", "task1", "task2", "clean", "cool", "examine"]
TARGET = {"task0": "Mug", "task1": "Egg", "task2": "Pillow", "clean": "SoapBar", "cool": "Mug", "examine": "Book"}
THOR_TYPE = {"sink": "SinkBasin"}  # ALFWorld receptacle name -> THOR object type used for its box


def thor_type(name):
    return THOR_TYPE.get(name.lower(), name)


def parse_tok(tok):
    """'sink_bar__minus_00_dot_30_..._bar_sinkbasin' -> ('sinkbasin', (-0.30, 0.80, 3.26))."""
    m = re.match(r"([a-z]+)_bar__(.+)$", tok)
    if not m:
        return None
    nums = [(-1 if s == "minus" else 1) * float(f"{a}.{b}")
            for s, a, b in re.findall(r"(minus|plus)_(\d+)_dot_(\d+)", m.group(2))]
    suffix = re.search(r"_bar_([a-z]+)$", m.group(2))
    return (suffix.group(1) if suffix else m.group(1)), tuple(nums[:3])


def match_object(objs, name, xyz):
    cands = [o for o in objs if o["objectType"].lower() == name]
    if not cands:
        return None
    return min(cands, key=lambda o: sum((o["position"][k] - v) ** 2 for k, v in zip("xyz", xyz)))["objectId"]


def restore(c, sc):
    c.reset(sc["floor_plan"])
    c.step(dict(action="Initialize", gridSize=0.25, cameraY=0.75, renderImage=True, renderDepthImage=False,
                renderClassImage=False, renderObjectImage=True, visibility_distance=1.5, makeAgentsVisible=False,
                fieldOfView=90))
    if sc.get("object_toggles"):
        c.step(dict(action="SetObjectToggles", objectToggles=sc["object_toggles"]))
    if sc.get("dirty_and_empty"):
        c.step(dict(action="SetStateOfAllObjects", StateChange="CanBeDirty", forceAction=True))
        c.step(dict(action="SetStateOfAllObjects", StateChange="CanBeFilled", forceAction=False))
    return c.step(dict(action="SetObjectPoses", objectPoses=sc["object_poses"]))


def roles_of(info, objs):
    roles = {}
    for step in info["walkthrough"]:
        for t in step.split():
            p = parse_tok(t)
            if not p:
                continue
            oid = match_object(objs, p[0], p[1])
            if oid is None:
                print("  unmatched", t, flush=True)
                continue
            if oid.split("|")[0] == TARGET[info["case"]]:
                roles.setdefault(oid, "object")
            elif step.startswith("take"):
                roles[oid] = "start"
            elif step.startswith(("heat", "cool", "clean", "use")):
                roles[oid] = "appliance"
            elif step.startswith("move"):
                roles[oid] = "dest"
    return roles


def boxes(event, types):
    det = getattr(event, "instance_detections2D", None) or {}
    return {oid: [int(v) for v in b] for oid, b in det.items() if oid.split("|")[0] in types}


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


infos = {}
for case in CASES:
    info = json.load(open(os.path.join(scene_dir, f"{case}.json")))
    info["case"] = case
    info["label_types"] = sorted({thor_type(k) for k in info["receptacles"] if k[0].isalpha()})
    infos[case] = info

# ---- pass 1: viewpoint search at low resolution -------------------------------------------
LW, LH = 400, 300
c = ac.Controller(quality="Very Low")
c.start(player_screen_width=LW, player_screen_height=LH)
views = {}
for case in CASES:
    info = infos[case]
    ev = restore(c, info["scene"])
    roles = roles_of(info, ev.metadata["objects"])
    info["roles"] = roles
    keys = [o for o, r in roles.items() if r in ("start", "appliance", "dest")]
    target = [o for o, r in roles.items() if r == "object"]
    types = set(info["label_types"]) | {o.split("|")[0] for o in roles}
    reach = c.step(dict(action="GetReachablePositions")).metadata["reachablePositions"]
    rng = np.random.default_rng(0)
    sample = [reach[i] for i in rng.choice(len(reach), size=min(220, len(reach)), replace=False)]
    cands = []
    for pos in sample:
        for rot in (0, 90, 180, 270):
            for hor in (0, 30):
                e = c.step(dict(action="TeleportFull", x=pos["x"], y=pos["y"], z=pos["z"], rotation=rot, horizon=hor))
                if not e.metadata["lastActionSuccess"]:
                    continue
                b = boxes(e, types)
                # a key place counts in proportion to how much of the image it fills (full credit at 3%)
                vis = {k: min(1.0, area(b[k]) / (0.03 * LW * LH)) for k in keys if k in b and area(b[k]) >= 0.004 * LW * LH}
                seen_tgt = {k for k in target if k in b}
                ntypes = len({o.split("|")[0] for o, bb in b.items() if area(bb) >= 0.002 * LW * LH})
                cands.append(((pos, rot, hor), vis, seen_tgt, ntypes))

    def best(want_keys):
        return max(cands, key=lambda t: (10 * sum(v for k, v in t[1].items() if k in want_keys)
                                         + 6 * len(t[2]) + t[3]))

    v1 = best(set(keys))
    chosen = [v1[0]]
    missing = set(keys) - set(v1[1])
    if missing:
        v2 = best(missing)
        if set(v2[1]) & missing:
            chosen.append(v2[0])
    views[case] = chosen
    print(case, info["scene"]["floor_plan"], "keys", keys, "view1 sees", {k: round(v, 2) for k, v in v1[1].items()}, "missing", sorted(missing),
          "n views", len(chosen), flush=True)
c.stop()

# ---- pass 2: full-resolution renders ---------------------------------------------------------
W, H = 1200, 900
c = ac.Controller(quality="Ultra")
c.start(player_screen_width=W, player_screen_height=H)
for case in CASES:
    info = infos[case]
    restore(c, info["scene"])
    types = set(info["label_types"]) | {o.split("|")[0] for o in info["roles"]}
    meta = {"case": case, "trial": info["trial"], "floor_plan": info["scene"]["floor_plan"], "goal": info["goal"],
            "receptacles": {k: v for k, v in info["receptacles"].items() if k[0].isalpha()},
            "walkthrough": info["walkthrough"], "roles": info["roles"], "views": []}
    for k, (pos, rot, hor) in enumerate(views[case]):
        e = c.step(dict(action="TeleportFull", x=pos["x"], y=pos["y"], z=pos["z"], rotation=rot, horizon=hor))
        Image.fromarray(e.frame).save(os.path.join(out_dir, f"{case}_view{k + 1}.png"))
        meta["views"].append({"file": f"{case}_view{k + 1}.png", "position": pos, "rotation": rot, "horizon": hor,
                              "boxes": boxes(e, types)})
    m = c.step(dict(action="ToggleMapView"))
    Image.fromarray(m.frame).save(os.path.join(out_dir, f"{case}_map.png"))
    md = m.metadata
    meta["map"] = {"file": f"{case}_map.png", "width": W, "height": H,
                   "cameraPosition": md.get("cameraPosition"), "cameraOrthSize": md.get("cameraOrthSize"),
                   "sceneBounds": md.get("sceneBounds"),
                   "objects": [{"objectId": o["objectId"], "objectType": o["objectType"], "position": o["position"],
                                "objectBounds": o.get("objectBounds"), "parentReceptacles": o.get("parentReceptacles")}
                               for o in md["objects"]]}
    c.step(dict(action="ToggleMapView"))
    json.dump(meta, open(os.path.join(out_dir, f"{case}_meta.json"), "w"), indent=1)
    print(case, "rendered", len(meta["views"]), "views; orth", md.get("cameraOrthSize"), md.get("cameraPosition"), flush=True)
c.stop()
print("done", flush=True)
