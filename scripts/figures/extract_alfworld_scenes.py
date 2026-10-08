#!/usr/bin/env python3
"""Extract the six case-study trials from the ALFWorld release archives.

For each case it writes SCENE_DIR/<case>.json with the ALFRED scene state (floor plan,
object poses, toggles), the expert walkthrough and goal text of the ALFWorld game, and
the room inventory (unique receptacle objects of the game's PDDL problem).

Archives (ALFWorld GitHub releases):
  json_2.1.1_json.zip      https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip
  json_2.1.2_tw-pddl.zip   https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip

Usage: extract_alfworld_scenes.py JSON_ZIP TWPDDL_ZIP SCENE_DIR
"""
import ast
import collections
import json
import re
import sys
import zipfile
from pathlib import Path

json_zip, pddl_zip, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

# Cases 1-3: trials whose goal text matches the run log (task 1: the countertop variant).
# Cases 4-6: one example task per family; the representative tasks' goals were not retained.
CASES = {
    "task0": "pick_and_place_simple-Mug-None-Desk-308/trial_T20190908_125200_737896",
    "task1": "pick_heat_then_place_in_recep-Egg-None-GarbageCan-10/trial_T20190908_113432_673307",
    "task2": "pick_two_obj_and_place-Pillow-None-Sofa-219/trial_T20190907_163327_486300",
    "clean": "pick_clean_then_place_in_recep-SoapBar-None-CounterTop-424/trial_T20190907_074106_050405",
    "cool": "pick_cool_then_place_in_recep-Mug-None-Cabinet-10/trial_T20190909_121559_082363",
    "examine": "look_at_obj_in_light-Book-None-DeskLamp-308/trial_T20190908_020029_636862",
}

jz, tz = zipfile.ZipFile(json_zip), zipfile.ZipFile(pddl_zip)
for case, trial in CASES.items():
    traj = json.loads(jz.read(f"json_2.1.1/valid_unseen/{trial}/traj_data.json"))
    game = json.loads(tz.read(f"json_2.1.1/valid_unseen/{trial}/game.tw-pddl"))
    walk = game["walkthrough"]
    walk = ast.literal_eval(walk) if isinstance(walk, str) else walk
    objs = sorted(set(re.findall(r"(\S+) - receptacle\b", game["pddl_problem"])))
    objs = [o for o in objs if o[0].isalpha()]
    recs = collections.Counter(o.split("_bar_")[0] for o in objs)
    goal = re.search(r'"task": \[\s*\{\s*"rhs": "Your task is to: ([^"]*)"', game["grammar"]).group(1)
    json.dump({"trial": trial, "scene": traj["scene"], "walkthrough": walk, "pddl_params": traj["pddl_params"],
               "receptacles": dict(sorted(recs.items())), "receptacle_objects": objs, "goal": goal},
              open(out / f"{case}.json", "w"))
    print(case, traj["scene"]["floor_plan"], goal, dict(sorted(recs.items())))
