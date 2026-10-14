"""Minimal HTTP wrapper around WebShop's WebAgentTextEnv.

Runs INSIDE the webshop conda env (py3.8). The harness (skillrc env) calls it over
HTTP so WebShop's pydantic-v1 world never mixes with the harness's openai/pydantic-v2.

Run:  ${WEBSHOP_PY} scripts/webshop_server.py --port 3000
Determinism: product prices and the price caps in synthetic instructions are drawn with
Python's `random` while WebShop builds its goals, before its own random.seed(233). This
wrapper seeds `random` (default --seed 0) before building the environment, so the same
session id gives the same instruction across restarts. Start it with PYTHONHASHSEED=0.

Endpoints:
  POST /reset {"session": <int>}      -> {obs, actions, category}  (category: goal's product
                                         category, for stratified analysis only; never shown
                                         to the agent)
  POST /step  {"action": "search[..]"} -> {obs, reward, done, actions}
  GET  /health                         -> {ok: true}
"""
import argparse
import random
import sys

from flask import Flask, jsonify, request

app = Flask(__name__)
ENV = None
INFO = {}


def _first(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def _actions():
    try:
        return ENV.get_available_actions()   # {'has_search_bar':bool,'clickables':[...]}
    except Exception:
        return None


@app.route("/health")
def health():
    return jsonify({"ok": True, **INFO})


@app.route("/reset", methods=["POST"])
def reset():
    session = (request.get_json(silent=True) or {}).get("session", 0)
    obs = ENV.reset(session=session)
    try:
        category = ENV.server.user_sessions[ENV.session]["goal"].get("category", "")
    except Exception:
        category = ""
    return jsonify({"obs": str(_first(obs)), "actions": _actions(), "category": category})


@app.route("/step", methods=["POST"])
def step():
    action = (request.get_json(silent=True) or {}).get("action", "")
    obs, reward, done, info = ENV.step(action)
    return jsonify({"obs": str(_first(obs)), "reward": float(reward or 0.0),
                    "done": bool(done), "actions": _actions()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3000)
    ap.add_argument("--num-products", type=int, default=None,
                    help="limit product set (e.g. 1000 for the small dev index)")
    ap.add_argument("--observation-mode", default="text")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for Python random before WebShop builds prices and goals")
    a = ap.parse_args()
    random.seed(a.seed)
    import numpy as np
    np.random.seed(a.seed)
    from web_agent_site.envs import WebAgentTextEnv
    kw = {"observation_mode": a.observation_mode}
    if a.num_products is not None:
        kw["num_products"] = a.num_products
    ENV = WebAgentTextEnv(**kw)
    INFO.update({"seed": a.seed, "num_products": a.num_products, "n_goals": len(ENV.server.goals)})
    print(f"WebShop server ready on :{a.port} (num_products={a.num_products})", file=sys.stderr)
    app.run(host="127.0.0.1", port=a.port, threaded=False)
