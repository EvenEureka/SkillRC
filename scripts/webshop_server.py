"""Minimal HTTP wrapper around WebShop's WebAgentTextEnv.

Runs INSIDE the webshop conda env (py3.8). The harness (skillrc env) calls it over
HTTP so WebShop's pydantic-v1 world never mixes with the harness's openai/pydantic-v2.

Run:  ${WEBSHOP_PY} scripts/webshop_server.py --port 3000
Endpoints:
  POST /reset {"session": <int>}      -> {obs, actions}
  POST /step  {"action": "search[..]"} -> {obs, reward, done, actions}
  GET  /health                         -> {ok: true}
"""
import argparse
import sys

from flask import Flask, jsonify, request

app = Flask(__name__)
ENV = None


def _first(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def _actions():
    try:
        return ENV.get_available_actions()   # {'has_search_bar':bool,'clickables':[...]}
    except Exception:
        return None


@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.route("/reset", methods=["POST"])
def reset():
    session = (request.get_json(silent=True) or {}).get("session", 0)
    obs = ENV.reset(session=session)
    return jsonify({"obs": str(_first(obs)), "actions": _actions()})


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
    a = ap.parse_args()
    from web_agent_site.envs import WebAgentTextEnv
    kw = {"observation_mode": a.observation_mode}
    if a.num_products is not None:
        kw["num_products"] = a.num_products
    ENV = WebAgentTextEnv(**kw)
    print(f"WebShop server ready on :{a.port} (num_products={a.num_products})", file=sys.stderr)
    app.run(host="127.0.0.1", port=a.port, threaded=False)
