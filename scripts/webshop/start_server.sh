#!/bin/bash
# Start the WebShop text environment (scripts/webshop_server.py) in the background and
# wait until GET /health answers. No GPU needed.
#
# usage: bash scripts/webshop/start_server.sh <port> [num_products]   (default 1000)
#   env: WEBSHOP_ENV  conda env with WebShop's deps (python 3.8, pyserini, openjdk 11) [required]
#        WEBSHOP_DIR  WebShop source tree with data/ and search_engine/indexes_1k
#                     (default: <repo>/third_party/WebShop; see README, "WebShop")
#        WS_LOG       server log (default: <repo>/logs/webshop_server.<port>.log)
#        WS_WAIT      seconds to wait for /health (default 300)
#        WS_SEED      seed for prices and instruction price caps (default 0)
# On success prints the PID (also written to logs/webshop_server.<port>.pid).
# Stop with:  kill "$(cat logs/webshop_server.<port>.pid)"
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PORT=${1:?usage: start_server.sh <port> [num_products]}
NUM=${2:-1000}
WEBSHOP_ENV=${WEBSHOP_ENV:?set WEBSHOP_ENV to the WebShop conda env (see README, "WebShop")}
WEBSHOP_DIR=${WEBSHOP_DIR:-$ROOT/third_party/WebShop}
WS_WAIT=${WS_WAIT:-300}
mkdir -p "$ROOT/logs"
WS_LOG=${WS_LOG:-$ROOT/logs/webshop_server.$PORT.log}
PIDFILE=$ROOT/logs/webshop_server.$PORT.pid

[ -d "$WEBSHOP_DIR/web_agent_site" ] || { echo "no WebShop tree at $WEBSHOP_DIR" >&2; exit 1; }
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT already serves /health; refusing to start a second server" >&2; exit 2
fi

# pyserini needs the env's JDK 11 (a system Java 8 fails at import); keep ~/.local out.
env JAVA_HOME="$WEBSHOP_ENV" JVM_PATH="$WEBSHOP_ENV/lib/jvm/lib/server/libjvm.so" \
    PATH="$WEBSHOP_ENV/bin:$PATH" PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 PYTHONPATH="$WEBSHOP_DIR" \
    nohup "$WEBSHOP_ENV/bin/python" "$ROOT/scripts/webshop_server.py" \
    --port "$PORT" --num-products "$NUM" --seed "${WS_SEED:-0}" > "$WS_LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"

for _ in $(seq 1 "$WS_WAIT"); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "$PID"
    echo "WebShop server up on 127.0.0.1:$PORT (pid $PID, num_products=$NUM, log $WS_LOG)" >&2
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "WebShop server exited during startup; last log lines:" >&2; tail -30 "$WS_LOG" >&2
    rm -f "$PIDFILE"; exit 1
  fi
  sleep 1
done
echo "WebShop server not healthy after ${WS_WAIT}s; killing pid $PID" >&2
tail -30 "$WS_LOG" >&2
kill "$PID" 2>/dev/null || true
rm -f "$PIDFILE"
exit 1
