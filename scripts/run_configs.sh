#!/usr/bin/env bash
# Run SkillRC configs concurrently against one OpenAI-compatible server, such as a local vLLM server.
#
# usage: bash scripts/run_configs.sh <base_url> <config.yaml> [<config.yaml> ...]
#   e.g. bash scripts/run_configs.sh http://127.0.0.1:8000/v1 configs/w1_2026_10_13/eval/*_Q8.yaml
#
# All configs given in one call run at the same time, so the conditions of a comparison share one
# time window and one server. Each run writes results/<run_id>.episodes.jsonl and
# results/<run_id>.summary.json, with its log in logs/<run_id>.log; existing outputs are never
# overwritten. Set PYTHON to choose the interpreter (default: python).
set -euo pipefail
cd "$(dirname "$0")/.."
URL=${1:?usage: run_configs.sh <base_url> <config.yaml> [<config.yaml> ...]}
shift
[ "$#" -gt 0 ] || { echo "no config given" >&2; exit 2; }
PY=${PYTHON:-python}
mkdir -p logs results
for c in "$@"; do
  run=$(awk '/^run_id:/{print $2}' "$c")
  for o in "results/$run.episodes.jsonl" "results/$run.summary.json"; do
    [ ! -e "$o" ] || { echo "refusing to overwrite $o" >&2; exit 2; }
  done
done
pids=()
for c in "$@"; do
  run=$(awk '/^run_id:/{print $2}' "$c")
  "$PY" -m skillrc.runner --config "$c" --override base_url="$URL" > "logs/$run.log" 2>&1 &
  pids+=($!)
  echo "started $run"
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ "$rc" -eq 0 ] && echo "all runs finished" || echo "a run failed; see logs/" >&2
exit "$rc"
