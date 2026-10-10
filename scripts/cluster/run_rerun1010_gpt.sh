#!/bin/bash
# 2026-10-10 rerun of the placebo gate with the coherent task-irrelevant control, GPT-4o-mini executor.
# API-bound: runs the four conditions as four background processes on the login node (no GPU needed).
# The key is read from ~/.skillrc.env (a line `OPENAI_API_KEY=sk-...`, chmod 600), never from the shell history.
# usage: bash scripts/cluster/run_rerun1010_gpt.sh            (from the repository root)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
PY=${SKILLRC_PY:?set SKILLRC_PY to the harness python (see README)}
mkdir -p logs
export ALFWORLD_DATA="$ROOT/data/alfworld"
if [ -f "$HOME/.skillrc.env" ]; then set -a; source "$HOME/.skillrc.env"; set +a; fi
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be configured in ~/.skillrc.env}"
for cond in none true placebo irrelevant; do
  for output in results/RERUN1010_${cond^^}_GPT.episodes.jsonl results/RERUN1010_${cond^^}_GPT.summary.json; do
    if [ -e "$output" ]; then echo "refusing to overwrite existing output: $output"; exit 2; fi
  done
done
PYTHONPATH=. "$PY" scripts/analysis/build_order_placebo.py >/dev/null
AUDIT_PY=${AUDIT_PY:-$PY}   # needs transformers for the Qwen tokenizer audit
HF_HUB_OFFLINE=1 PYTHONPATH=. "$AUDIT_PY" scripts/analysis/build_irrelevant_control.py >/dev/null
# record which snapshot the alias resolves to today (the paper's alias is mutable)
"$PY" - <<'EOF' > logs/rerun1010_gpt_model_snapshot.json
import json, os
from openai import OpenAI
c = OpenAI()
r = c.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "Reply with the single word ok."}], max_completion_tokens=5, temperature=0)
print(json.dumps({"alias": "gpt-4o-mini", "resolved_model": r.model, "system_fingerprint": r.system_fingerprint}))
EOF
cat logs/rerun1010_gpt_model_snapshot.json
for cond in none true placebo irrelevant; do
  nohup "$PY" -m skillrc.runner --config configs/rerun_2026_10_10/${cond}_gpt.yaml > logs/rerun1010_${cond}_gpt.log 2>&1 &
  echo "started $cond (pid $!)"
done
echo "four GPT conditions running in the background; logs in logs/rerun1010_*_gpt.log"
