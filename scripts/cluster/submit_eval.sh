#!/usr/bin/env bash
# Friendly wrapper around qsub submission for skillrc eval jobs.
#
# Why this exists:
# - CRC rejects some UGE submissions before the job even starts if the submitting
#   shell lacks a valid Kerberos/AFS token.
# - The raw qsub error is easy to miss; this script fails early with a clear hint.
#
# Example:
#   bash scripts/submit_eval.sh \
#     CONFIG=configs/pilot_equal_footprint_alfworld.yaml \
#     RUN_ID=PILOT_eqfp_expel_bm25_s0_t20 \
#     MODEL=gpt-4o-mini MAXTASKS=20 SEEDS=0 \
#     MEMORY=expel RETRIEVER=bm25 POOL=results/pools/expel_insights.jsonl
set -euo pipefail

ROOT=${SKILLRC_ROOT}
SCRIPT="$ROOT/scripts/run.sge"

if ! command -v qsub >/dev/null 2>&1; then
  echo "qsub not found. This helper is for the CRC UGE/SGE path." >&2
  exit 1
fi

if ! klist >/dev/null 2>&1; then
  cat >&2 <<'EOF'
No Kerberos credentials cache found.

Before submitting GPU jobs that touch /groups, run:
  kinit
  aklog

Then re-run this submit helper.
EOF
  exit 2
fi

if ! command -v aklog >/dev/null 2>&1; then
  echo "Warning: aklog not found in PATH. If qsub rejects the job, refresh AFS manually." >&2
else
  aklog >/dev/null 2>&1 || true
fi

if [ "$#" -eq 0 ]; then
  cat >&2 <<'EOF'
Usage:
  bash scripts/submit_eval.sh KEY=VALUE [KEY=VALUE ...]

All KEY=VALUE pairs are forwarded to qsub via -v, for example:
  bash scripts/submit_eval.sh \
    CONFIG=configs/m0_alfworld.yaml \
    RUN_ID=R001_small \
    MODEL=gpt-4o-mini \
    MAXTASKS=5 \
    SEEDS=0
EOF
  exit 1
fi

VARS=()
for kv in "$@"; do
  VARS+=("$kv")
done

(
  cd "$ROOT"
  qsub -v "$(IFS=,; echo "${VARS[*]}")" "$SCRIPT"
)
