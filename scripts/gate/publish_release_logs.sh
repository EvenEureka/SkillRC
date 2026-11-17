#!/bin/bash
# Copy the WebShop release-check evidence from results/ (not tracked) into data/summaries/ (tracked), compressed:
# episodes, attempt journals, manifests, completion markers, decisions, injection audit. The frozen analysis specs
# read results/...; scripts/gate/restore_release_logs.sh puts the files back there for a reproduction.
set -euo pipefail
cd "$(dirname "$0")/../.."
for study in gate_ws_2026_11 gate_ws32_2026_11; do
  src=results/$study; dst=data/summaries/$study
  [ -d "$src" ] || continue
  mkdir -p "$dst"
  for f in "$src"/*.episodes.jsonl "$src"/*.attempts.jsonl; do [ -f "$f" ] && gzip -n -9 -c "$f" > "$dst/$(basename "$f").gz"; done
  for f in "$src"/*.manifest.json "$src"/*_DONE "$src"/PILOT_PASSED "$src"/D_TRIGGERED "$src"/*decision.json \
           "$src"/injection_audit.json "$src"/jobs.json; do [ -f "$f" ] && cp "$f" "$dst/"; done
  echo "$study: $(ls "$dst" | wc -l) files, $(du -sh "$dst" | cut -f1)"
done
