#!/bin/bash
# Restore the published WebShop release-check evidence into results/ so the frozen specs
# (examples/webshop_release_check*/gate.yaml) reproduce the reports:  bash scripts/gate/restore_release_logs.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
for study in gate_ws_2026_11 gate_ws32_2026_11; do
  src=data/summaries/$study; dst=results/$study
  [ -d "$src" ] || continue
  mkdir -p "$dst"
  for f in "$src"/*.gz; do [ -f "$f" ] && gunzip -c "$f" > "$dst/$(basename "$f" .gz)"; done
  find "$src" -maxdepth 1 -type f ! -name "*.gz" -exec cp {} "$dst/" \;
  echo "$study: restored $(ls "$dst" | wc -l) files"
done
