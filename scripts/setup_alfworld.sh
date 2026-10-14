#!/usr/bin/env bash
# Prepare ALFWorld for the harness: game data under data/alfworld, the official base config,
# and the ReAct demonstrations (3 per task type), both pinned to the commits we used. Run once from any
# directory, with the python environment of requirements.txt active (it provides `alfworld-download`).
# data/alfworld_tasks/valid_unseen_gamefiles.json lists the game behind each of the 134 test task ids.
set -euo pipefail
cd "$(dirname "$0")/.."
export ALFWORLD_DATA="${ALFWORLD_DATA:-$PWD/data/alfworld}"
mkdir -p "$ALFWORLD_DATA"
alfworld-download -f
curl -fsSL -o configs/alfworld_base_config.yaml \
  https://raw.githubusercontent.com/alfworld/alfworld/f26512b22537a33de1f7c240bbba7196ab8d7500/configs/base_config.yaml
bash scripts/fetch_alfworld_prompts.sh
echo "ALFWorld ready: ALFWORLD_DATA=$ALFWORLD_DATA"
