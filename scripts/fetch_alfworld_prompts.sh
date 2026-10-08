#!/usr/bin/env bash
# Fetch the canonical ReAct ALFWorld demonstrations (3 per task type; keys react_<type>_<i>)
# used by `prompt_style: react_alfworld`. Source: Yao et al. (2023), github.com/ysymyth/ReAct.
set -euo pipefail
cd "$(dirname "$0")/.."
curl -fsSL -o configs/alfworld_react_fewshot.json \
  https://raw.githubusercontent.com/ysymyth/ReAct/master/prompts/alfworld_3prompts.json
python3 -c "import json;d=json.load(open('configs/alfworld_react_fewshot.json'));print(len([k for k in d if k.startswith('react_')]),'react prompts')"
