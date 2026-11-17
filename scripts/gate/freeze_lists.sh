#!/bin/bash
# Regenerate and verify the frozen-artifact lists for the WebShop release check (8B and 32B) and the analysis-code
# hash list, then print their digests for the pre-registration. Does NOT write PREREG_FROZEN (a separate, reviewed
# step). Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/../.."
shared=(data/pools/webshop1010_insights.jsonl data/pools/gate_ws_2026_11/* results/gate_ws_2026_11/injection_audit.json
        configs/webshop_react_fewshot.jsonl scripts/webshop_server.py skillrc/*.py skillrc/envs/*.py
        skillrc/gate/schema.py skillrc/gate/ingest/skillrc_jsonl.py
        data/summaries/webshop_2026-10-10/instructions_seed0_1k.json
        configs/gate_ws_2026_11/registry_T.jsonl configs/gate_ws_2026_11/registry_D.jsonl)
printf '%s\n' "${shared[@]}" configs/gate_ws_2026_11/GWS_*.yaml configs/gate_ws_2026_11/schedule.json \
  configs/gate_ws_2026_11/halves.json configs/gate_ws_2026_11/chain_B*_*.txt configs/gate_ws_2026_11/AUDIT_OK \
  scripts/gate/run_gate_ws_block.sge scripts/gate/prereq.py scripts/gate/phase_marker.py \
  examples/webshop_release_check/gate.yaml \
  | sort -u | xargs sha256sum > configs/gate_ws_2026_11/FROZEN.sha256
printf '%s\n' "${shared[@]}" configs/gate_ws32_2026_11/GWS32_*.yaml configs/gate_ws32_2026_11/schedule.json \
  configs/gate_ws32_2026_11/chain_*.txt configs/gate_ws32_2026_11/AUDIT_OK configs/gate_ws32_2026_11/registry_pilot.jsonl \
  scripts/gate/run_gate_ws32.sge scripts/gate/pilot32_check.py scripts/gate/trigger32_check.py scripts/gate/phase_marker.py \
  scripts/gate/prereq.py scripts/gate/decision_io.py \
  examples/webshop_release_check_32b/gate.yaml \
  | sort -u | xargs sha256sum > configs/gate_ws32_2026_11/FROZEN.sha256
{ find skillrc/gate -name "*.py" -o -name "*.j2" | sort | xargs sha256sum; sha256sum scripts/gate/*.py; } \
  > engineering/gate_code_sha256_2026-11-16.txt
sha256sum --quiet -c configs/gate_ws_2026_11/FROZEN.sha256
sha256sum --quiet -c configs/gate_ws32_2026_11/FROZEN.sha256
sha256sum --quiet -c engineering/gate_code_sha256_2026-11-16.txt
echo "8B  FROZEN.sha256  $(sha256sum configs/gate_ws_2026_11/FROZEN.sha256 | cut -c1-64)  ($(wc -l < configs/gate_ws_2026_11/FROZEN.sha256) files)"
echo "32B FROZEN.sha256  $(sha256sum configs/gate_ws32_2026_11/FROZEN.sha256 | cut -c1-64)  ($(wc -l < configs/gate_ws32_2026_11/FROZEN.sha256) files)"
echo "code list          $(sha256sum engineering/gate_code_sha256_2026-11-16.txt | cut -c1-64)  ($(wc -l < engineering/gate_code_sha256_2026-11-16.txt) files)"
echo "8B analysis spec   $(sha256sum examples/webshop_release_check/gate.yaml | cut -c1-64)"
echo "32B analysis spec  $(sha256sum examples/webshop_release_check_32b/gate.yaml | cut -c1-64)"
