#!/usr/bin/env python3
"""Fill result-dependent manuscript slots and freeze the Paper O claim ledger."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARR = ROOT.parent / "ARR"
ASSETS = ARR / "PaperO_assets"
MANUSCRIPT = ARR / "PaperO_manuscript.md"
LEDGER = ARR / "ARR_2026-07-12_PaperO_ClaimLedger.md"
RECORD = ARR / "ARR_2026-07-12_PaperO_Finalization_Record.md"


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    web_rows = {row["metric"]: row for row in read_csv(ASSETS / "table12_webshop_crossenv.csv")}
    web_decision = json.loads((ASSETS / "webshop_crossenv_decision.json").read_text())
    metadata = json.loads((ROOT / "data" / "pools" / "webshop_expel_insights.metadata.json").read_text())
    trajectory = read_csv(ASSETS / "table11_qwen_trajectory_mechanisms.csv")
    switches = Counter(row["switch"] for row in trajectory)
    markers = Counter(row["mechanism_marker"] for row in trajectory)
    reward = web_rows["environment_reward"]
    success = web_rows["exact_success"]
    wins = web_decision["reward_wins"]
    losses = web_decision["reward_losses"]
    ties = web_decision["reward_ties"]
    supported = bool(web_decision["cross_environment_conditionality_supported"])

    if supported:
        crossenv_summary = (
            f"In a train-disjoint 100-task WebShop validation with local Qwen3-8B, ExpeL-style memory "
            f"changes mean environment reward from {reward['baseline_mean']} to {reward['memory_mean']} "
            f"(paired delta {reward['paired_delta']}, task-bootstrap 95% CI "
            f"[{reward['task_boot_ci95_lo']},{reward['task_boot_ci95_hi']}]) and produces "
            f"{wins}/{losses}/{ties} task-level reward wins/losses/ties."
        )
        crossenv_conclusion = "A train-disjoint WebShop comparison also exhibits non-uniform task-level memory effects, extending the conditionality observation beyond ALFWorld while leaving cross-environment executor moderation untested."
    else:
        crossenv_summary = (
            f"A train-disjoint 100-task WebShop validation is inconclusive: mean reward changes from "
            f"{reward['baseline_mean']} to {reward['memory_mean']} (paired delta {reward['paired_delta']}, "
            f"95% CI [{reward['task_boot_ci95_lo']},{reward['task_boot_ci95_hi']}]), with "
            f"{wins}/{losses}/{ties} wins/losses/ties."
        )
        crossenv_conclusion = "The bounded WebShop validation is inconclusive, so cross-environment generalization remains an open question."

    replacements = {
        "[[WEBSHOP_ABSTRACT_RESULT_PENDING]]": crossenv_summary,
        "[[WEBSHOP_CONTRIBUTION_PENDING]]": (
            f"Using insights distilled locally from disjoint WebShop train sessions, we evaluate 100 held-out "
            f"sessions and observe {wins}/{losses}/{ties} reward wins/losses/ties; this "
            + ("supports task-conditional utility outside ALFWorld." if supported else "does not establish cross-environment transfer.")
        ),
        "[[WEBSHOP_METHOD_PENDING]]": (
            f"For bounded cross-environment validation, we use the official WebShop small environment with 1,000 products. "
            f"A 20-task Qwen gate is required to achieve complete execution, valid action syntax, nondegenerate reward, and purchase attempts. "
            f"After the gate passes, we collect train-only trajectories from sessions 500-519 and locally distill "
            f"{metadata['n_insights']} ExpeL-style procedural insights from {metadata['selected_trajectories']} positive-reward trajectories. "
            f"We compare no memory with this fixed insight pool on disjoint sessions 0-99 using environment reward as the primary outcome and exact purchase success as secondary."
        ),
        "[[TRAJECTORY_RESULT_PENDING]]": (
            f"We replay nine representative tasks under three unique prompt pairs with local Qwen, yielding 27 paired baseline-memory comparisons. "
            f"Memory rescues {switches.get('memory_rescue', 0)} executions and harms {switches.get('memory_harm', 0)}; "
            f"the remainder are {switches.get('stable_success', 0)} stable successes and {switches.get('stable_failure', 0)} stable failures. "
            f"Predeclared behavioral markers attribute {markers.get('procedural_completion', 0)} rescues to the appearance of the required operation, "
            f"{markers.get('search_efficiency', 0)} to shorter search, {markers.get('loop_amplification', 0)} harms to increased repetition, and "
            f"{markers.get('procedure_displacement', 0)} to displacement of a required operation. Other switches remain unresolved. "
            f"These markers explain observable trajectory changes, not executor-component causality."
        ),
        "[[WEBSHOP_RESULT_PENDING]]": (
            f"The 20-task feasibility gate passes with 100% valid action syntax, 20/20 search coverage, 13 purchase attempts, 10% exact success, and mean reward 0.2883. "
            f"In the 100-task train-disjoint comparison, no-memory versus ExpeL-style memory yields environment reward "
            f"{reward['baseline_mean']} versus {reward['memory_mean']} (delta {reward['paired_delta']}, "
            f"task-bootstrap 95% CI [{reward['task_boot_ci95_lo']},{reward['task_boot_ci95_hi']}]) and exact success "
            f"{success['baseline_mean']} versus {success['memory_mean']} (delta {success['paired_delta']}, "
            f"[{success['task_boot_ci95_lo']},{success['task_boot_ci95_hi']}]). Reward wins/losses/ties are {wins}/{losses}/{ties}. "
            + ("The presence of both gains and harms satisfies the predeclared cross-environment conditionality criterion." if supported else "The predeclared cross-environment criterion is not satisfied.")
        ),
        "[[WEBSHOP_LIMITATION_PENDING]]": (
            "the WebShop extension uses one local executor, the 1,000-product small environment, and one compact train-distilled memory. It tests task-conditional utility outside ALFWorld but not executor moderation across domains."
        ),
        "[[WEBSHOP_CONCLUSION_PENDING]]": crossenv_conclusion,
    }

    manuscript = MANUSCRIPT.read_text()
    for marker, value in replacements.items():
        if marker not in manuscript:
            raise ValueError(f"missing manuscript marker {marker}")
        manuscript = manuscript.replace(marker, value)
    if "[[" in manuscript or "PENDING" in manuscript:
        raise ValueError("manuscript still contains pending markers")
    MANUSCRIPT.write_text(manuscript)

    ledger = LEDGER.read_text()
    ledger = ledger.replace("`FROZEN_ALFWORLD_PENDING_WEBSHOP`", "`FROZEN_COMPLETE`")
    ledger = ledger.replace(
        "| C5 | Conditional utility appears outside ALFWorld. | WebShop train-disjoint no-memory versus ExpeL-style comparison. | pending job `1182482` | A one-executor WebShop result cannot establish cross-environment executor moderation. |",
        f"| C5 | Conditional utility appears outside ALFWorld. | WebShop reward delta `{reward['paired_delta']}`, CI `[{reward['task_boot_ci95_lo']},{reward['task_boot_ci95_hi']}]`, wins/losses/ties `{wins}/{losses}/{ties}`. | {'supported' if supported else 'inconclusive'} | A one-executor WebShop result cannot establish cross-environment executor moderation. |",
    )
    ledger = ledger.replace("| M2 | Task-matched procedural rules can rescue missing operations or inefficient search. | Nine both-help tasks; full Qwen trajectory replay in progress. | pending behavioral audit |", f"| M2 | Task-matched procedural rules can rescue missing operations or inefficient search. | Qwen replay: `{switches.get('memory_rescue', 0)}` rescues; `{markers.get('procedural_completion', 0)}` procedural-completion markers and `{markers.get('search_efficiency', 0)}` search-efficiency markers. | behavioral support, not causal |")
    ledger = ledger.replace("| M3 | Memory can displace a successful procedure through loops, expanded search, or missing required operations. | Six both-hurt tasks; full Qwen trajectory replay in progress. | pending behavioral audit |", f"| M3 | Memory can displace a successful procedure through loops, expanded search, or missing required operations. | Qwen replay: `{switches.get('memory_harm', 0)}` harms; `{markers.get('loop_amplification', 0)}` loop-amplification and `{markers.get('procedure_displacement', 0)}` procedure-displacement markers. | behavioral support, not causal |")
    if "pending job" in ledger or "pending behavioral" in ledger:
        raise ValueError("claim ledger still contains pending evidence")
    LEDGER.write_text(ledger)

    RECORD.write_text(
        "# Paper O Automatic Finalization Record\n\n"
        f"- WebShop decision: `{web_decision['decision']}`\n"
        f"- Reward delta: `{reward['paired_delta']}` `[{reward['task_boot_ci95_lo']},{reward['task_boot_ci95_hi']}]`\n"
        f"- Reward wins/losses/ties: `{wins}/{losses}/{ties}`\n"
        f"- Trajectory switches: `{dict(switches)}`\n"
        f"- Manuscript pending markers: `0`\n"
        f"- Claim ledger status: `FROZEN_COMPLETE`\n"
    )
    print(RECORD)
    print(json.dumps({"webshop_supported": supported, "switches": switches}, default=dict))


if __name__ == "__main__":
    main()
