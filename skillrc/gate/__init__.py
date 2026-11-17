"""skillrc-gate: release verdicts for agent changes, computed from episode logs.

The gate reads the logs of two or more agent versions that ran on the same tasks and returns, per declared
release target, one of PASS / BLOCK / INCONCLUSIVE / INVALID / DESCRIPTIVE with paired intervals, cost and
latency guardrails, and links from the verdict to the trajectories behind it.

Contract: engineering/GATE_DESIGN_v3.md as amended by GATE_DESIGN_v4.md.
"""

__version__ = "0.1.0"
SCHEMA_VERSION = 1
