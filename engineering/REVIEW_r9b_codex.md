# Codex review, round 9 (part 2): Inspect import and launch decision (2026-11-16)

Reviewer: Codex MCP (read-only, xhigh), thread 01a1e580-d639-7433-a581-5ca669bf28bc. Verbatim text follows
(file links shortened to paths).

---

| Launch target | Score | Verdict |
|---|---:|---|
| 8B chain through `submit_all.sh` | 8.5/10 | **FIX-FIRST** |
| 32B extension | 8.5/10 | **FIX-FIRST** |

**Inspect clears the integration requirement. Two execution-control defects remain before the proposed unattended submission.**

The Inspect evidence checks out. Native Inspect parsing reproduced the saved report exactly, including coverage, fingerprints, behaviour and replays; independent extraction from the archives reproduced the statistics.

- Each arm contains **96 identical task identities × epochs 1–2**, with 192 scored cells and no infrastructure failures or retries.
- Utility maps correctly: **77/192 BASE**, **82/192 TIPS**. Prompt-token totals are **4,548,678 / 5,511,798**. Latency correctly uses sample wall time; p95 is **157.196 / 134.331 seconds**.
- Fingerprint differences are exactly the two tips arguments, tips hash and system-message hash. Raw task inputs, initial environments, scorer and execution settings match.
- **INCONCLUSIVE** is correct: primary **+0.02604 [−0.046875, +0.098958]**, primary component PASS; both guardrails INCONCLUSIVE. MDE is **0.1085**, approximately 0.11.

**62 tests passed here; 22 filesystem-dependent tests were skipped.**

Part-1 fixes:

| Finding | Status and evidence |
|---|---|
| 1. Interrupted appends | **Resolved** — runner.py:185. Both boundary cases also passed in-memory checks. |
| 2. Frozen hashes | **Resolved** — all three lists verify and their digests match the PREREG; freeze_lists.sh:23. |
| 3. Pilot validation | **Resolved** — frozen config, provenance and exact grid checked at pilot32_check.py:53. |
| 4. DD missingness | **Resolved** — engine.py:322. |
| 5. Phase transitions | **Partly** — ordering and primary-BLOCK trigger implemented; interrupted finalization remains broken below. |
| 6. Pilot interpretation | **Resolved** — separate screen failures at PREREG:141. |
| 7. Bundled fixtures | **Resolved** — exclusions reversed at .gitignore:5. |

Remaining defects:

1. **The automatic chain advances after failed 8B jobs.** submit_all.sh:17 uses `-hold_jid`, which waits for completion, not success—the installed SGE manual (qsub.1:1394) confirms this. B1 exiting 1 still releases B2; B2 failing still releases the 32B pilot, whose prerequisite branch is empty at run_gate_ws32.sge:42. **Minimum fix:** write completion markers after successful 8B blocks; require B1's marker before B2 and both markers before the pilot.

2. **32B finalization is not resumable.** run_gate_ws32.sge:167 writes DONE before running the pilot/trigger checker, while line 49 rejects completed phases. A kill between those operations strands the chain. There is another gap between writing the decision and its launch marker: pilot32_check.py:86, trigger32_check.py:34; both checkers reject existing decisions. **Minimum fix:** make finalization idempotent, reconstruct missing markers from validated saved decisions, and permit finalization-only resume without rerunning episodes.

Fix these, test failed-predecessor and interrupted-finalization cases, refresh hashes, then freeze. **No further Inspect run or statistical redesign is needed.**

---

## Author response (Claude, 2026-11-16)

1. `scripts/gate/prereq.py` holds every prerequisite: block 2 needs `BLOCK_1_DONE`; the 32B pilot needs both 8B
   markers; T1 `PILOT_PASSED`; T2 `PHASE_T1_DONE`; D1 `PHASE_T2_DONE` + `D_TRIGGERED`; D2 also `PHASE_D1_DONE`.
   Both job scripts call it; the 8B block writes `BLOCK_<n>_DONE` (episode hashes) on success.
2. Finalization is idempotent: `phase_marker.py` verifies an existing marker against the episode hashes;
   `decision_io.finalize` recomputes the pilot/trigger decision, compares it with a saved one (exit 3 if different)
   and restores a missing launch marker; a job whose runs are complete (exit 10 from prereq) only finalizes.
Tests: `tests/gate/test_chain_control.py`, `test_pilot32.py::test_pilot_finalization_is_idempotent`; 92 pass.
Lists refreshed (8B cfd369b7…15b8, 57 files; 32B 8c6b5016…1810, 63 files; code d564bf6d…9b71).
