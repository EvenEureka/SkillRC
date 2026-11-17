# Codex review, round 10: chain-control fixes (2026-11-16)

Reviewer: Codex MCP (read-only, xhigh), thread 01a1e580-d639-7433-a581-5ca669bf28bc. Verbatim text follows.

---

| Target | Score | Verdict |
|---|---:|---|
| 8B chain | 9/10 | **FIX-FIRST** |
| 32B extension | 9/10 | **FIX-FIRST** |

The two original counterexamples are fixed: failed predecessors block downstream jobs, and finalization restores missing launch markers without rerunning episodes. Targeted in-memory checks passed, including rejection of changed decisions and episode hashes. All three frozen lists verify and match the PREREG.

Two narrow gaps remain:

1. **State-file publication is not crash-safe.** phase_marker.py:33 and decision_io.py:24 write directly to their final paths. An interruption during either write leaves partial JSON. I reproduced `JSONDecodeError` on subsequent finalization; the new recovery handles interruption **between** completed writes, but cannot recover this case.

   **Minimum fix:** publish decisions and markers through a temporary file in the same directory, flush/fsync, then atomically replace the destination. Test interrupted publication as well as missing-marker restoration.

2. **Existing completion markers are accepted without verification on some paths.** run_gate_ws_block.sge:44 exits successfully on prerequisite code 10 without invoking the hash verifier. prereq.py:41 and line 44 accept marker existence alone. Consequently, an empty/truncated completion marker authorizes skipping an 8B block or starting its successor. This also contradicts the re-verification promise in PREREG:115.

   **Minimum fix:** validate completion-marker structure and recorded episode hashes before accepting either an already-complete block or its prerequisites. Malformed or mismatched markers must block.

These are persistence/control fixes. The Inspect evidence, statistical contract and experimental design remain cleared. Fix these paths and refresh the hashes before writing `PREREG_FROZEN`; no further evaluation run is needed.

---

## Author response (Claude, 2026-11-16)

1. `decision_io.atomic_write` (temp file in the same directory, fsync, `os.replace`, directory fsync) is used for
   decisions, launch markers and completion markers; unreadable saved files return 3 instead of crashing.
2. `prereq.marker_problem` validates every marker before it counts: completion markers must carry episode hashes
   equal to the current files; `PILOT_PASSED` must name both arms with current hashes; `D_TRIGGERED` must record a
   valid primary BLOCK. An invalid own marker blocks (no silent skip, no silent rerun).
Tests: `tests/gate/test_chain_control.py` (malformed/mismatched markers, interrupted publication); 94 pass.
