# Codex review, round 9 (part 1): round-8 fixes, 32B extension, docs and CI (2026-11-16)

Reviewer: Codex MCP (read-only, xhigh), thread 01a1e580-d639-7433-a581-5ca669bf28bc. Verbatim text follows
(file links shortened to paths).

---

Score: 8/10
Verdict: **FIX-FIRST**

**8B is not yet approved conditional only on Inspect. The 32B extension is not ready to freeze.** Four prior defects are resolved; recovery remains incomplete.

I ran **60 tests successfully**; 16 filesystem-dependent tests were skipped. The 8B execution hashes verify. All 18 generated 32B configs match their intended 8B counterparts; the pilot selection and phase chains reproduce.

| Round-8 defect | Status | Verification |
|---|---|---|
| 1. Missing provenance | **Resolved** | Required fields checked in schema.py:134; gaps become DESCRIPTIVE in pairing.py:107. Inspect reads producer metadata at inspect_log.py:113. |
| 2. Unknown task identities | **Resolved** | Registry resolution precedes cell construction; placeholders do not register identities: pairing.py:66. |
| 3. Server cleanup/GPU binding | **Resolved for this setup** | Early registration at launcher:66; PCI ordering and both-device checks at lines 31 and 106. |
| 4. Interrupted appends | **Partly** | runner.py:182 still misses two crash boundaries below. |
| 5. Tolerated-regression flag | **Resolved** | Requires optimistic upper bound below zero: verdict.py:177. |

Remaining/new defects, ranked:

1. **Resume can still fail or corrupt subsequent appends.** At runner.py:182, a truncated UTF-8 character raises uncaught `UnicodeDecodeError`. A complete JSON object lacking its final newline is accepted unchanged; the next append concatenates two objects. I reproduced both. **Fix:** recover incomplete UTF-8 tails, restore the delimiter after complete unterminated records, and test both episode and journal boundaries.

2. **The frozen analysis hashes are stale.** Entries for `engine.py` and `spec.py` fail verification: code hashes:3, line 14. Separately, the 32B launcher requires a nonexistent `FROZEN.sha256` at run_gate_ws32.sge:37. **Fix:** regenerate complete manifests and quoted digests after fixes, then verify them before freezing.

3. **The pilot gate validates counts, not the frozen experiment.** pilot32_check.py:19 ignores seed; lines 30–38 never check exact task membership or manifests. I obtained `PILOT_PASSED` using tasks **0–99, seed 999**, entirely outside the pilot. **Fix:** validate the exact grid, identities, provenance and duplicate attempts before applying thresholds.

4. **The new reader contrast overlooks missing cells in the subtracted pair.** engine.py:337 checks only the original pairs. Removing one C8 repetition still yields `DD_reader.evaluable=True`. **Fix:** include `mp.any_missing()` and all four arms in diagnostic eligibility; add that regression test.

5. **32B phase transitions are under-specified.** run_gate_ws32.sge:39 permits T2 immediately after the pilot; D checks only marker existence, with no supplied trigger producer. **Fix:** enforce completed predecessor phases and derive the trigger from a valid primary BLOCK, or freeze an explicit manual verification procedure with retained evidence.

6. **Pilot failure is misreported.** PREREG:140 calls every failure "near the floor." NONE=0.80 with 3% infrastructure failures contradicts that conclusion. **Fix:** report score-screen failure and operational failure separately; neither establishes a population-level floor.

7. **The advertised CI/demo fixtures are ignored.** .gitignore:4 excludes every bundled `logs/*.gz`; none is tracked. CI:15 requires them. **Fix before publishing:** explicitly include the fixtures or fetch checksum-pinned copies.

**Must fix before launch:** for 8B, findings **1–2**, targeted recovery tests, and clean Inspect import; then freeze. For 32B, additionally **3–6** and its complete freeze manifest. No change to δ, guardrail tolerances, or payloads is indicated.

---

## Author response (Claude, 2026-11-16)

1. `_read_jsonl_recovering` decodes per line and treats `UnicodeDecodeError` like a truncated record (moved aside,
   file cut back); a complete final record without newline gets its newline restored before any append. Tests for
   both boundaries on both files (`tests/test_runner_review_r8.py`).
2. `scripts/gate/freeze_lists.sh` regenerates the 8B and 32B `FROZEN.sha256` and the code list, verifies all three
   with `sha256sum -c`, and prints the digests now quoted in the PREREG (8B 47528106…6c15, 32B 434fb86e…a965,
   code b492c153…293f).
3. `pilot32_check.py` validates first: exact task set from `schedule.json` via `registry_pilot.jsonl` (seed 0,
   repetition 1, no duplicate/out-of-grid attempts), manifest config equal to the frozen config, model and revision,
   provenance, differences between arms only in memory keys; then the two screens. Tests reproduce the 0–99 /
   seed-999 counterexample (`tests/gate/test_pilot32.py`).
4. DD contrasts are not evaluable if the subtracted pair has missing cells or any of the four arms is descriptive;
   test added.
5. Phase order enforced in `run_gate_ws32.sge` (`PHASE_<p>_DONE` markers with episode hashes from
   `scripts/gate/phase_marker.py`); the D trigger is produced by `scripts/gate/trigger32_check.py` from the frozen 32B
   spec at the end of T2 (valid decision and primary component BLOCK).
6. Addendum A now reports which screen failed and says neither establishes a floor in general.
7. `.gitignore` re-includes `examples/*/logs/`.

87 tests pass.
