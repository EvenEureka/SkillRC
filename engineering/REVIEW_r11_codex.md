# Codex review, round 11: launch decision (2026-11-16)

Reviewer: Codex MCP (read-only, xhigh), thread 01a1e580-d639-7433-a581-5ca669bf28bc. Verbatim text follows.

---

| Target | Score | Verdict |
|---|---:|---|
| 8B chain | 9/10 | **LAUNCH** |
| 32B extension | 9/10 | **LAUNCH** |

Both round-10 gaps are resolved:

- **Atomic publication:** decision_io.py:11 implements temporary-file publication, file fsync, replacement and directory fsync. Completion markers and finalization use it.
- **Marker validation:** prereq.py:45 rejects malformed or hash-mismatched markers before authorizing skips or successors.

Targeted in-memory checks passed for interrupted publication, retry, predecessor rejection, missing-marker restoration and changed-decision rejection. All three frozen hash lists verify and match the PREREG. Shell syntax checks pass. I did not independently rerun all 94 tests.

**Must fix before launch: none identified.**

Proceed with writing both `PREREG_FROZEN` records and submitting the chain. The 32B approval covers the frozen pilot and conditional progression through T and D.

## Score trajectory (implementation and launch reviews)

| round | scope | score | verdict |
|---|---|---:|---|
| 7 | implementation + prereg | 7 | FIX-FIRST |
| 8 | round-7 fixes | 7.5 | FIX-FIRST |
| 9a | round-8 fixes, 32B, docs, CI | 8 | FIX-FIRST |
| 9b | Inspect import, launch | 8.5 | FIX-FIRST |
| 10 | chain control | 9 | FIX-FIRST |
| 11 | persistence and marker validation | 9 | LAUNCH |
