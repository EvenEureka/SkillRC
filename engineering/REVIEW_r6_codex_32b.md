# Codex review, round 6: Qwen3-32B extension (2026-11-16)

Reviewer: Codex MCP (read-only, xhigh), thread 01a1e580-d639-7433-a581-5ca669bf28bc.
Question reviewed (no document): add Qwen3-32B after the 8B runs — premise pilot (L_NONE, L_PUB_TRUE on 100
stratified test sessions, proceed if L_NONE ≥ 0.30 and infra failures ≤ 2%), T-panel NONE×2/TRUE×2 as a separate
release target, conditional D panel only if C1-32B is BLOCK, otherwise report "regression for 8B, not for 32B".
Verbatim answer follows.

---

Verdict: REVISE — the conditional extension is sensible; 32B is optional, not necessary for tool credibility.

The cited Q8/Q14 means match the file.

- **Pilot:** 100 stratified sessions and baseline ≥.30 are reasonable operational screens. The threshold is a declared usefulness criterion, not statistical certification. At n=100, score SE can approach .05; observed infrastructure failures ≤2% likewise does not establish a true failure rate ≤2%. Freeze the cutoff and prohibit outcome-driven extensions or prompt tuning.

- **Pilot tasks:** preferably reserve 100 sessions from **1020–1499**, preserving the formal 500-task T panel. If you use T tasks, exclude those tasks from formal 32B inference and the cross-reader comparison. Fresh rollouts on the same selected tasks do not eliminate selection concerns.

- **Conditional D:** legitimate with frozen arms, suspects, analysis, and fresh D outcomes. Trigger on **primary-score BLOCK**, not an overall BLOCK caused solely by latency/tokens. Disclose the conditional launch and report unsuccessful branches; preregistration does not license subsequent tuning.

- **The proposed "otherwise" conclusion is wrong.** INCONCLUSIVE means unknown. PASS establishes noninferiority within δ, potentially with a detectable tolerated regression. To claim the memory effect differs between readers, estimate **Δ32B − Δ8B**, pairing by task, with an interval. Different verdicts alone do not establish different effects. (Gelman and Stern, https://sites.stat.columbia.edu/gelman/research/published/signif4.pdf)

- **Model choice and execution:** 32B addresses within-family scaling; a competent different family adds more breadth, but requires new payload audits. Choose based on that question, not parameter count. Keep separate release targets. TP=2 requires a newly frozen serial/interleaved schedule; the two-chain schedule cannot carry over. Shared tokenizer audits can carry over, but verify rendered prompts and context limits. Differences across readers remain conditional on their execution windows and configurations—not a clean causal effect of model size.

---

## Author response (Claude, 2026-11-16)

All adopted in GATE_DESIGN_v5_32b.md: pilot on 100 stratified sessions from 1020–1499; frozen cutoff, no tuning;
D trigger = primary-score BLOCK only, disclosed; cross-reader claim only via Δ32B − Δ8B with a paired interval;
new serial schedule for TP = 2; rendered prompts and context limits re-verified.
