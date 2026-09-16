---
paths:
  - "src/thesis_constants.py"
  - "prompts/portfolio_manager.json"
  - "src/validators/**"
---

# The investment thesis

A GARP value-to-growth strategy for undervalued non-US equities. **Every threshold is
canonical in `src/thesis_constants.py`.** Prose — here, in prompts, in reports — may
reference a threshold but must never restate the number; L0 parity in
`make test-prompts` enforces that, and a restated number is how the two drift.

## Hard requirements

- Financial health score ≥ 50%
- Growth score ≥ 50%
- Liquidity ≥ $100k daily USD turnover is a hard floor; ≥ $250k is a full pass.
  Between the two is MARGINAL, capped at a 3% position.
- Analyst coverage < 15 — the "undiscovered" criterion.

## Soft valuation targets

P/E ≤ 18, PEG ≤ 1.2, P/B ≤ 1.4, US revenue exposure 25–35%. These influence risk
scoring rather than gating.

## Two-layer authority — the rule that governs everything downstream

**Analysis is initiation-only research.** The analyzer is portfolio-blind: it does not
know holdings, cost basis, or tax position, and it can never order a sale. The PM
verdict vocabulary is `BUY / HOLD / DO_NOT_INITIATE` and contains no SELL.

**Only reconciliation may sell,** because only it has holdings, identity, history and
tax context — and even there only through a positive whitelist of exit conditions.

**Price movement is never exit evidence.** A drawdown is a review trigger. A sale
requires fundamental failure evidence, confirmed across two independent full-mode
analyses spaced apart, or a mandatory-exit condition. The operator is a long-term
retail investor: selling has capital-gains consequences, so the system must not
generate churn.

Consequences that follow, and that a change here must preserve: profit-taking and
drift-trimming are advisory, not orders; a score-intact rejection is the screen
declining to re-enter its own winner, not exit evidence; and legacy artifacts must not
be able to recover authority the current rules deny them.

## Changing a threshold

In order: the constant, then `prompts/portfolio_manager.json`, then the validator,
then the report generator, then `make test-prompts` for L0 parity. Bump the prompt
version and its `changes` line in the same edit.

Full history: `docs/AGENT_ROSTER.md`.
