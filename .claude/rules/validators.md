---
paths:
  - "src/validators/**"
  - "src/decision_inputs.py"
  - "src/evidence_disposition.py"
---

# Deterministic screening

## Why this layer is code, not a prompt

Exact thresholds, sector-specific branches, and arithmetic on parsed numbers. A model
asked for "very high leverage" cannot be held to a number, and short-circuiting a
doomed candidate before the debate saves most of its token cost.

## Sector-aware, three profiles

Leverage and coverage thresholds branch on the GICS sector: financials disable the
debt gates entirely, capital-intensive sectors get much wider ones, everything else
takes the standard profile. A flat threshold rejects normal capital structures — that
is what the branching exists to prevent.

Auto-reject is reserved for the genuinely fatal: extreme leverage, an earnings-quality
inversion, refinancing risk, and a value-destroying distribution that is not already
improving. Everything else is a **warning with a risk penalty**, because governance
and tax-classification problems trap value rather than killing companies.

## Absent evidence has a direction, and it is not uniform

**Missing evidence about *merit* resolves NEUTRAL; missing evidence about *hazard* or
*authority* resolves CLOSED.** Scoring merit punitively invents a failure; screening
hazard permissively invents an all-clear.

The use is a property of the **consumer**, not of the datum: the same unknown is a
merit input to a growth rubric and an authority input to a sell gate. Declare the
direction through `EvidenceUse` rather than assuming "be conservative" — the two
classes need opposite treatment, and conflating them produced defects with opposite
signs.

Concretely, in a scored rubric: **`N/A` leaves the denominator, `0` stays in it.** An
unclassified criterion must score strictly higher than a diagnosed failure. Writing
`0` for "could not tell" turns absence into failure on a gate-critical score.

## Read the authoritative source

Deterministic checks consume the snapshot-backed typed inputs, not re-parsed prose. A
check that changes a gate, a score, BUY eligibility, or that *removes* another flag
must consume canonical evidence directly. A rejected or undecodable contract makes the
score unreliable in **both** directions — never a pass, never a fail.

The persisted flag ledger is authoritative for anything downstream; re-deriving flags
from a report silently drops the ones the run actually recorded.

## Before adding a token

Grep for a producer. A token no code assigns and no prompt advertises is dead, and a
dead token gating a decision fails silently forever — this repo has lost real
behaviour to that three times. Reserved-but-unemitted tokens are allowed, but must
never gate anything while unemitted.

Full history: `docs/PROVENANCE.md`.
