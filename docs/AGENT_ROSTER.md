# The agent roster and the workflow

Last updated: 2026-08-25

What each agent contributes, how the graph is sequenced, and the design decisions
behind the specialist nodes. This document explains the rationale and intended
behaviour; the implementation and its tests are authoritative. If this file and the
repo disagree, trust the repo.

## Workflow

1. **Parallel data gathering.** Seven analyst streams, plus an optional forensic
   auditor, dispatch simultaneously. Market, sentiment and news feed the sync check
   directly. Junior fundamentals, foreign-language and legal counsel feed a
   fundamentals barrier. The value-trap detector runs alongside.
2. **Fundamentals barrier.** Waits for raw data, native-language sources, and legal
   risk before the senior analyst runs.
3. **Senior fundamentals.** Cross-validates cash flow against filing data, flags
   segment deterioration and ownership concentration, and produces the scored data
   block the hard gates read.
4. **Red-flag pre-screening.** Deterministic. On a catastrophic finding it skips the
   debate entirely and routes to the portfolio manager — a fast fail that saves most
   of a doomed candidate's token cost and prevents the debate rationalising a fatal
   flaw.
5. **Adversarial debate.** Bull and bear argue in parallel rounds. Quick mode runs one
   round, full mode two.
6. **Research synthesis**, then a parallel valuation calculator and external
   consultant, converging on the trader.
7. **Risk assessment** from three perspectives, converging on the portfolio manager.
8. **Executive decision**, then charts generated from the final verdict.

## Why the debate is adversarial and same-vendor

Bull and bear are *role*-adversarial, not vendor-adversarial. They share memory
collections, one sync barrier, and one research manager, and every downstream gate is
calibrated against that debate. Splitting them across vendors would confound "the bear
case was stronger" with "that vendor writes more forceful prose". Vendor diversity
belongs at **verification** boundaries — the review plane — not inside a debate.

## The specialists

**Legal counsel.** Detects passive-foreign-investment-company and variable-interest-
entity exposure for a US investor, emitting structured JSON for deterministic parsing.
Searches conditionally on a high-risk sector or jurisdiction profile. These are
**warnings with penalties, not rejections**: a tax classification is a reporting
burden, not a viability problem, and a company with one can still be a good
investment. Its fallback stub leaves statuses null **on purpose** — both callers wrap
it in a failure result, and the flag detector keys on that to emit an unavailability
flag that blocks a BUY at zero penalty. Filling in "uncertain" tokens there would
manufacture findings from a provider outage.

Legal Counsel's code-owned preflight, bounded tool loop, and forced JSON synthesis are
one self-contained transaction. Other tool-using analyst seats use the graph's shared
tool node, but both paths obey the same provider-neutral transcript rule: an assistant
tool call and every matching result are retained as one exchange and validated before
another model request. Provider tolerance is never used to repair malformed history.

**Value-trap detector.** Identifies businesses that are cheap and will stay cheap:
entrenched ownership, capital hoarding, no catalyst. Its distinctive move is searching
jurisdiction-specific governance terminology that English-only searches miss —
cross-shareholdings, parent-child listings, chaebol structures, mid-term plans. Its
self-reported score is cross-checked against the authoritative governance card, and a
contradiction marks the score untrusted rather than trusting the model's number.

**Foreign-language analyst.** Finds segment breakdowns, parent-subsidiary ownership,
and filing-level cash flow that English aggregators miss — data that is genuinely
unavailable rather than merely inconvenient. Reaches the senior analyst through
context injection rather than the message list, to avoid polluting parallel agents.
Its search menu is prioritized coverage, not a checklist: a shared code-owned research
ledger caps rounds, fan-out, tools and purposes; suppresses equivalent calls; opens
circuits for repeated failures; and forces a tool-free final synthesis when research
closes. Prompt wording may choose what evidence matters but cannot expand those caps.

**Forensic auditor.** An independent accounting check on a different vendor, working
from primary documents through multilingual search, testing for earnings quality,
receivable inflation, solvency, inventory, goodwill, payables stretching, and
suspicious cash yields. It plans its searches up front and issues them as one parallel
batch; a deterministic metrics pre-call replaces its own aggregator fetching.

**External consultant.** Cross-validates the analysis on the review plane. A completed
review with a *minority* of failed verification calls reaches the PM tagged as partial
rather than being discarded — losing the whole counterweight to one failed call flipped
a verdict once, which is what the partial policy exists to prevent.

The consultant and auditor gate on the **same** review plane, and neither is
subordinate to the other: the auditor has independent consumers, so switching off the
consultant must not disable it. Full mode preserves that independent forensic path.
Quick mode deliberately disables the auditor because its constrained screening tools
do not provide diligence-grade coverage; promoted candidates restore it in full mode.

## Red-flag screening

Deterministic and code-driven because exact thresholds matter, sector branches prevent
false rejections, and number parsing must not hallucinate.

Thresholds branch across three sector profiles: financials disable the debt gates,
capital-intensive sectors get much wider ones, everything else takes the standard
profile. Auto-rejection is reserved for extreme leverage, an earnings-quality
inversion, refinancing risk, and a value-destroying distribution that is not already
improving. Everything else is a warning carrying a risk penalty.

Score-consistency checking treats the rubric arithmetic as auditable: provable
arithmetic errors are corrected, while template or denominator violations are marked
*suspect* and never repaired by inference. A suspect score makes its gate
**indeterminate in both directions** — never an automatic rejection, and never counted
as a pass supporting a BUY.

## Prompts

Versioned JSON with a system message and metadata. They encode algorithms in natural
language — explicit thresholds, explicit failure conditions, explicit instructions to
document which tool failed before reporting a value as unavailable. That combination
of deterministic business rules with model reasoning is the core pattern, and it is
why prompt and parser must be kept in agreement mechanically. See
`docs/PROMPT_CONTRACTS.md`.

Provider transport completion is not the same as contract completion. A transient
partial response without a terminal reason may receive the invocation runtime's
ordinary provider retry; an explicit output-cap stop does not repeat the same request.
Once a provider returns successfully, the owning node separately validates required blocks,
field vocabularies, and end markers. Eligible full-mode analysts may receive one
text-only structural regeneration; quick mode reserves that recovery for Senior
Fundamentals and Portfolio Manager. The recovery binding stays in the base provider,
uses a reasoning intent, inherits the originating seat's output budget, and cannot
restart tool gathering.

## Quick mode is a screener

Cheaper models, one debate round, and judgement-layer scores that flap near the gates.
A quick-mode BUY is qualified as a candidate for full analysis rather than investable
output, and reconciliation treats it as a review. The verdict token itself is never
rewritten — downstream parsers all still read it — so the qualification rides the
persisted text and a derived run-summary flag. The forensic auditor does not run in
quick mode; its full-mode review is part of the diligence restored for promoted
candidates.
