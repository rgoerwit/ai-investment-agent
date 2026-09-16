# Provenance: which layer owns which decision

Last updated: 2026-09-15

This file records how evidence moves through the pipeline and why the boundaries sit
where they do. It explains the rationale and intended behaviour; the implementation
and its tests are authoritative. If this file and the repo disagree, trust the repo.

## The governing idea

> The deterministic layer owns decisions. Agent prose is a projection of those
> decisions, never their source.

Every defect in this area has the same shape: a value computed correctly in one layer
is re-derived, re-parsed, or re-minted in another, and the second copy wins. The
system does not crash when that happens. It produces a confident answer built on a
number nobody stands behind.

## The authority matrix

Which source may each deterministic consumer read?

| Consumer | Authoritative source |
|---|---|
| Pre-screening red-flag engine | typed decision inputs, snapshot-backed |
| Analysis outcome | merged pre-screening result and persisted red-flag ledger |
| Verdict floor | typed decision inputs, snapshot-backed |
| Review-candidate tagging | the PM decision text — deliberately |
| Portfolio flag index | the persisted red-flag ledger |
| Broker position identity | validated symbol/exchange mapping or conid cache |
| Report-stage flags | the persisted red-flag ledger, passthrough |
| Publication gate | the decoded snapshot and trace contract |

The cross-layer test suite asserts *which source* each consumer reads when two layers
disagree. Per-consumer unit tests structurally cannot catch this, because each layer
is individually correct.

`analysis_outcome` is a persisted projection, not writable graph authority. Runtime
validity derives it from the merged gate primitives; persistence stores that same
derived value only after validity has been built. A contradictory legacy projection
fails publication instead of overriding the merged rejection.

**A deterministic check may consume projected text only when that text is guaranteed
to be a valid canonical projection.** Anything that changes a gate, a score, BUY
eligibility, or that removes another flag must consume canonical evidence directly.

## Broker identifiers are not market tickers

IBKR may place an internal contract token such as `IBCID<conid>` in a field that
normally contains a market symbol. That token is an identity-recovery instruction,
not a ticker and never a yfinance input. The portfolio boundary classifies broker
symbols before constructing a `Ticker`: ordinary securities are converted normally,
corporate-action receivables are excluded as non-securities, and contract identifiers
must recover a validated yfinance identity from the conid map or live contract metadata.

If recovery fails, the holding remains in accounting and known-currency exposure as
an unresolved identity, while research, refresh scheduling, and order construction
receive no ticker at all. Sector and exchange concentration continue to describe the
identified book; reports expose the missing identity coverage separately rather than
inventing a sector or geography. Portfolio-increasing BUY and ADD recommendations are
withheld until inventory identity is complete, while reductions on resolved holdings
remain available. Cached conid mappings remain usable during a transient live API
failure, but placeholder observations cannot enter that cache. Reports expose
unresolved holdings explicitly so an identity outage cannot silently create either a
research vacuum or a false impression that the holding disappeared.

## Decision score versus advisory score

The score projected into the hard gates is computed from **dependency-backed awards
only**. A positive award on a criterion with no configured evidence producer is
excluded from the decision numerator while the full rubric stays the denominator —
conservative by construction, so it can only lower a score, never raise one. The
model's raw number survives as an advisory figure.

A fully-backed scorecard is byte-identical to what it was before this split. That
parity is the point: the mechanism only bites where evidence is missing.

The neighbouring veto is different and must not be confused with it: when a
*configured* dependency fails to resolve on a given run, the criterion goes to `N/A`
rather than scoring zero. An earlier blunt version exempted unbacked awards entirely,
which let an unevidenced point lift a score across a gate.

## The second question: what happens when the source is absent?

The matrix says which source may be read. It does not say how a consumer resolves that
source coming back empty, and that was left to per-site judgement for a long time.

> **Missing evidence about *merit* resolves NEUTRAL. Missing evidence about *hazard*
> or *authority* resolves CLOSED.**

Scoring merit punitively invents a failure; screening hazard permissively invents an
all-clear. The use is a property of the **consumer**, not of the datum: the same
unknown baseline is a merit input to a growth rubric and would be an authority input
to a sell gate. The resolver is total over the enum, so a new use must declare a
direction rather than inherit a default.

**The direction is not uniform, which is exactly why it must be declared.** Both
readings are load-bearing here: a marginal BUY is withheld when quality flags are
unavailable, an unknown run mode carries no sell authority, and analysis matching
fails closed on a missing currency — while an unreliable score makes a gate
indeterminate *in both directions*, an uncomparable cash-flow period carries zero
penalty, and an unresolved lineage leaves a criterion at `N/A`. Conflating the two
classes produced defects with **opposite signs**, which is why "just be conservative"
is not the rule.

### The defect that produced the rule

A predicate returned a boolean, and "could not classify" sat in the same set as four
*positive diagnoses* of distortion. The single award-writing site then wrote `0` — and
**`0` stays in the rubric denominator while `N/A` leaves it**, so absent evidence
scored as *failed* evidence on a criterion feeding a hard gate.

Measured on a twelve-ticker basket it fired on six, and dominated every large score
drop. Corpus replay over 1,631 artifacts found 118 affected, five of them crossing the
gate. On a screener that is a false negative, and a false negative is never revisited.

The seam: separate the *diagnosed* statuses from the *unclassified* ones, and return a
three-valued disposition — keep, refuted, unresolved — rather than a boolean, because
a boolean is what let the caller lose the distinction. One helper owns the choice
between `0` and `N/A`.

Note the manufacturing chain that hid it: a normalizer folded null tokens to
`UNKNOWN`, so an *omitted* field arrived indistinguishable from a considered "I could
not tell", and both were then scored as distortion. Absence became failure in three
hops, none of them a declared choice.

## Dead tokens

A branch that is never taken fails nothing, so a token no producer emits is invisible
until someone measures it. Three instances are on record: a flag string nothing
assigned, an output key advertised in a prompt that no parser read, and a status value
gating a demotion that no code ever produced.

The last one is the clearest. A field could be demoted from "not disclosed" to
"unresolved" unless some evidence status equalled a particular value — a value **no
producer emits, in roughly 4,700 artifacts**. The distinction had been designed in
deliberately and was simply unreachable, and the flag downstream of it went from 1.2%
incidence to 38.5% with a 0% BUY rate wherever it fired. At 1.2% a blocking flag is an
informative signal; near 40% it is a constant that stops the screener finding
anything.

The guard asserts that every contract token is either code-assigned or
prompt-advertised. A naive "assigned somewhere in the source" scan is wrong, because
several legitimate tokens are authored by the model and only validated by code.
Genuinely reserved tokens are allowed but must never gate a decision while unemitted.

## Fail-closed decoding

The three gate-critical payloads have one typed definition and a versioned wire codec.
Decoding raises rather than defaulting on: a future or non-integer schema version, a
present-but-type-corrupt field, a non-finite float, a non-bool where a bool is
required, a wrong-typed collection, or a list holding a non-string element. A
**missing** version is legacy and loads.

Two distinctions worth keeping straight. The revision counter and the schema version
are different fields; never reuse the first for compatibility. And decoding is
type-only on purpose — payload semantics belong to their producers at construction,
because re-deriving a business rule inside a wire codec couples them and they drift.

At the boundary, a decode failure maps to a distinct status and the artifact is
non-publishable.

## Verdict interventions are durable data

The saved `decision_policy` record captures the model's original verdict, the final
verdict, each deterministic verdict-changing adjustment, verdict-preserving
qualifications, the growth-gate assessment, and BUY-blocking flags. Reports consume
that record before showing model prose. When a verdict changed, the policy notice is
canonical, pre-policy rationale is omitted from the memo, and the original narrative
appears only in the explicitly labeled audit appendix.

This avoids a second natural-language rewrite. Rewriting every sentence would add a
new model-owned interpretation layer; retaining the original narrative without a
structured intervention record allowed stale HOLD/BUY language to contradict the
final header.

## Gate inputs are code-owned where the data permits

The growth rubric's `ROA_ROE_IMPROVING` point is calculated from adjacent annual
statement periods. The model still supplies its complete rubric projection for
coherence checking, but code replaces that one award with `1`, `0`, or `N/A` from the
canonical ROA/ROE YoY claims. A disagreement is retained as a derivation conflict; it
cannot change the decision score.

The growth data-vacuum exception is a field-coverage rule, not score arithmetic. It is
available only when a valid snapshot shows all four current growth observations absent:
revenue and earnings at both TTM and MRQ horizons. Missing expansion, margin, or other
rubric criteria cannot manufacture the exception.

Liquidity follows the same authority pattern. The market tool emits a typed assessment
using the thresholds in `src/thesis_constants.py`; the market node carries it through
graph state and maps a measured hard fail to the existing `REJECT`/`AUTO_REJECT`
contract; snapshot refresh records its status and USD turnover. Gate aggregation is
reject-dominant, and the sync barrier waits on explicit financial-validator completion
rather than treating a gate outcome as a completion sentinel. Retrieval or
currency-resolution failure is recorded as uncertainty and does not become issuer
risk. The verdict-policy boundary consumes the generic `AUTO_REJECT` contract after
the Portfolio Manager runs, so a model-authored BUY or HOLD cannot weaken a
deterministic rejection; no liquidity threshold is duplicated in the verdict layer.

## Tool outcomes have declared scope

`run_summary.tool_failures` is a deprecated compatibility field. It counts manual
failure counters plus error-like `ToolMessage` objects still retained in capped graph
state, so it is neither a run-wide execution-failure count nor a budget-block count.

New artifacts carry `run_summary.tool_outcomes`, which keeps these scopes separate:

- shared research-budget ledgers distinguish policy blocks, ordinary insufficient
  results, evidence-acquisition failures, and execution errors;
- the run-scoped evidence recorder reports the executed calls it observed; and
- manual consultant-style counters remain separately identified.

Unique tool-name lists are not event counters. A tool blocked twice appears once in
`blocked_tools` and twice in `blocked_reasons`.

## External evidence promotion is observable

The source-required promotion path is deliberately strict: discovery result → selected
document → inspected evidence → structured source-required field → bound canonical
claim → decision eligibility → optional decision-fact reference. Search-result URLs
cannot skip the document-inspection step.

`run_summary.evidence_promotion` records the counts at the final four stages, and the
report's Decision Evidence section renders only decision-trace claims. A link appears
only when a decision-eligible claim is bound to an inspected evidence record. Zero
links are stated explicitly rather than hidden behind a general bibliography. Quick
screens also record that external-document extraction was disabled by design, so their
expected zero-promotion result is not misclassified as a full-mode regression.

## What was deliberately not done

- **Promoting the loose source-family classifier to invalidating.** It is
  intentionally loose, flagging even negated mentions for observability, so a blunt
  promotion false-positives on benign prose.
- **Re-typing the qualitative fields as canonical claims.** Auditing the red-flag
  engine's inputs shows most are the senior analyst's *judgements*, not deterministic
  facts. They legitimately originate in that analyst's output, and the deterministic
  layer consuming them is the design working — a judgement lives in exactly one place,
  so it cannot diverge. Promoting them would mislabel assessment as evidence.
- **A period-mismatch suppression path** that could only ever *remove* a risk flag,
  reduced to bare floats so none of its guards ran, and produced its outcome in zero
  of 4,621 artifacts. Retiring it was behaviour-preserving in production.

## Method

Every priority in an earlier version of this analysis was checkable against the saved
artifacts in under a minute, and none had been. Before rating a provenance defect,
measure how often the path actually fires against saved run artifacts before rating it.
