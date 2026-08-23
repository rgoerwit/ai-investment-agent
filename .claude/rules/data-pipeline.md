---
paths:
  - "src/data/**"
  - "src/fx_normalization.py"
  - "src/liquidity_calculation_tool.py"
---

# Data ingestion and merge

## Sources are fetched in parallel and merged by quality

Several vendors are queried simultaneously and merged per metric, with a web-search
fallback for critical gaps. Add a source by registering a builder, not by threading a
special case through the merge.

Two properties must survive any change here: a source that returns nothing leaves the
merged payload **byte-identical**, and no source is permanently disabled after a
failure. Every run is a fresh process that re-probes, so an entitlement or outage that
later clears recovers on its own. **Do not add a persistent "this source failed, stop
trying" latch** — that breaks auto-recovery.

Divergences above the tolerance are recorded as source conflicts and surfaced to the
downstream reviewers. Recording a conflict is the mechanism; silently preferring one
side is not.

## Filing authority is not blind authority

A filing-level value outranks an aggregator **only when it is an exact, confidently
extracted statement figure**. A hedged, approximate, or parenthetical value must not
win: it propagates as ground truth and inflates the narrative built on it. Prevent the
bad value at the input stage rather than correcting it downstream — that is the
cheapest layer and the only one that stops it entering the record.

## Currency

A currency code **is** the unit. Never up-case a code (that folds a minor-unit code
onto its major unit), never convert speculatively, and never rescale by venue — a
venue-keyed multiplier double-applies the moment an upstream layer already converted.
Let the code decide the scale at the one point a common basis is needed.

Suffix and provider answer different questions: the suffix is authoritative for the
*economy*, the provider for the *denomination*, which the suffix cannot express. They
agree when the provider's code normalizes to the suffix's currency.

FX resolves live first, then falls back to the static table, through the shared cache.
A few call sites read the table directly **on purpose** — reconstructing what a rate
was at some past moment — and must not be wired to live rates.

## Screener and analyzer see different data

The screener uses a single free source; the analysis pipeline uses the merged set. A
metric that is null in the screener's source silently rejects the row, so any gated
metric needs a computed fallback, and each rejection is tagged so the end-of-run
summary shows which reason dominates per exchange. A 0% pass rate for an exchange is a
data-source bug, not filtering.

Full history: `docs/CURRENCY.md`.
