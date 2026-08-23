# Currency, denomination, and price scale

Last updated: 2026-08-22

How money is represented across the analysis and reconciliation layers, and the two
independent checks that keep a price comparable to the price it is compared against.
This document explains the rationale and intended behaviour; the implementation and
its tests are authoritative. If this file and the repo disagree, trust the repo.

## A currency code *is* the unit

A London listing quoting `GBp` is not "GBP in a different unit" — `GBp` is its own
currency code, and the FX layer already treats it as one, returning a rate exactly one
hundredth of the major unit's.

That fact collapses what looked like a hard problem. There was nothing to build: the
answer was being computed correctly and then thrown away, five times over — by
up-casing the code, by letting a venue-suffix table override the provider
unconditionally, by callers passing only a ticker, by up-casing again at load, and
finally by a venue-keyed multiplication standing in for the information discarded
three layers up.

**The rule: never discard the currency code you were given, never convert
speculatively, and let the code decide the scale at the one point you need a common
basis.** The fix was deletions and preservations, not additions.

### Three tempting fixes, all rejected

- *Patch the last layer.* Whack-a-mole: it adds a sixth layer re-deriving what
  upstream already knew.
- *Add a quote-unit column to the exchange table.* Plausible — "this venue quotes in
  minor units" is a genuine venue fact — but it turns that table into a *transform*,
  which double-applies the moment an upstream layer has already converted.
- *Introduce a money type.* Correct in the abstract, disproportionate here.

### Case sensitivity is load-bearing

Canonicalisation upper-cases **without** collapsing denominations, and matching is
deliberately **case-sensitive**: the minor and major codes differ *only* in case, so a
case-folded lookup maps one onto the other. Exact alias hit wins; otherwise upper-case,
so a lower-cased major code stays the major code.

### Suffix and provider answer different questions

The venue suffix is authoritative for the **economy**; the provider is authoritative
for the **denomination**, which the suffix cannot express. When the provider's code
normalises to the suffix's currency they *agree* — the provider is merely more
specific. A real disagreement still loses to the suffix, and still warns.

A new minor-unit venue is one row in the alias table plus an FX entry. No code path
changes.

## Code agreement is necessary and not sufficient

A post-fix artifact carried a correct price in major units alongside derived levels in
minor units — **every value labelled with the same currency code.** Any helper keying
on codes passes that pair straight through. An external review proposed exactly such a
helper as the fix, and it would not have caught this.

Two complementary checks, neither subsuming the other:

- **Same economy?** Convert by code; fail closed on an unknown code, a cross-economy
  pair, or a non-positive price. This catches a minor-unit position against a
  major-unit analysis.
- **Same scale?** A tri-state verdict on whether derived levels sit within a
  plausibility band of the reference price they were derived from. This catches a
  hundred-fold disagreement *inside* one code.

`UNASSESSED` must never fold into `COHERENT`. The record-level field is
three-valued for the same reason: an early version wrote `not is_incoherent`,
collapsing "could not check" into an asserted clean result — a mistake made one layer
below a helper whose tri-state was itself tested and negative-controlled. A guard
proves the layer it covers and nothing more.

**Non-finite fails closed.** `NaN` fails every comparison, so it passed both guards
and poisoned downstream ratios silently. A non-finite reference resolves
`UNASSESSED`, never `INCOHERENT` — we could not check, which is not the same as having
checked and found a contradiction.

## Refuse, never rescale

A hundred-fold ratio against a currency with a documented minor unit has one
consistent reading, so recovery is tempting. It remains a guess about *which side is
wrong*, and these values gate portfolio actions: missing evidence about authority
resolves closed.

Refusing at the **reader** fixes every consumer at once, because each already guards
on a null. Nulling the canonical record while leaving a nested copy intact is not
refusing — a first version did exactly that, and the rejected numbers still reached
the operator through a serializer. The regression test therefore scans the whole
serialized payload for the rejected values rather than checking named fields, since
checking named fields is what missed the shadow copy.

Measured before shipping, over the full artifact corpus: 99.66% unaffected, fifteen
refusals across six tickers, zero false positives. Five were the expected venue with
the hundred-fold signature. The sixth had a corrupt *price* rather than corrupt levels
— there, refusing both is the designed behaviour. Do not assume a defect class has
only the instances you first noticed.

## Both block contracts carry the unit

The fundamentals and trade blocks each carry an explicit price-currency field, named
so it cannot be confused with the reporting currency — a company may quote in one and
report statements in another.

**The model declares the field; the code decides its value.** The value is stamped
from the merged payload, and the derived levels inherit the stamped code, because a
transcription must never *be* the unit of record.

Three defects were found by *executing* adversarial inputs rather than reasoning about
them: a value containing a regex replacement template was interpreted as one and
raised inside the node; a duplicated field line survived with a *different* value, so
a parser taking the last match got the wrong one; and a routine markdown-emphasised
line went unmatched, leaving a contradiction in place. The fix validates the code
against a strict pattern before use — which makes injection impossible by
construction rather than by escaping at each call site — replaces every occurrence,
and tolerates emphasis.

Stamping stays inside the structured-payload gate: with no payload the denomination is
unverifiable, and stamping a null would destroy a possibly-correct transcription while
adding a line to every thin-data block.

## FX resolution

Live rate first, then a static fallback table, through a process-wide cache with a
one-hour TTL. Each currency in a batch resolves independently under a concurrency
bound — an earlier design let one preflight currency gate the whole batch, so a single
pair's outage forced valid pairs onto the stale table.

Several call sites read the fallback table **directly and on purpose**: they are
reconstructing what a rate was at some past moment, and a live rate would be
semantically wrong there. Do not wire them to live FX.

Liquidity thresholds are evaluated in USD. Financial metrics stay in local currency in
tool output, and agent scoring uses ratios, which are currency-neutral.
