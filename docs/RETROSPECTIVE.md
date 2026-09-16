# The retrospective: outcomes, attribution, and lessons

Last updated: 2026-08-22

How past verdicts are compared against market outcomes, what the resulting vocabulary
does and does not claim, and why most of the apparatus exists to *withhold* a
conclusion rather than reach one. This document explains the rationale and intended
behaviour; the implementation and its tests are authoritative. If this file and the
repo disagree, trust the repo.

## How it works

Every analysis saves a prediction snapshot at zero model cost. On a later run the
system loads past snapshots, fetches the current price and a local benchmark, computes
the excess return, and — only past a threshold — spends one cheap model call to write
a lesson. Lessons are stored in a global collection and the relevant ones are injected
into later research prompts.

Excess return against a benchmark, not raw return, is the whole basis: it strips
market-wide and currency moves that would otherwise manufacture bogus lessons.

## What the vocabulary claims

These tokens are persisted and asserted in tests. **Do not rename them.** The entire
apparatus rests on one arithmetic fact: residual equals stock return minus benchmark
return, which says only what the country index does *not* account for. It never says
what did.

| Dominant driver | Means | Does **not** mean |
|---|---|---|
| `MARKET` | the benchmark leg dominates | "macro caused it" |
| `RESIDUAL` | the residual leg dominates | "a company problem was proved" |
| `MIXED` | neither dominates | "market-driven" — it means *could not attribute* |
| `UNKNOWN` | no usable attribution | "the benchmark was flat" |

A *missing* benchmark is a third thing again: unassessable. It produces no trigger, no
model call, and no stored lesson, and it never becomes `UNKNOWN`.

| Lesson scope | Means |
|---|---|
| `CONTEXTUAL` | associated with a recorded macro regime; usable only where that regime matches |
| `UNRESOLVED` | an outcome we cannot responsibly explain; keep for review, never inject |
| `VALIDATED` | reserved for a future evidence-backed post-mortem; **no producer exists, and no gate may key on it** |

Eligibility is a *separate* question from scope: scope is about the outcome,
eligibility is about whether the record carries what retrieval needs to apply it. Both
must pass.

## The specimen

A defensive name was rejected on a day the local index fell 20% while the stock rose
6%. Excess return was +26%, the outcome scored "wrong", and the stored lesson advised
future runs that such names are attractive *"even if they appear technically
overextended by standard valuation metrics"* — a rule to relax valuation discipline,
induced from a benchmark crash. The regime at decision time was in the snapshot and
was never passed to the model, which was nonetheless asked to choose between a macro
explanation and an operational one.

## An unattributed move is not a regime observation

The first real batch after the rewrite failed its own acceptance gate. Every lesson
labelled `CONTEXTUAL` took an *unattributed* residual as its premise and prescribed
de-weighting fundamentals — the same defect as the specimen with the sign flipped.

The mapping had been "residual implies unresolved, everything else implies
contextual". Measured over the batch: **zero market-dominated outcomes**, so 100% of
the contextual set had arrived as `MIXED` — *neither leg dominates*, i.e. could not
attribute — relabelled as market-dominated. The permissive direction of the same
category error the provenance work exists to prevent.

The scope stamp is not merely a label: it is printed into the lesson prompt and gates
the instruction telling the model that an unexplained residual is not a diagnosis. So
every ambiguous outcome was generated with that instruction switched off. One defect,
two effects.

Eligibility is now an explicit, persisted, **positive** property: injectable requires
market dominance, a recorded regime, and a regime that demonstrably held still.
Everything else is review-only with a stored reason, because "the driver was mixed" is
a finding about the snapshot while "the macro cache was stale" is a finding about the
run.

`is False`, not `is not True`: an unknown regime must not qualify. An argument that
the strict form would leave the branch dead used an *unconditional* distribution to
reason about a *conditional* branch that never sees it.

Quarantine, not deletion: retrieval requires the marker positively, so pre-policy
records are withheld with no deletion and no regeneration, and stay readable.

## Attribution has two views and only two additive legs

Excess is price minus benchmark by construction, so a third additive leg would
double-count. Two internally-exact views are emitted: a local relative view where
market plus residual equals the price return, and a multiplicative investor view
combining local and currency returns.

**Currency is reported and never competes for the dominant driver.** The second leg is
named *residual*, **not alpha**: with no sector benchmark it is "what the country
index does not explain" and still contains sector rotation. The prompt says so.

Recorded hazards — regulatory and structural flags at decision time — are carried into
the prompt and **noted, never tested**: a country benchmark cannot net out that kind of
exposure, and the prompt forbids naming a mechanism absent from its inputs.

## The retrospective may not adjudicate the thesis

Pre-registered thesis-break criteria reach the snapshot, but whether one fired is a
question about filings published *after* the decision, which this path never fetches.
The status is permanently unevaluated and the prompt says so rather than inviting a
guess.

Two prompt rules exist because of observed evasions rather than from principle: no
failure mechanism absent from the inputs, and no contrastive clause demoting valuation
or growth. The model obeyed an instruction to condition on regime and then undid it
with a trailing "rather than…" clause, so restating the principle would not have
helped.

## Operability

Runs are budgeted, spend the budget on the best evidence first, and select
deterministically so two runs over an unchanged corpus choose the same set — a
reshuffling budget would starve the same snapshots forever. A memo records *evaluated*
rather than *lesson written*, because a snapshot found below threshold otherwise pays
fresh network round-trips on every future run. "Processed" means the lesson is durably
stored, not that pricing happened; proving that needs a *second* run in the test, since
a single-run assertion passes either way.

A dry-run mode reports projected cost with no fetch, no model call, and no write. Run
it before a batch.

**Expect near-zero injectable outcome lessons, and do not read that as failure.**
Triggering requires a large residual while market dominance requires the market leg to
be larger still. Same-ticker prior-rejection records carry the practical load, and
those are fetched deterministically by metadata — "have I screened this ticker out
before?" is an exact-match fact that similarity search can simply lose.

## Provenance versus weighting

Record model names: they are what make runs comparable over time. Do **not** key
judgements on them. A name-to-quality table matched **zero** stored snapshots, pinning
that factor at a default — which, combined with a retrieval floor, silently made whole
classes of lesson unreachable. Weighting now keys on the binding layer's own closed
enum, so a new model generation needs no edit and cannot re-rot.

**A floor is only meaningful against every term that can lift a candidate over it.**
The retrieval score sums reliability and relevance, which are incommensurable; a
discount applied for reliability reasons can move a record across a relevance
boundary. That is currently benign because the relevance terms dominate for the case
that matters. Before declaring a scored class unreachable, enumerate every branch
contributing to its score — a claim here that quick-mode records were unreachable was
wrong precisely for skipping one branch.

## One historical trap

The lessons collection kept reappearing empty, and the cause was not external: the
test suite was deleting it. An unmocked cleanup call with no scope argument wiped
every collection in the real store. Test isolation is now redirected at the
configuration chokepoint, which is the single point between a test and real data
because the store constructs its own client from configuration. If the collection
empties again, that is a **new** defect — check the isolation guard first.
