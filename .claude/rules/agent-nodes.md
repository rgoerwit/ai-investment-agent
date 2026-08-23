---
paths:
  - "src/agents/**"
---

# Agent nodes

## Refusals are a first-class failure, not an empty success

A provider-blocked or refused response used to return as a clean empty success and
flow downstream unlabelled. It is now a non-retryable failure kind, excluded from the
circuit breakers — a content block is not a provider-health fault.

**Detection is metadata-only.** Finish and stop reasons, structured refusal fields,
prompt-feedback block reasons. **Never prose-match the model's text**: this system
analyses risk and safety for a living, and a report discussing "safety" would
false-positive and kill a good run.

There is no same-model retry for a refusal — an identical prompt does not clear a
content block. The one exception re-issues with a *changed* request, which is why it
does not contradict the rule.

## Optional seats degrade; the run must say so

An optional cross-check seat failing is publishable by design. That makes the analysis
look *safer*, because the flags that seat would have raised simply never appear — so a
degraded run must announce itself, and `--quiet` may reduce that notice to one line
but never suppress it.

"Degraded" means output is missing, not that something recovered. A retried call that
then succeeded is not degradation, and a label that fires on routine retries is noise.

A partially-verified review reports its own reduced status rather than a recomputed
predicate: detection reads the machine-written marker, so the status reports what
occurred and cannot disagree with the text the consumer received.

## Bounded loops

Tool loops carry an iteration ceiling, a per-turn call cap, and a total deadline.
Execute a turn's tool calls concurrently and fold the results back in **argument
order**, so failure accounting stays deterministic. Check the deadline once before the
fan-out: it bounds the turn, not each call.

A response cut off at the output cap is a fragment, not an answer — and the cap is not
the only kind of partial. Key recovery on the full partial-response classifier, not on
the cap alone. Refusals are excluded by construction, since re-asking cannot clear
them.

## Writing state

Bound every artifact write with `cap_state_value`. Deliver untrusted external content
as a human-authored message rather than a system one, and wrap it in the
trust-boundary formatter — that is what keeps retrieved text from acting as
instructions.

Prefer the canonical renderer for anything a downstream prompt consumes.
Interpolating a raw Python structure into a prompt is not a serialization format, and
a truncating preview drops the decision-critical keys first, precisely because they
are ordered after the long prose ones.

Full history: `docs/AGENT_ROSTER.md`.
