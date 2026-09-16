---
paths:
  - "src/llm_runtime/**"
  - "src/llms.py"
---

# Seats, bindings, and providers

**Provider identity is bound per seat group, not baked into a factory.** Every LLM in
this repository — in the graph and out of it — is constructed by
`build_model_for_seat(SeatId.X, ...)`. A provider SDK constructor may be imported only
inside `src/llm_runtime/adapters/` and the shrinking allowlist in `src/llms.py`; an
architecture test enforces that.

## Seats, not call sites

`src/llm_runtime/seats.py` holds the canonical `SeatId` → `SeatSpec` registry, keyed to name spaces
that already exist: `prompt_key` matches the prompt filename, `budget_key` matches an
output-budget entry, `callback_name` matches the token-tracking display name. **Do not
invent a second key space.**

The binding groups are deliberate. Verification seats are grouped so they can be bound
to a different vendor than the seats they check; role-adversarial pairs that share
memory and a debate barrier stay on one vendor, because splitting them would confound
"the bear case was stronger" with "that vendor writes more forceful prose". Vendor
diversity belongs at *verification* boundaries.

## Rules that must not be softened

- **Independence is enforced, not conventional.** A collapse between a base and a
  review binding is a startup error, compared on both vendor and model lineage — two
  proxies fronting one model would pass a naive name check.
- **Unknown models fail closed.** No profile, no binding, one deduped error naming the
  model. Do not add a blanket allow-unreviewed escape hatch; the repo-consistent one
  is a typed profile override declaring identity, capabilities, ladder and scope.
- **Capability is not qualification.** Profiles record vendor-documented transport
  facts; the policy module records what this repository has evidence for. Widening a
  qualification row requires running the live procedure, not reading a vendor doc.
- **Per-seat construction facts are data**, never `if seat == ...`. Temperature,
  client timeout, SDK retries and tier pins live in `SeatExecutionPolicy` so every
  caller gets them — a pin placed at a graph call site reaches only the graph.
- **One model instance per seat, never shared.** The tracking callback is bound per
  agent and carries that agent's output budget; returning one instance to several
  seats collapses both.

## Reasoning effort is a contract, not a preference

On a reasoning model the hidden reasoning draws from the same completion budget as the
answer, so an *unset* effort lets the model spend the entire budget thinking and
publish a fragment. Call sites request an ordered **preference** and the first value
the model family documents wins; `None` means send no parameter. The reserve scales
with the effort actually requested — bounding the effort alone is not enough.

Unregistered families are safe by construction (no parameter is sent) and are logged
once, so a stale table can only under-serve, never break a call.

## Every gate must read the plan

A gate that decides whether a seat runs must consult the resolved binding plan, **not
a legacy credential**. Keying on a provider key that a valid configuration does not
set wires the node and never dispatches to it — raising nothing, with the only
evidence a `NOT_RUN` status in the saved artifact.

Full history: `docs/LLM_PROVIDERS.md`.
