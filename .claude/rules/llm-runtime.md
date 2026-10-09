---
paths:
  - "src/llm_runtime/**"
  - "src/llms.py"
  - "src/token_tracker.py"
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

## Transport failures fail over by model, in one place

Model failover is configured by model in `LLM_MODEL_FAILOVERS` and runs only on
retryable `server_error`/`timeout`, from the retry loop. **Never hand-wire a
per-node backup model for transport errors.** Any walk of a composed runnable must
descend only real LangChain composition types: attribute probing never terminates
on a mock.

## Request deadlines are sized at dispatch, never at construction

**A tier switch, re-issue or failover re-sizes from the enclosing absolute
deadline; never fix a request deadline at construction.** The tier is chosen per
call, and Google enforces the client timeout server-side, so a timeout baked into
a flex-built client rode along on the standard re-issue, degraded-standard calls
and the failover copy (the October 2026 quick PM 504s). The deadline lives in a
ContextVar set by `invoke_with_rate_limit_handling`, not on the model; the policy,
`TimeoutClass` and admission live in `src/llm_runtime/timeouts.py`. Attempt counts
are **total** attempts everywhere; translate to a vendor's `max_retries` only via
`sdk_max_retries`. A seat's budget class is data on its `SeatSpec`, never a
display-name set.

**No plain OpenAI-family or Anthropic client.** Build every one with
`new_openai_chat` / `new_anthropic_chat` in
`src/llm_runtime/adapters/transports.py`, passing `dispatch_provider` (the vendor
that receives the request) and the seat's `budget_timeout_class` explicitly. Never
infer or default the provider: it selects the request ceiling and the flex-health
key, and the factory's `_llm_runtime_provider` stamp lands only after construction.
A compatible vendor never gets a `service_tier`. Tests patch those two seam
functions, never `langchain_openai.ChatOpenAI`/`ChatAnthropic`.
`tests/architecture/test_llm_constructor_boundary.py` rejects a plain client.
Streaming stays refused until it is sized at dispatch and qualified.

**A caller outside the graph invokes through `invoke_with_rate_limit_handling`,
keyed by its seat's display name — never a bare `ainvoke` under a cap it computed
itself.** The editor review, retrospective lesson and semantic judge were moved
there; a bare `invocation_scope` would size the requests but drop the retries,
since a scoped Google request sends one SDK attempt. A seat whose calls the loop
owns but whose class is mixed declares `transport_attempts=1`. Read a cap from
`resolve_invocation_budget` in `src/agents/runtime.py`, which resolves the seat (so
its `client_timeout_seconds` holds inside the scope); a dispatch keeps the scope's
deadline but takes the ceiling of the vendor it sends to (`policy_for_provider`).
Still direct: the writer chain and the health check.

Attempt telemetry must use the transport tier actually reached, including an in-call
flex-to-standard retry or model failover. A quick-mode timeout after a flex latency
fallback is still queue-affected even if the final request used standard; capacity
and capability fallbacks do not establish that cause. Keep seat-specific tier policy
in `SeatExecutionPolicy` so legacy and bound construction agree.

## The pricing table goes stale silently

`MODEL_PRICING_PER_1M` in `src/token_tracker.py` is hand-copied from vendor pages that
change without notice, and a wrong row never fails a run: it just misreports
`total_cost_usd`, which model-choice decisions are then made from. **Re-verify every
row against the source URLs in its header comment** whenever a configured model
changes, a vendor announces a model, the header's "Last verified" date is a month old,
or a dated promotion it records lapses. Bump that date, and add new rows to the
`TestCurrentModelPricing` parametrize list in the same change.

Full history: `docs/LLM_PROVIDERS.md`.
