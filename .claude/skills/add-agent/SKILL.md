---
name: add-agent
description: >
  Add a new agent or LLM seat to the graph. Covers the prompt file, seat registry,
  node factory, graph wiring, ticker-isolated memory, and the tests that must
  accompany it.
disable-model-invocation: true
---

# Adding an agent

Six steps, in order. Each one is load-bearing; skipping the seat registry in
particular produces an agent that constructs but is never dispatched, and that failure
is silent.

## 1. Prompt

Create `prompts/<agent_key>.json` with `agent_key`, `agent_name`, `version`,
`category`, `requires_tools`, `system_message`, and a `metadata` block carrying
`last_updated` and a `changes` line. Edit it only through Python's `json` module —
the prompt-contract rule loads automatically when you touch that directory.

## 2. Seat

Add a `SeatId` and `SeatSpec` in `src/llm_runtime/seats.py`. Three keys must line up
with name spaces that already exist; do not invent a second one:

- `prompt_key` matches the JSON filename
- `budget_key` matches an `AGENT_OUTPUT_BUDGET_FRACTIONS` entry
- `callback_name` matches the token-tracking display name

Choose the binding group deliberately. A *verification* seat belongs in `review`,
where vendor diversity is the point; an analysis input belongs in `base`. Declare
`requires` honestly — capability validation is what catches a chat-only model bound to
a tool-calling seat, before the run rather than during it.

Per-seat call semantics — temperature, client timeout, SDK retries, tier pins — are
`SeatExecutionPolicy` data, never an `if seat == ...` at a call site. Putting a pin at
the call site means only that caller gets it.

## 3. Node factory

Add the factory in the owning module under `src/agents/`, following the sibling
patterns. Construct the model with `build_model_for_seat(SeatId.NEW_AGENT, ...)` —
never a provider constructor directly. An architecture test enforces that boundary.

If the node runs in parallel with others, three things are mandatory: give it its own
state field rather than sharing one under a last-write-wins reducer, tag its outgoing
messages with the agent key, and filter incoming messages to its own history.
Remember the retry path — tagging applied only to the main response is a recurring
bug here.

Bound every artifact write with `cap_state_value`, and every blocking call with the
repository's hard-timeout wrapper.

## 4. Graph

Register the node in `src/graph/` with its routing and tool-node wiring. **Every gate
that decides whether a seat runs must read the binding plan, not a legacy credential.**
A gate keyed on a provider key that a newer configuration legitimately does not set
will wire the node and never dispatch to it, raising nothing; the only evidence is a
`NOT_RUN` status in the saved artifact.

## 5. Memory

If the agent needs memory, add its instance through the ticker-isolated factory so it
gets its own per-ticker collection. Never reach for a global memory object — that is
what lets one company's data contaminate another's analysis.

## 6. Tests

Add `tests/agents/test_<agent>.py`, and extend the legacy-parity test if the seat has
a legacy construction path. Cover the edge cases this system actually hits: missing
data, a refused or empty provider response, malformed output, and the timeout path.

Then run the owning directory, `make test-prompts`, and `poetry run ruff check src/`.

Full history: `docs/AGENT_ROSTER.md`.
