---
paths:
  - "src/agents/**"
  - "src/graph/**"
---

# Parallel agent execution

Multiple agents run concurrently in this graph. Three failure classes follow, and all
three are silent.

## 1. State reducer races

When parallel agents write the **same** state field, a last-write-wins reducer keeps
one and discards the rest. Give each agent its own field, or use an explicit merging
reducer. The risk analysts write to separate `current_*_response` fields for exactly
this reason.

## 2. Message pollution

Parallel agents share `state["messages"]` through an appending reducer, so without
filtering agent A sees agent B's tool calls and reasons about them. Both halves are
required:

- **Tag outgoing messages.** Set `response.name = agent_key` on the AIMessage, and
  `msg.additional_kwargs["agent_key"] = agent_key` on ToolMessages.
- **Filter incoming messages** with `filter_messages_by_agent()` so an agent sees only
  its own history.

## 3. Retry paths

**Fix the retry path too.** If the main response is tagged, the retry response must be
tagged identically. Search for the retry variable and apply the same treatment — a
tag applied only to the first attempt is a recurring bug here, and it surfaces as an
agent reasoning about another agent's tools only under load.

## Adding a parallel agent

1. Give it state fields with an appropriate reducer, not a shared one.
2. Tag every outgoing message.
3. Filter incoming messages.
4. Check the main path **and** every retry and fallback path.
5. Bound artifact writes with `cap_state_value`.

Concurrency inside a node follows the same discipline: gather tool calls rather than
awaiting them in a loop, then fold results back in **argument order** so failure
accounting stays deterministic regardless of completion order.

Full history: `docs/AGENT_ROSTER.md`.
