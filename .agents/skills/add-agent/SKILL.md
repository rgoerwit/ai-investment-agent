---
name: add-agent
description: Add or substantially change an analysis agent, graph node, or LLM seat in this repository. Use when introducing a new agent role or changing its prompt, tools, state, routing, persistence, attribution, and tests.
---

# Add an analysis agent

First determine whether the requested concept is an LLM seat, deterministic validator,
tool, or orchestration node. Extend the owning abstraction instead of forcing every
concept into an agent shape.

For an LLM-backed role:

1. Read `docs/AGENT_ROSTER.md`, `docs/PROMPT_CONTRACTS.md`, and the relevant graph and
   runtime modules.
2. Add or update the prompt through structured JSON handling under `prompts/`.
3. Register the canonical seat and capabilities in `src/llm_runtime/seats.py`; construct
   it through `src/llm_runtime/construction.py`.
4. Add node logic in the owning `src/agents/` module and tools in the owning
   `src/tools/` module. Tool execution must stay inside the inspection and audit chain.
5. Update state, reducers, graph wiring, retry/fallback behavior, output budgets,
   persistence, attribution, and publication requirements only where the role needs
   them. Parallel branches require isolated fields or an intentional reducer.
6. Add contract tests for seat registration, prompt loading, graph topology, attribution,
   malformed/refused output, retry behavior, and saved-artifact semantics.
7. Run `make test-prompts`, the relevant targeted tests, and `make check-all`.

Do not copy thresholds, model defaults, or enum values into this skill. Read them from
the owning code and update that source plus its tests when behavior changes.
