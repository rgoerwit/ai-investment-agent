# Codebase Memory

Last updated: 2026-04-24

This file is a durable orientation note, not the source of truth.
Use it to get context quickly, then verify against the live tree.
If this file and the repo disagree, trust the repo.

## What This Repo Is

This is a multi-agent international equity analysis system built on LangGraph.
It combines:

- parallel analyst agents
- deterministic pre-screening via `RedFlagDetector`
- adversarial bull/bear debate
- markdown report generation with charts
- ticker-isolated ChromaDB memory plus lessons learned
- IBKR portfolio and reconciliation workflows

The system is no longer just “analyze one ticker.” It also supports:

- batch analysis
- retrospective learning
- article generation
- portfolio-aware recommendations and reconciliation

## Fast Orientation

Read in this order:

1. `AGENTS.md`
2. `README.md`
3. top of `CHANGELOG.md`
4. `src/main.py`
5. `src/cli.py`
6. `src/persistence.py`
7. `src/output.py`
8. `src/runtime_services.py`
9. `src/tooling/`
10. `src/graph/`
11. `src/agents/`
12. `src/tools/`
13. `src/data/fetcher.py`
14. `src/runtime_diagnostics.py`
15. `src/validators/red_flag_detector.py`
16. `src/validators/sector_classifier.py`
17. `src/validators/metric_extractor.py`
18. `src/validators/financial_rules.py`
19. `src/validators/supplemental_extractors.py`
20. `src/validators/supplemental_flags.py`
21. `src/memory.py`
22. `src/ibkr/`

## Runtime Spine

`src/main.py` is now orchestration-first: runtime setup, macro-context prefetch, graph execution, tracing, and mode dispatch.
`src/cli.py` owns CLI parsing, validation, and output/article path resolution.
`src/persistence.py` owns saved-artifact assembly, JSON persistence, and rejection-record helpers.
`src/output.py` owns banners, CLI/report rendering, and article generation helpers.

For runtime/control-plane state design, use `docs/RUNTIME_MODEL.md` as the canonical Stage 0 model before changing storage or orchestration seams.

`src/runtime_services.py` owns runtime-scoped service binding.
`RuntimeServices` uses `ContextVar` scoping so CLI runs, graph execution, dashboard snapshot loads, and worker jobs can bind their own tool execution, inspection, provider runtimes, and hooks without sharing mutable globals by accident.

`src/graph/` owns:

- routing and sync barriers
- graph component construction
- graph wiring
- graph-scoped per-agent tool-node filtering

`src/agents/` owns the node logic:

- analyst nodes
- fundamentals and validator nodes
- research/debate nodes
- PM/trader/risk nodes
- consultant/legal/auditor nodes

`src/tools/` holds the domain tool implementations.
`src/toolkit.py` is deleted.
Package roots such as `src/__init__.py`, `src/tooling/__init__.py`, and `src/tools/__init__.py` are intentionally inert; do not assume convenience re-exports.

`src/tooling/` owns cross-cutting tool execution, audit hooks, argument policy, and untrusted-content inspection.

`src/runtime_diagnostics.py` owns artifact completion/validity and publishability checks.

## Information Flow Model

Primary agent-to-agent flow is through typed state fields, not just message history.

Important distinction:

- artifact content field: the report or degraded fallback text/json
- `artifact_statuses`: execution/completion metadata

Current semantics:

- `complete=True, ok=True`: agent ran and produced valid output
- `complete=True, ok=False`: agent ran but failed; may still leave conservative fallback content
- `complete=False`: agent did not complete

Graph barriers use completion, not validity.
Downstream decision logic should use valid content helpers where correctness matters.

## High-Value Files

If something breaks, check these first:

- `src/main.py`
- `src/graph/routing.py`
- `src/graph/components.py`
- `src/agents/analyst_nodes.py`
- `src/agents/decision_nodes.py`
- `src/agents/consultant_nodes.py`
- `src/data/fetcher.py`
- `src/runtime_diagnostics.py`
- `src/validators/red_flag_detector.py`
- `src/ibkr/reconciler.py`
- `scripts/portfolio_manager.py`

## Major Subsystems

### Data ingestion

`src/data/fetcher.py` is the core market/fundamental data pipeline.
It merges multiple sources and is a common regression surface.
Ownership now lives across:

- `src/data/source_fetchers.py`
- `src/data/metric_extraction.py`
- `src/data/merge_policy.py`
- `src/data/gap_fill.py`

### Validator

`src/validators/red_flag_detector.py` is the thin public validator facade.
Ownership now lives in:

- `src/validators/sector_classifier.py`
- `src/validators/metric_extractor.py`
- `src/validators/financial_rules.py`
- `src/validators/supplemental_extractors.py`
- `src/validators/supplemental_flags.py`

Together they parse the fundamentals `DATA_BLOCK` and drive auto-reject or risk-penalty outcomes.

### Memory

`src/memory.py` provides ticker-isolated ChromaDB memory plus macro/lesson retrieval.
Memory outages should degrade analysis, not abort it.
Memory writes are inspected per document with `SourceKind.memory_write`; blocked writes are skipped and fully blocked batches return `False`.

Macro surfaces are intentionally split:

- `MacroEventsStore` holds sparse portfolio-detected discrete shocks
- `src/macro_context.py` holds a short cached regional regime brief under `results/.macro_context_cache/`
- saved analysis JSON records macro-context status/region metadata separately so operator review can tell whether the pre-graph summarizer ran

### Reporting

Main files:

- `src/report_generator.py`
- `src/charts/`
- `src/article_writer.py`

Large graph-state artifacts are bounded with `src/agents/output_limits.py::cap_state_value()` at the write points so oversized LLM output does not silently bloat state.

### Content-ingress hardening

The current trust-boundary model is:

- inspect tool output in the tool execution plane
- inspect cached or replayed untrusted context before it re-enters prompts
- inspect financial-API free-text fields before they become prompt-visible
- keep blocked content out of primary prompt paths rather than storing sentinel text as valid analysis

### IBKR path

Main files:

- `src/ibkr/`
- `scripts/portfolio_manager.py`

This path now includes:

- holdings + watchlist loading
- order-awareness
- recommendation/reconciliation logic
- portfolio-health and macro-event handling

## Testing Guidance

The test suite is broad and behavior-heavy.
Strong coverage areas include:

- graph routing
- validator behavior
- memory isolation
- fetcher/data edge cases
- chart extraction/rendering
- IBKR reconciliation

Patch owning modules in tests, not facades.
Use facade tests only for real public APIs.
Prefer patching `src.runtime_services`, `src.tools.*`, or the owning helper used at call time.

## Practical Notes

- Preserve the distinction between execution failure and bad business metrics.
- Do not put raw error strings into primary report fields unless that is an intentional degraded output.
- Optional cross-checkers may fail without blocking publication if core artifacts are still valid.
- For long CLI phases, keep user-visible progress at the caller layer and make deeper logging optional.

## Current Refactor State

Already split:

- `src/agents.py` -> `src/agents/`
- `src/toolkit.py` -> `src/tools/` with facade removed
- `src/graph.py` -> `src/graph/`
- `src/validators/red_flag_detector.py` -> facade plus validator ownership submodules

Recent completed control-plane/security work:

- runtime-scoped service container via `RuntimeServices`
- memory-write inspection
- financial-API text-field inspection
- artifact bounding via `cap_state_value()`
- broader heuristic prompt-injection coverage
- `src/main.py` -> orchestration plus `src/cli.py`, `src/persistence.py`, and `src/output.py`

Next likely large seams:

- `src/ibkr/reconciler.py`
- `src/report_generator.py`

## Provenance

This note was refreshed from the current repo layout and recent refactor state.
It is intentionally shorter than older versions; use `README.md`, `CHANGELOG.md`, tests, and the live tree for deeper detail.
