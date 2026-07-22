# Codebase Memory

Last updated: 2026-07-19

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

The forensic Auditor uses `src/forensic_budget.py` as its single budget policy,
`src/tools/official_documents.py` for approved-host HTML/text/PDF evidence, and
`src/tools/forensic.py` for deterministic completeness gates and ratios. PDF
extraction is intentionally bounded and text-only: `pypdf` falls back to
`pdftotext`; approved exchange domains are built in, while direct issuer/IR
document hosts require the operator's `AUDITOR_OFFICIAL_DOCUMENT_HOSTS`
allowlist. OCR and general-purpose issuer-site crawling remain out of scope.

The Auditor maintains separate `CURRENT_STATEMENTS` and `AUDITED_BASELINE`
tracks. A signed audit opinion may establish the latest audited baseline even
when fresher interim/current statements exist, but figures are compared only
when period type/end, currency, and consolidation scope align. Fiscal-year
labels are never inferred from calendar years. Saved artifacts include the
actual Auditor budget ledger and attribution lists only models that made calls.

Management-guidance baseline evidence is owned by
`src/agents/management_guidance.py`. Before the Foreign Language Analyst runs,
it searches the latest results package and earnings bridge, checks the statutory
filing API, and in full mode extracts query-relevant passages from discovered
sources. Exchange-aware searches include the active fiscal period where the
exchange convention supports it and local filing vocabulary otherwise. An exact
ticker-matched result title can supply a bounded local listing name in any script
for the bridge search, with the previously resolved name retained only as fallback.
Search provenance is code-owned, then promoted deterministically into Senior
Fundamentals so temporary tax/regulatory benefits can affect scoring and BUY
eligibility even when no extraordinary-expense line appears. A results URL
without operating- and net-income guidance is treated as an unresolved baseline,
not evidence of durable earnings. `src/guidance_vocabulary.py` owns the
jurisdiction-specific search and excerpt vocabulary; the generic search formatter
accepts explicit priority terms and has no embedded finance-language bias.
`src/earnings_baseline.py` owns the pure status sets and scoring predicates. If
the Foreign Language Analyst returns useful research but omits or malforms only
the guidance block, code appends a conservative unresolved/search-failed block;
empty or unusable agent output still fails closed.

`src/tooling/` owns cross-cutting tool execution, audit hooks, argument policy, and untrusted-content inspection.

`src/runtime_diagnostics.py` owns artifact completion/validity and publishability checks.

`src/data_block_utils.py` owns structured data-block marker vocabulary. Use
`BLOCK_SHAPES`, `fenced_start()`, `fenced_end()`, `build_fenced_block()`,
`unfenced_label()`, and the shared tolerant fenced matcher instead of hand-rolled
`--- START` / `--- END` regexes. `tests/prompts/test_marker_parity.py` is the L0
guard for prompt marker form, parser shape parity, and source-level marker drift.

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

Ownership is now split across:

- `src/ibkr/reconciler.py` for orchestration
- `src/ibkr/analysis_index.py` for latest-analysis cache/load/update
- `src/ibkr/reconciliation_rules.py` for FX, staleness, verdict, and sell helper rules
- `src/ibkr/position_evaluator.py` for held-position routing
- `src/ibkr/watchlist_evaluator.py` for watchlist routing
- `src/ibkr/opportunity_finder.py` for off-watchlist BUY discovery
- `src/ibkr/portfolio_health.py` for portfolio-health and correlated-sell handling
- `src/ibkr/buy_stability.py` for the BUY stability / hysteresis gate

**Agents-free invariant (load-bearing):** the IBKR reconciliation path must not
import `src.agents` (importing any `src.agents.*` submodule runs the heavy
`agents/__init__`, pulling in the whole LangGraph/LLM surface). The BUY stability
gate (`src/ibkr/buy_stability.py`) is **default-on** (`BUY_STABILITY_ENABLED`,
flipped June 2026) and runs on every reconcile, so it — and the analysis index —
parse PM verdicts via the neutral, stdlib-only `src/pm_decision_parser.py`
(`canonicalize_pm_verdict`, `parse_final_decision_scores`), never via
`src.agents.pm_verdict_metadata`. That neutral module is also the single home for
`canonicalize_pm_verdict` shared by charts/report/memo, which dissolves the old
`pm_block → src.agents.pm_verdict_metadata → agents/__init__ → decision_nodes →
pm_block` circular import. Boundary tests in `tests/ibkr/test_buy_stability.py`,
`tests/ibkr/test_analysis_index.py`, and `tests/test_pm_decision_parser.py`
enforce that these paths import no `src.agents`; do not reintroduce such an import
or move the parser back under `src.agents`.

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
- `src/ibkr/reconciler.py` -> orchestration plus IBKR ownership submodules
- Snyk pre-commit redesign (July 2026): `scripts/snyk_check.sh` now takes a mode arg — `snyk-deps` BLOCKS commits (exit 1) on MEDIUM+ dependency findings (was always exit 0 / misleading `Passed`; `SNYK_ADVISORY=1` escapes), `snyk-container-base` is advisory REVIEW-REQUIRED for the base image. CI gains a token-gated, non-gating `snyk-container` job (Snyk CLI pinned in a `run:` step, scans base tag + built image) restoring the coverage the disabled Trivy jobs provided. 7-day dependency-release cooldown before remediation bumps.

Next likely large seams:

- `src/report_generator.py`

## Provenance

This note was refreshed from the current repo layout and recent refactor state.
It is intentionally shorter than older versions; use `README.md`, `CHANGELOG.md`, tests, and the live tree for deeper detail.
