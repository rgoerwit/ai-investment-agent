# Codebase Memory

Last updated: 2026-03-14

This note is a durable orientation brief for the current repository state. It is meant to reduce repeated "learn the codebase" work by capturing the architecture, runtime flow, history, and practical hotspots in one place.

## Snapshot

- Repo: `investment-agent-public`
- Current workspace branch: `feat-prompt-refinements-corrections`
- Current `HEAD` when this note was written: `9b4d798`
- `main` points to release tag `3.9.2` at commit `c8ab4d6`
- Worktree was dirty during review: `src/main.py` had uncommitted user changes
- Git history is concentrated in one maintainer: `Richard Goerwitz` dominates the commit history

## What This Project Is

This is a Python-based multi-agent international equity analysis system. It uses LangGraph to orchestrate a debate-style workflow of specialized analyst agents, mostly backed by Gemini models, with optional OpenAI cross-validation. The project started as a report generator for single-ticker analysis and has grown into a broader investment research and portfolio-reconciliation toolkit.

Primary value proposition:

- Ex-US equity analysis with exchange/currency awareness
- Multi-agent thesis generation instead of a single LLM call
- Deterministic red-flag screening before expensive debate
- Visual markdown output with charts
- Persistent memory / retrospective learning using ChromaDB
- Newer IBKR portfolio-awareness and reconciliation workflows

## Runtime Spine

The fastest way to understand the live application path is:

1. `src/main.py`
2. `src/graph.py`
3. `src/agents.py`
4. `src/toolkit.py`
5. `src/data/fetcher.py`
6. `src/validators/red_flag_detector.py`
7. `src/charts/`
8. `src/memory.py`
9. `src/ibkr/`

### CLI / entrypoint

`src/main.py` is the main CLI entrypoint. Important flags include:

- `--ticker`
- `--quick`
- `--strict`
- `--quiet`
- `--brief`
- `--output`
- `--imagedir`
- `--article`
- `--retrospective-only`
- `--no-memory`
- `--trace-langfuse`

Operationally, `main.py` is responsible for:

- validating env/config
- resolving output and image paths
- constructing the graph
- running the analysis
- formatting/writing reports
- optionally generating an article
- optionally running retrospective processing

### Graph orchestration

`src/graph.py` is the architectural center. The current graph matches the README fairly closely:

- parallel analyst fan-out from `Dispatcher`
- fundamentals barrier for Junior + Foreign Language + Legal Counsel
- Senior Fundamentals synthesis
- deterministic Financial Validator
- fast-fail route on `REJECT`
- Bull/Bear debate rounds with sync barriers
- Research Manager synthesis
- Valuation Calculator and optional Consultant
- Trader
- three risk analysts in parallel
- Portfolio Manager
- Chart Generator

One implementation detail worth remembering: this repo has custom per-agent tool nodes via `create_agent_tool_node()`. That code exists to prevent parallel agents from consuming each other's tool results in shared state.

### Agents

`src/agents.py` is large and function-factory oriented. It contains the creation logic for:

- analyst nodes
- researcher nodes
- research manager
- trader
- risk debaters
- portfolio manager
- consultant
- legal counsel
- auditor
- financial validator wrapper

This file is a major behavior hotspot because many product changes land as prompt/context changes, stricter guardrails, or node-level post-processing here.

## Major Subsystems

### Data ingestion and normalization

`src/data/fetcher.py` is the core market/fundamental data pipeline. The repo docs describe a multi-source fallback chain using yfinance, yahooquery, FMP, EODHD, Alpha Vantage, and some web fallback behavior. Expect data quality/coverage logic here to be both important and fragile.

Related files:

- `src/data/eodhd_fetcher.py`
- `src/data/fmp_fetcher.py`
- `src/data/alpha_vantage_fetcher.py`
- `src/data/validator.py`
- `src/fx_normalization.py`
- `src/ticker_utils.py`
- `src/ticker_corrections.py`

### Tool surface exposed to agents

`src/toolkit.py` defines the tools agents can call. This includes market data, news, sentiment, legal/tax searches, foreign-language search, value-trap research, and liquidity helpers.

Important practical note:

- The Tavily integration includes XML wrapping / truncation logic as a prompt-injection boundary.

### Deterministic validator

`src/validators/red_flag_detector.py` is a key non-LLM safety layer. It parses the structured `DATA_BLOCK` and performs threshold-based auto-reject or penalty decisions. This is a major product differentiator and a common regression surface.

### Memory and retrospective learning

`src/memory.py` uses ChromaDB plus Gemini embeddings for ticker-isolated memory collections.

Important characteristics:

- ticker-scoped memory isolation
- retrying embedding calls
- persistence in `./chroma_db`
- stale embedding model detection / recreation

Related retrospective learning lives in `src/retrospective.py`.

### Reporting and charts

The output pipeline is markdown-first.

- `src/report_generator.py`
- `src/charts/chart_node.py`
- `src/charts/generators/football_field.py`
- `src/charts/generators/radar_chart.py`
- `src/charts/extractors/`
- `src/thesis_visualizer.py`
- `src/article_writer.py`

The chart generator now runs after the PM verdict so visuals reflect the final decision, not just raw valuation context.

### IBKR and portfolio-aware workflows

This is one of the biggest recent expansions of scope.

Files to know:

- `src/ibkr/client.py`
- `src/ibkr/models.py`
- `src/ibkr/ticker.py`
- `src/ibkr/ticker_mapper.py`
- `src/ibkr/order_builder.py`
- `src/ibkr/reconciler.py`
- `src/ibkr/portfolio.py`
- `src/ibkr/throttle.py`
- `scripts/portfolio_manager.py`

The evaluator is no longer just "analyze one ticker." It increasingly supports:

- reconciling AI verdicts against live holdings
- stale-analysis review flows
- BUY vs ADD vs TRIM vs SELL decisions
- lot-size rounding and exchange-aware order sizing
- watchlist reconciliation
- live-order annotations

Recent git history shows this is an active area.

## Repository Layout

Useful top-level directories/files:

- `src/` application code
- `tests/` test suite
- `scripts/` operational scripts
- `docs/` human docs, still incomplete
- `examples/` sample inputs and helpers
- `terraform/` infra/deployment configuration
- `images/` generated chart examples
- `scratch/` generated or ad hoc analysis artifacts
- `README.md` public-facing overview
- `CLAUDE.md` dense developer-oriented notes
- `CHANGELOG.md` strongest source for feature evolution

Observed script inventory includes:

- `find_gems.py`
- `portfolio_manager.py`
- `run_pipeline.sh`
- `run_tickers.sh`
- quick/loud/verbose run variants

## Tech Stack

From `pyproject.toml` and the code:

- Python 3.11-3.12 supported, README recommends 3.12+
- Poetry for dependency management
- LangChain 1.x / LangGraph 1.x
- Google Gemini via `langchain-google-genai`
- Optional OpenAI consultant path via `langchain-openai`
- Optional Anthropic writer path
- Pydantic Settings for config
- ChromaDB for memory
- pandas / numpy / matplotlib / seaborn
- pytest / ruff / black / mypy

## Tests and Quality Signals

At the time of review:

- about 168 files under `src/` and `tests/`
- 109 test files
- tests are organized by domain rather than only flat filenames

The test suite appears strongest around:

- graph routing
- validator logic
- memory isolation
- chart extraction/rendering
- fetcher behavior
- IBKR behavior
- prompt loading / content

This repo relies heavily on tests to pin behavior around prompt/schema-driven code and edge-case financial parsing.

## Configuration Model

`src/config.py` uses Pydantic Settings and centralizes:

- API key access
- file-system paths
- telemetry toggles
- rate-limit settings
- env validation

The app is designed to fail early when required keys are missing. The required runtime keys documented in code are mainly:

- `GOOGLE_API_KEY`
- `FINNHUB_API_KEY`
- `TAVILY_API_KEY`

Optional but important:

- `EODHD_API_KEY`
- `FMP_API_KEY`
- `OPENAI_API_KEY`
- LangSmith / Langfuse keys

## Git History Summary

High-confidence project evolution, based on `README.md`, `CHANGELOG.md`, tags, and recent commits:

### Foundation

- Initial release: `6f52e2a` (`v3.0.0`)
- Early project shape centered on single-ticker multi-agent equity analysis

### Major expansions

- consultant / cross-model review added
- legal counsel and stricter structured data blocks added
- red-flag detector added
- football-field and radar-chart output added
- auditor introduced, then later repositioned/demoted in influence
- value-trap detector added
- prompt hardening, truncation controls, and attribution handling improved
- retrospective learning and lessons-learned memory added
- batch pipeline / find-gems workflow added
- IBKR integration and portfolio reconciliation added

### Recent release line

Recent tags visible locally:

- `3.9.2`
- `3.9.0`
- `3.7.0`
- `3.6.1`
- `3.6.0`
- `3.5.0`
- `3.3.0`
- `3.2.0`
- `3.1.x`
- `3.0.x`

### Current branch vs main

`main` is at `3.9.2`, but the active branch reviewed here is newer and includes unreleased work such as:

- more IBKR throttling / 429 retry logic
- exchange and FX correctness fixes
- macro event feedback loop work
- watchlist/candidate dedup rules
- live-order and stale-analysis reconciliation improvements

## Practical Hotspots

If something breaks, the highest-value files to inspect first are usually:

- `src/main.py`
- `src/graph.py`
- `src/agents.py`
- `src/toolkit.py`
- `src/data/fetcher.py`
- `src/validators/red_flag_detector.py`
- `src/prompts.py`
- `src/report_generator.py`
- `src/ibkr/reconciler.py`

## Doc / Reality Mismatches To Expect

- `docs/INDEX.md` still marks much of the docs tree as "coming soon"
- the README is directionally accurate, but some implementation details have continued moving on feature branches
- some of the most up-to-date system behavior is better captured in `CHANGELOG.md`, tests, and commit messages than in the docs tree

## Recommended Reorientation Path For Future Sessions

For a fast refresh without repeating full research:

1. Read this file
2. Read `README.md`
3. Read `CHANGELOG.md` top sections
4. Check `git status --short`
5. Check `git log --oneline --decorate -n 20`
6. Read `src/main.py` and `src/graph.py`
7. If the task is portfolio-related, jump directly into `src/ibkr/` and `scripts/portfolio_manager.py`
8. If the task is analysis-quality related, jump into `src/agents.py`, `src/toolkit.py`, `src/data/fetcher.py`, and `src/validators/red_flag_detector.py`

## Provenance

This note was prepared by reviewing:

- `README.md`
- `CLAUDE.md`
- `CHANGELOG.md`
- `pyproject.toml`
- `docs/INDEX.md`
- `src/main.py`
- `src/graph.py`
- `src/config.py`
- `src/memory.py`
- `src/toolkit.py`
- `src/ibkr/reconciler.py`
- selected tests
- git branch / tag / shortlog / README history / recent commit history
