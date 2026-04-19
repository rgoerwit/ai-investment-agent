# Runtime Model

Last updated: 2026-04-18

This document defines the canonical runtime/control-plane model for the operator workbench roadmap.
It is a design contract for future storage and orchestration work, not an implementation description.

## Purpose

Freeze the runtime state model before storage, scheduling, and observability work begins so Stage 1 can be implemented without rediscovering ownership, identifiers, status values, or migration inputs.

## Non-Goals

- No runtime behavior changes
- No storage implementation
- No schema migrations
- No CLI, dashboard, or portfolio behavior changes
- No attempt to bridge eval `run_id` values into future pipeline `run_id` values in this stage

## Current Runtime State Map

| Concern | Current source of truth | Current writer | Current readers | Notes |
| --- | --- | --- | --- | --- |
| Analysis artifacts | `results/*_analysis.json` | `src.main.save_results_to_file()` | Reconciler, portfolio manager, dashboard drilldown/snapshot paths, retrospective/eval paths | Saved JSON artifact is the richest persisted analysis payload today; it now records macro-context execution metadata alongside run summary and token usage. |
| Latest-per-ticker analysis view | `.{results_dir}.latest_analyses_index.json` | `src.ibkr.reconciler.update_latest_analyses_index()` and `load_latest_analyses()` rebuild path | Reconciler, recommendation service, portfolio manager via `load_latest_analyses()` | Current supported index baseline is `version = 2`. |
| Cache/lock compatibility artifact | `.{results_dir}.latest_analyses_index.lock` | `src.ibkr.reconciler._analysis_index_lock()` | Reconciler incremental/full index writers | This is a concurrency artifact, not a migration data source. |
| Dashboard refresh jobs | `runtime/ibkr_dashboard/jobs.sqlite` | `src.web.ibkr_dashboard.job_store.RefreshJobStore` | Dashboard API/UI, dashboard worker | Separate SQLite store today. |
| Pipeline freshness marker | `results/.pipeline_last_run.json` | `scripts/run_pipeline.sh` | `src.ibkr.screening_freshness.load_screening_freshness()` | Marker-only programmatic API today. |
| Regional macro context cache | `results/.macro_context_cache/*.json` | `src.macro_context.get_macro_context()` | `src.main.run_analysis()` prefetch path, News Analyst via `TradingContext` | Pre-graph cached regime brief; distinct from `MacroEventsStore`. |
| Token/cost data | In-memory singleton in `src.token_tracker.py` | `src.token_tracker.TokenTracker` | `src.main`, report/logging paths | No durable persistence today beyond the saved analysis snapshot of the current run. Totals may include pre-graph macro summarization calls when they execute. |
| Observability traces | SDK-managed Langfuse root trace context via `src.observability.py` | Workflow entrypoints such as `src.main.run_with_args()` plus nested LangChain callbacks and explicit tool observations | External Langfuse UI, logs, article/retrospective flows, and operator diagnostics | Langfuse is opt-in per run; when disabled the app uses a no-op observability runtime and skips tracing overhead. |
| Eval run identity | Capture manifests under `evals/captures/...` | `src.eval.baseline_capture` | Eval/capture tooling | Current eval `run_id` is a separate concept from future pipeline run identity. |

## Current Compatibility Baselines

## Current Observability Shape

The app now treats Langfuse as the primary observability system when explicitly enabled.

Current rules:

- `src.observability.py` owns the Langfuse adapter and the no-op fallback.
- Workflow boundaries create root traces; lower-level helpers consume callbacks or the active trace context.
- `src.main.py` owns the root analysis trace for normal CLI analysis runs.
- Standalone article and retrospective entrypoints create their own root traces.
- `src.tooling/runtime.py` owns explicit tool observations for the shared tool execution seam.
- When Langfuse is disabled, callbacks, prompt fetches, scores, and flush work are bypassed.

Current trace contract:

- Trace names:
  - `analysis:{ticker}`
  - `article:{ticker}`
  - `retrospective:{ticker}`
- Stable tags:
  - workflow (`analysis`, `article`, `retrospective`)
  - run mode (`quick`, `full`, `article_only`, `retrospective_only`)
  - coarse operator toggles such as consultant/auditor enablement
- Stable metadata:
  - `ticker`
  - `session_id`
  - `environment`
  - `run_mode`
  - `deep_model`
  - `quick_model`
  - `prompt_source`
- `release`

## Trading Context Notes

`TradingContext` remains the graph-config seam for non-artifact runtime context.
It now also carries optional pre-graph macro fields:

- `macro_context_report`
- `macro_context_region`
- `macro_context_status`

These fields are advisory context only. They do not change graph topology,
barrier semantics, or artifact validity rules.

The macro summarizer itself remains a pre-graph helper rather than a graph node.
When it executes under an active analysis trace, it uses the standard callback
path so token/cost reporting and Langfuse tracing stay aligned with other
LLM-backed surfaces.

Batch/session note:

- `LANGFUSE_SESSION_ID` may be supplied by an operator or batch runner to group multiple CLI invocations under one Langfuse session.
- If it is not set, the CLI uses its normal per-run generated session identifier.

Known boundary:

- Local `src.token_tracker.py` remains the repo-owned accounting source.
- Langfuse callback-based generation tracking may undercount vendor-specific reasoning or "thinking" tokens.
- The app intentionally avoids adding duplicate explicit Langfuse generations around normal LangChain-traced LLM calls.

### Reconciler latest-analysis cache

- Current supported baseline: `version = 2`
- Source: `_ANALYSIS_INDEX_VERSION = 2` in `src/ibkr/reconciler.py`
- Current behavior for non-v2 payloads: treat as invalid and rebuild from file scan

Stage 1 may preserve this behavior. It is not required to add explicit import support for older cache payload versions unless real legacy payloads are discovered.

### Dashboard refresh jobs

Current status enums:

- `jobs.status`
  - `queued`
  - `running`
  - `completed`
  - `partial`
  - `failed`
  - `cancelled`
- `job_tickers.status`
  - `pending`
  - `running`
  - `completed`
  - `failed`
  - `cancelled`

Known limitation:

- A worker crash can leave jobs stuck in `running` indefinitely.
- Stage 1 must not make this worse.
- Stage 1 should create a seam for later stale-running-job recovery, but that recovery mechanism is out of scope for Stage 0.

## Canonical Entities

### `pipeline_runs`

One row per end-to-end screening pipeline run.

Required fields:

- `run_id`
- `started_at`
- `finished_at`
- `stage_started`
- `stage_completed`
- `status`
- `flags_json`
- `screening_file`
- `results_dir`
- `candidate_count`
- `buy_count`
- `error`

Frozen `status` values:

- `running`
- `completed`
- `failed`
- `partial`
- `cancelled`

Ownership:

- Future writer: pipeline orchestration seam
- Future readers: screening freshness, dashboard run history, notifications, operator diagnostics

### `pipeline_run_tickers`

One row per `(run, stage, ticker)`.

Required fields:

- `run_id`
- `stage`
- `ticker`
- `status`
- `analysis_id`
- `started_at`
- `finished_at`
- `error`

Frozen `status` values:

- `pending`
- `running`
- `completed`
- `failed`
- `cancelled`

Ownership:

- Future writer: pipeline orchestration seam
- Future readers: dashboard run history, failure diagnosis, notifications

### `analyses`

One row per completed saved analysis artifact set.

Required fields:

- `analysis_id`
- `ticker`
- `analysis_date`
- `created_at`
- `file_path`
- `markdown_path`
- `article_path`
- `is_quick_mode`
- `verdict`
- `health_adj`
- `growth_adj`
- `zone`
- `position_size`
- `current_price`
- `currency`
- `fx_rate_to_usd`
- `entry_price`
- `stop_price`
- `target_1_price`
- `target_2_price`
- `conviction`
- `sector`
- `exchange`
- `token_cost_usd`
- `run_id`

Critical semantic rules:

- `current_price`, `entry_price`, `stop_price`, `target_1_price`, and `target_2_price` are local-currency prices.
- `currency` is the trading currency for those values.
- `fx_rate_to_usd` is the local-to-USD conversion rate.
- These fields must not be reinterpreted as USD during migration.

Identity and path rules:

- `analysis_id` is the durable identity.
- `file_path` is an artifact locator, not the record identity.
- Current path strings are not normalized consistently.
- Stage 1 migration must normalize durable artifact paths relative to `RESULTS_DIR`.

Ownership:

- Future writer: `src.main.save_results_to_file()`
- Future readers: `src.ibkr.reconciler.py`, `src.ibkr.recommendation_service.py`, `scripts/portfolio_manager.py`, dashboard drilldown/snapshot paths

### `refresh_jobs`

Carry forward the dashboard queue with minimal semantic change.

Required fields:

- `job_id`
- `created_at`
- `started_at`
- `finished_at`
- `scope`
- `results_dir`
- `watchlist_name`
- `quick_mode`
- `refresh_limit`
- `max_age_days`
- `status`
- `error`

Frozen `status` values:

- `queued`
- `running`
- `completed`
- `partial`
- `failed`
- `cancelled`

Ownership:

- Future writer/reader: shared storage adapter behind `src.web.ibkr_dashboard.job_store.py`
- Future readers: dashboard API/UI, worker, notifications

### `refresh_job_tickers`

Required fields:

- `job_id`
- `ticker`
- `status`
- `error`
- `analysis_id`

Transitional compatibility field:

- `output_path`

Compatibility note:

- Current store persists `output_path`.
- Future durable link target should be `analysis_id`.
- Stage 1 may keep `output_path` temporarily for compatibility, but it is transitional rather than canonical.

Ownership:

- Future writer: dashboard worker and shared storage adapter
- Future readers: dashboard API/UI and diagnostics paths

### `notifications`

Required fields:

- `notification_id`
- `created_at`
- `sent_at`
- `channel`
- `event_type`
- `ticker`
- `run_id`
- `payload_json`
- `status`
- `error`

Trigger families that should be supported later:

- pipeline completion
- dashboard refresh completion/failure
- portfolio or reconciler alert conditions
- stale-analysis warnings

Ownership:

- Future writers: pipeline completion seam, dashboard refresh seam, portfolio/reconciler alert seam
- Future readers: notification sender and operator diagnostics

### `run_metrics`

Required fields:

- `metric_id`
- `run_id`
- `analysis_id`
- `agent_key`
- `event`
- `prompt_tokens`
- `completion_tokens`
- `cost_usd`
- `recorded_at`

Ownership:

- Future writer: token tracker / agent runtime seam
- Future readers: dashboard observability and run summaries

## Relationship Model

- `pipeline_runs.run_id -> pipeline_run_tickers.run_id`
- `pipeline_runs.run_id -> analyses.run_id` when the analysis was produced by a pipeline run
- `analyses.analysis_id -> refresh_job_tickers.analysis_id` after the worker/job migration completes
- `pipeline_runs.run_id -> notifications.run_id`
- `pipeline_runs.run_id / analyses.analysis_id -> run_metrics`

## Indexed Metadata vs Artifact Content

### Indexed into `analyses`

These are queryable/indexed fields:

- ticker
- analysis_date
- verdict
- is_quick_mode
- health/growth/zone
- position size
- key local-currency price fields
- currency and FX
- conviction
- sector
- exchange
- token cost
- run linkage
- artifact paths

### Left in artifact files

These remain in the saved JSON or markdown files:

- full reports
- debate history
- `prediction_snapshot`
- `artifact_statuses`
- `analysis_validity`
- `run_summary`
- full `token_usage` breakdown
- prompt metadata
- memory statistics

Rationale:

- The saved artifact payload is much richer than the `AnalysisRecord` projection used by current consumers.
- Stage 1 should not collapse artifact blobs into DB columns without a clear query need.

## Latest Analysis Lookup Contract

This is the highest-risk migration contract in the system.

### Current state

- Primary lookup path: `load_latest_analyses(results_dir)`
- Data source: reconciler cache index when valid, otherwise scan `*_analysis.json`
- Returned shape: `dict[yf_ticker -> AnalysisRecord]`

### Future state

- Primary lookup contract: latest analysis for ticker `X`
- Canonical query semantics:
  - filter by ticker
  - order by `analysis_date desc, created_at desc`
  - optionally filter by `is_quick_mode`
  - return an `AnalysisRecord`-equivalent projection

Current and future users of this lookup:

- reconciler
- portfolio manager
- dashboard drilldown and snapshot paths
- refresh/freshness decisions

## Path Policy

- `results/` remains the artifact store.
- Durable artifact references should be stored relative to `RESULTS_DIR`.
- Runtime DB path should live under `RUNTIME_DIR`.
- Current stored path strings are inconsistent and may be relative or absolute depending on caller/config.
- Stage 1 must normalize durable path storage, not preserve that inconsistency.

## Migration Inputs vs Compatibility Artifacts

### Migration inputs

- `results/*_analysis.json`
- `.{results_dir}.latest_analyses_index.json`
- `runtime/ibkr_dashboard/jobs.sqlite`
- `results/.pipeline_last_run.json`

### Compatibility artifacts and behaviors to preserve during cutover

- `.{results_dir}.latest_analyses_index.lock`
- current `output_path` in dashboard ticker rows
- current `.pipeline_last_run.json` read path in `src.ibkr.screening_freshness.py`
- current `load_latest_analyses()` behavior surface for portfolio/recommendation consumers

## Future Producer and Consumer Impact

### Primary future writers

- `src.main.py`
  - writes analysis JSONs now
  - future `analyses` writer
- `scripts/run_pipeline.sh`
  - writes screening freshness marker now
  - future `pipeline_runs` and `pipeline_run_tickers` writer through a storage seam
- `src.web.ibkr_dashboard.worker.py`
  - persists `output_path` now
  - future `refresh_job_tickers.analysis_id` updater
- `src.token_tracker.py`
  - in-memory only now
  - future `run_metrics` writer

### Primary future readers/adapters

- `src.ibkr.reconciler.py`
  - owns cache/index logic now
  - future storage-backed latest-analysis reader
- `src.web.ibkr_dashboard.job_store.py`
  - owns standalone SQLite queue now
  - future adapter over shared storage
- `src.ibkr.screening_freshness.py`
  - marker-only reader now
  - future run-state reader with compatibility fallback
- `src.ibkr.recommendation_service.py`
  - future storage-backed analysis reader consumer

## Risks

### What changes now

Stage 0 changes:

- documentation
- design constraints
- ownership clarity
- migration safety

Stage 0 does not change:

- CLI behavior
- dashboard behavior
- portfolio recommendations
- current saved output format
- current filesystem layout

### Main risks if the model stays vague

- Stage 1 invents the wrong path normalization policy.
- Stage 1 collapses artifact blobs into DB columns without justification.
- Job status or ticker status values drift from current behavior.
- Latest-analysis lookup semantics change subtly and break reconciler/portfolio logic.
- Crash-recovery behavior for running jobs remains implicit.

### Risk level by area

- High risk
  - latest analysis lookup contract
  - path normalization policy
  - `analysis_id` vs `file_path` identity split
- Medium risk
  - job/ticker status compatibility
  - worker crash/recovery semantics
  - screening freshness cutover
- Low risk
  - notification event taxonomy
  - run-metrics ownership notes

## Test Plan for Stage 1

Stage 0 is docs-only. It adds no runtime tests.

Stage 1 must add tests for:

- latest analysis parity
  - storage-backed lookup matches current `load_latest_analyses()` semantics
- path normalization
  - mixed relative/absolute current path strings migrate safely to `RESULTS_DIR`-relative durable paths
- job store parity
  - enqueue/claim/update/complete behavior remains unchanged when storage moves
- worker compatibility
  - dashboard worker still reports successful and partial jobs with a stable per-ticker link target
- screening freshness compatibility
  - `.pipeline_last_run.json` remains readable during cutover
- analysis field semantics
  - local-currency price fields and `fx_rate_to_usd` survive projection/migration unchanged
- migration idempotence
  - importing existing artifacts and old job-store state can be rerun safely
- running-job recovery seam
  - Stage 1 preserves current behavior and exposes enough state for later timeout/cleanup logic

Existing tests that should become parity gates:

- `tests/ibkr/test_reconciler.py`
- `tests/web/test_job_store.py`
- `tests/web/test_worker.py`
- `tests/test_main_cli.py`
- `tests/ibkr/test_screening_freshness.py`

## Acceptance Checklist

Stage 0 is not complete unless this document answers:

- what are all current runtime state sources?
- what is the current reconciler index baseline version?
- what are the frozen status enums?
- what is the durable identity for an analysis?
- what is the lookup key and ordering for the latest analysis of ticker `X`?
- which fields are indexed metadata vs artifact-only payload?
- which artifacts are migration inputs vs compatibility-only artifacts?
- which module will eventually write each entity?
- which module will eventually read each entity?

## Assumptions and Defaults

- Stage 0 is docs-only.
- Current supported reconciler cache baseline is `version = 2`.
- Non-v2 cache payloads may be treated as invalid and rebuilt from file scan rather than explicitly migrated.
- Current path strings are not normalized consistently; future durable storage will normalize them relative to configured roots.
- `analysis_id` is the future durable analysis identity.
- `file_path` and current dashboard `output_path` are transitional locators, not durable identities.
- `AnalysisRecord` and `ReconciliationItem` remain the public in-process models.
- Eval `run_id` is not bridged to future pipeline `run_id` in Stage 0.
