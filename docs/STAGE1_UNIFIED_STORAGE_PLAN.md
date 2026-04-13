# Stage 1 Plan: Unified SQLite Storage That Would Pass Senior Review

## Summary

Stage 1 replaces the current split runtime-state model with one shared SQLite control-plane store while preserving current behavior for reconciler consumers, dashboard refresh jobs, and saved analysis artifacts.

This stage does not rewrite the pipeline, change CLI semantics, or move artifact blobs into the database. It does four things:

1. adds a new `src/storage/` package with one DB connection and schema owner
2. adds durable `analyses`, `refresh_jobs`, `refresh_job_tickers`, `pipeline_runs`, `pipeline_run_tickers`, `notifications`, and `run_metrics` tables
3. migrates current analysis and dashboard job state into the unified DB
4. cuts the reconciler and dashboard job store over to that DB without misleading compatibility layers

The highest-risk compatibility surface is latest-analysis lookup. Stage 1 must preserve the current caller-facing behavior while moving the implementation to explicit artifact, repository, and persistence-service abstractions.

## Engineering Standard

### Design rules for Stage 1

- No caller outside `src/storage/` writes SQL directly.
- DB setup, schema, migrations, row mapping, and path normalization are centralized.
- New modules stay focused and readable. Do not create a second `reconciler.py` under `src/storage/`.
- Shared parsing logic is extracted once and reused, not reimplemented.
- Every migration/import path is idempotent.
- Error handling degrades safely:
  - artifact save failure is fatal to persistence
  - metadata upsert failure after artifact write is explicit and logged
  - bad legacy rows/files are skipped and reported, not allowed to crash the whole migration

### New ownership model

Stage 1 should introduce three explicit seams:

- `AnalysisArtifactStore`
  - owns saving/loading artifact files under `results/`
- `AnalysisRepository`
  - owns DB-backed analysis metadata
- `RefreshJobRepository`
  - owns DB-backed dashboard refresh jobs

Recommended orchestration seam:

- `AnalysisPersistenceService`
  - coordinates "write artifact + upsert metadata" as one operation

This avoids pretending that a function only "saves results to file" once it really persists a compound analysis record across storage layers.

## Step 1: Establish storage package and schema bootstrap

### What this step does

Create one canonical runtime DB at `runtime/system.db` with migration versioning and normalized path helpers.

### Files

Create:

- `src/storage/db.py`
- `src/storage/schema.py`
- `src/storage/pathing.py`
- `src/storage/models.py`
- `src/storage/errors.py`
- `src/storage/analyses.py`
- `src/storage/refresh_jobs.py`
- `src/storage/migration.py`
- `src/storage/__init__.py`

### Responsibilities

- `db.py`: connection factory and DB bootstrap
- `schema.py`: numbered migrations and DDL
- `pathing.py`: path normalization and results-root handling
- `models.py`: small row/result dataclasses
- `errors.py`: storage exceptions
- `analyses.py`: metadata repository and persistence service
- `refresh_jobs.py`: dashboard queue repository
- `migration.py`: idempotent import from legacy state

### Schema

Create all seven canonical tables now:

- `analyses`
- `refresh_jobs`
- `refresh_job_tickers`
- `pipeline_runs`
- `pipeline_run_tickers`
- `notifications`
- `run_metrics`

Only `analyses`, `refresh_jobs`, and `refresh_job_tickers` are active in Stage 1. The others are schema stubs for later stages.

### Key schema rules

- add `results_root` to `analyses`
- add `analysis_id` to `refresh_job_tickers`
- keep `output_path` transitional in `refresh_job_tickers`
- do not make `(ticker, analysis_date, is_quick_mode)` unique
- make `(results_root, file_path)` unique instead

### Public interfaces

```python
# src/storage/db.py
def get_db(path: Path | None = None) -> sqlite3.Connection: ...
def initialize_db(path: Path | None = None) -> None: ...

# src/storage/pathing.py
def normalize_results_root(results_dir: Path) -> str: ...
def normalize_results_relative_path(
    raw_path: str | Path | None,
    *,
    results_dir: Path,
) -> str | None: ...
```

### Sample code

```python
def normalize_results_root(results_dir: Path) -> str:
    return str(results_dir.resolve())


def get_db(path: Path | None = None) -> sqlite3.Connection:
    db_path = (path or (Path("runtime") / "system.db")).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn
```

### Impact

- adds one DB seam
- removes future need to embed migration logic into job store or reconciler
- keeps storage concerns out of callers

### Complexity

- Medium

### Risk

- Low

### Done correctly when

- empty DB bootstraps cleanly
- repeated initialization is idempotent
- all tables exist, including Stage 4 stubs
- path normalization is deterministic

### Tests

Add:

- `tests/storage/test_schema.py`
- `tests/storage/test_pathing.py`

Cover:

- first-time bootstrap
- repeat bootstrap
- schema version progression
- relative path normalization
- absolute path under root
- absolute path outside root
- bad/empty path inputs
- normalized root consistency for relative vs absolute caller input

## Step 2: Extract shared analysis parsing out of reconciler

### What this step does

Move saved-analysis parsing logic out of `reconciler.py` into a shared module so storage and reconciler use the same rules.

### Files

Create:

- `src/ibkr/analysis_io.py`

Update:

- `src/ibkr/reconciler.py`

### Responsibilities

Move these concerns into `analysis_io.py`:

- build `AnalysisRecord` from saved payload
- build `AnalysisRecord` from file
- derive `created_at`
- derive deterministic `analysis_id`

`reconciler.py` should stop owning analysis-artifact parsing.

### Exact interfaces

```python
def build_analysis_record_from_data(filepath: Path, data: dict[str, Any]) -> AnalysisRecord | None: ...
def build_analysis_record_from_file(filepath: Path) -> AnalysisRecord | None: ...
def derive_analysis_created_at(filepath: Path, data: dict[str, Any]) -> str: ...
def derive_analysis_id(*, results_root: str, relative_file_path: str, record: AnalysisRecord) -> str: ...
```

### Exact `analysis_id` rule

Use UUIDv5 over:

- normalized `results_root`
- normalized relative `file_path`
- `ticker`
- `analysis_date`
- `is_quick_mode`

That is deterministic and explicit.

### Impact

- reduces duplication
- reduces risk of storage/reconciler drift
- shrinks reconciler responsibilities

### Complexity

- Medium

### Risk

- Medium

### Done correctly when

- both reconciler and storage use the same parsing path
- no analysis metadata extraction logic is duplicated
- same-day reruns produce distinct ids when file path differs

### Tests

Add:

- `tests/ibkr/test_analysis_io.py`

Cover:

- current artifact shape
- old artifact shape using fallback parsing
- missing optional fields
- bad payloads
- deterministic identity
- same-day distinct analyses

## Step 3: Introduce explicit artifact persistence and metadata persistence

### What this step does

Replace the current file-only save helper with explicit artifact and metadata persistence classes.

### Files

Create or add to:

- `src/storage/analyses.py`

Update:

- `src/main.py`
- `src/web/ibkr_dashboard/worker.py`

### New classes

```python
@dataclass(frozen=True)
class PersistedAnalysis:
    analysis_id: str
    artifact_path: Path
    results_root: Path
    created_at: str
    ticker: str
    analysis_date: str
    is_quick_mode: bool


class AnalysisArtifactStore:
    def write_artifact(
        self,
        result: dict[str, Any],
        *,
        ticker: str,
        quick_mode: bool,
        results_dir: Path,
    ) -> Path: ...


class AnalysisRepository:
    def upsert_from_artifact(
        self,
        artifact_path: Path,
        *,
        results_dir: Path,
        run_id: str | None = None,
    ) -> str: ...

    def get_latest(
        self,
        ticker: str,
        *,
        results_dir: Path,
        quick_mode: bool | None = None,
    ) -> AnalysisRecord | None: ...

    def load_latest_map(
        self,
        *,
        results_dir: Path,
        quick_mode: bool | None = None,
    ) -> dict[str, AnalysisRecord]: ...

    def get_by_artifact_path(
        self,
        *,
        results_dir: Path,
        artifact_path: Path,
    ) -> dict[str, Any] | None: ...


class AnalysisPersistenceService:
    def persist_analysis(...) -> PersistedAnalysis: ...
```

### Refactor in `main.py`

Replace the current save path with:

- artifact serialization preparation
- artifact write via `AnalysisArtifactStore`
- metadata upsert via `AnalysisPersistenceService`

The code should read as "persist analysis", not "save results to file".

### Refactor in `worker.py`

Worker should call `persist_analysis(...)` and receive:

- `analysis_id`
- `artifact_path`

Then update the job ticker row with both fields.

### Important rule

Do not keep a fake compatibility layer where the worker still conceptually thinks it is getting "just a file path." That would obscure the new architecture.

### Impact

- makes persistence behavior explicit
- gives worker direct access to `analysis_id`
- aligns names and responsibilities with the actual storage mechanism

### Complexity

- High

### Risk

- High

### Done correctly when

- `main.py` no longer uses misleading file-only persistence naming
- worker does not need a second lookup to recover `analysis_id`
- artifact and metadata writes are clearly coordinated in one place

### Tests

Add:

- `tests/storage/test_analyses.py`

Update:

- `tests/test_main_cli.py`
- `tests/web/test_worker.py`

Cover:

- successful persist returns `PersistedAnalysis`
- artifact write + DB upsert both happen
- DB failure after artifact write is surfaced clearly and handled intentionally
- same-day reruns remain distinct
- worker stores `analysis_id` and `output_path`

## Step 4: Replace standalone dashboard job DB with repository-backed queue

### What this step does

Replace the standalone dashboard SQLite store with a repository backed by `runtime/system.db`, but keep `RefreshJobStore` only as a thin public façade because that name still matches its actual role.

### Files

Update:

- `src/web/ibkr_dashboard/job_store.py`

Create:

- `src/storage/refresh_jobs.py`

### Repository methods

```python
class RefreshJobRepository:
    def enqueue(self, request: RefreshJobRequest) -> str: ...
    def list_jobs(self, *, limit: int = 50) -> list[dict[str, Any]]: ...
    def get_job(self, job_id: str) -> dict[str, Any] | None: ...
    def claim_next(self) -> QueuedRefreshJob | None: ...
    def update_ticker_status(
        self,
        job_id: str,
        ticker: str,
        status: TickerStatus,
        *,
        error: str | None = None,
        output_path: str | None = None,
        analysis_id: str | None = None,
    ) -> None: ...
    def complete_job(...) -> None: ...
```

### Rules

- preserve job and ticker status enums
- preserve current claim/update/complete semantics
- preserve `output_path`
- add `analysis_id`
- keep crash-left-`running` behavior unchanged in Stage 1

### Impact

- one queue store instead of a separate DB
- no public dashboard API change
- prepares dashboard paths for `analysis_id`

### Complexity

- Medium

### Risk

- Medium

### Done correctly when

- dashboard code no longer owns raw schema DDL
- tests still pass with current caller expectations
- queue semantics are unchanged

### Tests

Add:

- `tests/storage/test_refresh_jobs.py`

Update:

- `tests/web/test_job_store.py`
- `tests/web/test_worker.py`

Cover:

- enqueue/list/get parity
- claim race safety
- partial job completion
- no-op updates for missing rows
- legacy job migration
- preservation of `running` jobs after interruption

## Step 5: Replace reconciler’s JSON-index path with repository-backed lookup

### What this step does

Make DB-backed lookup the normal path for latest-analysis lookup while keeping only those compatibility entrypoints whose names still accurately describe the operation.

### Files

Update:

- `src/ibkr/reconciler.py`
- `src/ibkr/recommendation_service.py`

### Design

Recommended approach:

- keep `load_latest_analyses(results_dir, *, progress=None)` temporarily because multiple callers use it and the name still accurately describes the caller-facing operation
- move its implementation to `AnalysisRepository.load_latest_map(...)`
- keep a clearly marked private fallback scan path
- remove JSON-index incremental update semantics as canonical behavior

### Important reviewer-facing rule

The code should make it obvious that:

- DB-backed lookup is canonical
- file scan is fallback
- JSON index is deprecated implementation residue, not active architecture

### Progress callback

If fallback scan is used, pass `progress` through exactly.

### Impact

- reconciler stops relying on sibling JSON index as the normal durable source
- portfolio and recommendation callers keep the same function contract
- save path becomes explicit artifact + metadata persistence

### Complexity

- High

### Risk

- High

### Done correctly when

- callers still get the same result shape
- repository lookup is the normal path
- fallback behavior is clear and not misleading
- recommendation service does not need broad redesign

### Tests

Update:

- `tests/ibkr/test_reconciler.py`

Cover:

- repository-backed latest lookup parity
- fallback scan parity
- progress callback preserved
- same-day rerun ordering
- missing/corrupt artifacts
- mixed ticker normalization cases

## Step 6: Add migration entrypoint and document the new canonical architecture

### What this step does

Provide a single idempotent migration entrypoint and document what is canonical after Stage 1.

### Files

Create/update:

- `src/storage/migration.py`
- `docs/RUNTIME_MODEL.md`
- `docs/ROADMAP_OPERATOR_WORKBENCH.md`
- `docs/CODEBASE_MEMORY.md`

### Migration design

Expose:

```python
@dataclass(frozen=True)
class MigrationSummary:
    analyses_imported: int
    analyses_updated: int
    analysis_errors: int
    jobs_imported: int
    job_tickers_imported: int
    job_errors: int
```

```python
def migrate_legacy_runtime_state(
    *,
    results_dir: Path,
    legacy_jobs_db: Path | None = None,
) -> MigrationSummary: ...
```

### Inputs

- `results/*_analysis.json`
- `runtime/ibkr_dashboard/jobs.sqlite`

### Non-inputs

- `.latest_analyses_index.lock`
- JSON index file as canonical truth
- `.pipeline_last_run.json` for active Stage 1 functionality

### Explicit non-change

`src/ibkr/screening_freshness.py` remains unchanged in Stage 1. That should be stated directly in code comments or docs to avoid implementer confusion.

### Impact

- makes cutover operationally safe
- prevents later contributors from treating deprecated JSON/index state as canonical

### Complexity

- Medium

### Risk

- Medium

### Done correctly when

- migration is idempotent
- docs clearly identify canonical DB-backed state
- docs do not describe deprecated wrappers as normal architecture

### Tests

Add:

- `tests/storage/test_migration.py`

Cover:

- analyses-only import
- jobs-only import
- rerun idempotence
- bad artifact file
- malformed legacy jobs row
- empty inputs
- preexisting imported rows

## Naming, Style, and Reviewer Expectations

### Naming conventions

Use names that match current repo style:

- `Repository` for DB-facing read/write components
- `Store` for artifact or file-backed persistence
- `Service` for orchestration across repositories/stores
- `models.py` for typed dataclasses / light DTOs

Avoid generic `manager` classes unless they truly coordinate multiple subsystems.

### Duplication policy

Consolidate:

- analysis parsing
- path normalization
- DB connection/bootstrap
- timestamp/id derivation
- migration row mapping

Do not consolidate unrelated concerns into a single `utils` file.

### Error handling expectations

Every storage step must define:

- what is retried
- what is skipped
- what is logged
- what is fatal
- what degrades safely

At minimum:

- artifact write failure is fatal for persist
- metadata upsert failure after artifact write is explicit and logged
- migration continues after per-file/per-row failures and reports summary counts
- lookup never corrupts state; fallback may be used where explicitly designed

## Test Plan

Run at minimum:

- `tests/storage/test_schema.py`
- `tests/storage/test_pathing.py`
- `tests/storage/test_analyses.py`
- `tests/storage/test_refresh_jobs.py`
- `tests/storage/test_migration.py`
- `tests/ibkr/test_analysis_io.py`
- `tests/ibkr/test_reconciler.py`
- `tests/web/test_job_store.py`
- `tests/web/test_worker.py`
- `tests/test_main_cli.py`
- `tests/ibkr/test_screening_freshness.py`

Required edge-case coverage:

- same-day same-ticker reruns
- corrupt JSON artifacts
- missing optional snapshot fields
- mixed absolute/relative paths
- worker partial success
- legacy DB migration rerun
- DB bootstrap on existing partial schema
- fallback scan path
- race on `claim_next()`
- artifact write succeeds but metadata path fails
- missing results directory
- non-v2 legacy index treated as invalid without special migration

## Assumptions and Defaults

- Stage 1 is free to rename/refactor misleading persistence interfaces.
- `save_results_to_file()` should be replaced by a clearer persistence API, not stretched to fit the new architecture.
- `RefreshJobStore` may remain only if its meaning stays accurate.
- `load_latest_analyses()` may remain temporarily because it still describes the caller-facing operation, but its implementation should clearly delegate to repository-backed lookup.
- SQLite remains the only backend in Stage 1.
- `runtime/system.db` is the canonical DB.
- `results/` remains the canonical artifact store.
- Stage 1 creates stub tables for later stages but does not write to them.
- `screening_freshness.py` stays marker-backed in Stage 1.

## Summary

This Stage 1 plan intentionally favors clarity over superficial backward compatibility:

- replace misleading file-only persistence naming with explicit artifact/repository/service abstractions
- extract shared analysis parsing to a real module
- keep compatibility only where it remains semantically truthful
- centralize DB, pathing, parsing, identity, and migration logic
- require edge-case and failure-path tests at every step

That is the version most likely to read as deliberate senior-level engineering rather than an over-cautious compatibility patch series.
