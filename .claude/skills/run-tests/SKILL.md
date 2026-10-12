---
name: run-tests
description: >
  Select and run this repository's checks after a change or while diagnosing test
  failures. Covers owning seams, contract gates, strict offline isolation and
  restricted collection; excludes live analysis batches.
---

# Run repository checks

Inspect the actual diff and owning modules. Run the narrowest meaningful checks
while iterating, then the required shared gates. Do not copy suite counts or timing
catalogs into this skill: they become stale as the tree grows.

## Select the owning checks

| Changed surface | Checks |
|---|---|
| Agent nodes, graph routing or output contracts | Owning `tests/agents/` files and compiled-graph tests |
| LLM dispatch, budget admission or retries | `tests/llm_runtime/test_timeouts.py`, `tests/llm_runtime/test_migrated_caller_budgets.py`, `tests/llm_runtime/test_run_deadline.py`, `tests/test_timeout_literals.py` |
| Tool/inspection or MCP scopes | `tests/tooling/test_tool_timeout_scopes.py`, `tests/mcp/test_mcp_timeouts.py` |
| Blocking work and late results | `tests/test_blocking_io.py`, `tests/test_to_thread_timeout_consistency.py`, `tests/agents/test_runtime_hard_timeout.py` |
| Memory probes or availability | `tests/memory/test_embedding_probe_admission.py`, `tests/memory/test_embedding_probe_offline.py` |
| Article workflow and writer fallbacks | `tests/advanced/test_article_workflow_deadline.py`, `tests/advanced/test_article_writer.py` |
| Main orchestration, persistence or baseline capture | `tests/eval/`, `tests/test_main_cli.py` |
| Logging policy | `tests/test_logging_consistency.py` |
| Screening behavior | `tests/scripts/test_find_gems.py` |
| Test isolation or fixtures | `tests/test_offline_boundary.py`, `tests/test_isolation_fixtures.py`, `tests/config/test_env_file_policy.py`, `tests/test_conftest_recollection.py` |

Use the actual adapter's effective request options for timeout tests, with fake
clocks and transports. Cover spent allowance, tier downgrade, failover, sync/async,
outer cancellation and concurrent calls sharing one model. Model fields alone do
not prove a request was sized correctly. Release and join releasable stalled workers
in `finally` before patch teardown; an orphan surviving the fake can reach a provider.

Plant the regression a new architectural guard claims to catch, prove it fails,
then restore the valid case. Prefer AST scans for code patterns and read the owning
constant or registry rather than a test-owned copy.

## Required gates

After Python changes:

```bash
poetry run ruff check src/ tests/ scripts/
poetry run ruff format --check src/ tests/ scripts/
poetry run mypy src/
```

| Target | Purpose |
|---|---|
| `make test-ci` | Deterministic credential-free suite; excludes slow, integration, LLM and API markers |
| `make test-prompts` | Static parity and contract round-trip after prompt or parser changes |
| `make replay` | Deterministic replay over tracked frozen fixtures |
| `make eval-semantic` | Paid semantic judge; requires authorization and prerequisites |
| `make check-all` | Format, lint, types, documentation and metadata gates |
| `make docs-guards` and `make claude-metadata-guards` | Claude metadata or public-documentation changes |
| `make pre-commit` | Shared static gates plus deterministic tests |

Broaden testing when the change crosses shared configuration, construction or
contracts, or when a focused result leaves a specific concern unresolved. A green
directory alone does not prove shared fixtures still work. Report exact commands,
outcomes and integration surfaces left unverified.

## Strict offline isolation

Every non-`integration` test is offline under `tests/conftest.py`. The autouse
boundary refuses sockets, DNS and libcurl, records intercepted network attempts and
fails teardown even when application code catches those refusals. Assert request
counts at faked SDK seams as well; SDK-method refusals do not enter that ledger.
Inject a fake at the data-source seam.
Only `integration` is skipped by default; `slow` is filtered by the command.
`--run-integration` opts in to live tests and needs authorization for external or
paid work. Do not launch provider probes as a diagnostic shortcut.

Keep configured addopts and plugins; never clear them with `-o addopts=""`.
pytest-socket and its `--allow-unix-socket` addopt are removed; the repository
boundary permits AF_UNIX and socketpair locally. `PYTEST_DISABLE_DOTENV` must have
the exact string `true`, as configured in `pyproject.toml`. Only `.env.example`
contents may be inspected by agents.

Dummy keys can construct real SDKs and trigger embedding health probes. Supply
`tests/helpers/fake_embeddings.py` when a unit test needs availability; keep the
probe and wire fakes hermetic. Redirect persistent stores to session temporary
paths and preserve root fixtures that reset shared clients, limiters and bindings.

Standalone audit probes must install isolation before application/provider imports:
set the dotenv switch to literal `true`, disable `LANGSMITH_TRACING`,
`LANGCHAIN_TRACING` and `LANGCHAIN_TRACING_V2`, and refuse network/DNS using
`tests/helpers/offline_network.py`. Prefer pytest's existing boundary; assert no
attempts and retain refusal until every worker has finished.

## Restricted collection and stalls

If rootdir collection fails on ancestor guidance outside the allowed surface, retry
the same selected tests with:

```bash
poetry run pytest <paths> -q --rootdir=tests --confcutdir=tests
```

This preserves `tests/conftest.py` and isolation. Report the actual retry outcome;
it does not authorize reading the denied file. Do not pipe a long run through
`tail`: it hides progress until exit.

For a confirmed macOS cloud-placeholder stall, let the public-input preflight hydrate
files. If it fails, run `poetry run python scripts/check_macos_cloud_files.py --hydrate`
and retry once. It excludes ignored local configuration and private environment files.
