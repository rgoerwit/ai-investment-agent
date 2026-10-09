---
name: run-tests
description: Select and run this repository's relevant static checks and tests after a change or when diagnosing CI. Use for test selection, validation, prompt contracts, or failing quality gates in this repository.
---

# Run repository checks

Base selection on the actual diff and the owning modules. Do not assume a user-level
skill or local configuration exists.

1. Inspect `git status --short` and the relevant diff.
2. Run the narrowest tests that exercise the changed behavior, including a happy path,
   an edge case, and an error or fallback path where applicable.
3. Run formatting and linting for touched Python files. Before handoff, run the
   repository-wide static gates when the change affects shared infrastructure.
4. Add the specialized contract gate when the touched surface requires it.
5. Report exact commands, outcomes, and any credentialed or integration surface that
   remains unverified.

## Required mappings

- Prompt JSON, prompt parsers, labels, enums, block markers, or verdict headers:
  `make test-prompts`.
- Main orchestration, persistence, or baseline capture:
  `poetry run pytest tests/eval/ tests/test_main_cli.py -v`.
- Blocking work or async timeouts:
  `poetry run pytest tests/test_to_thread_timeout_consistency.py tests/agents/test_runtime_hard_timeout.py -v`.
- LLM budget resolution, transport admission or retry ownership:
  `poetry run pytest tests/llm_runtime/test_timeouts.py tests/llm_runtime/test_migrated_caller_budgets.py tests/llm_runtime/test_run_deadline.py tests/test_timeout_literals.py -v`.
- Tool/inspection scope or MCP deadline changes:
  `poetry run pytest tests/tooling/test_tool_timeout_scopes.py tests/mcp/test_mcp_timeouts.py -v`.
- Blocking worker lifecycle or late-result handling:
  `poetry run pytest tests/test_blocking_io.py tests/test_tavily_utils.py tests/data/test_gap_fill.py -v`.
- Logging and exception safety:
  `poetry run pytest tests/test_logging_consistency.py -v`.
- MCP behavior: `poetry run pytest tests/mcp/ -v`.
- Screening behavior: `poetry run pytest tests/scripts/test_find_gems.py -v`.
- Codex metadata or public documentation: `make docs-guards` and
  `make agent-metadata-guards`.

Use `make check-all` for the complete static surface and `make test-ci` for the
deterministic credential-free suite. Do not launch paid, networked, integration, or
destructive tests without the request and prerequisites that authorize them.

Budget tests must exercise the selected adapter's effective request options, not
only mocked factory arguments or model fields. Cover quick/full mode, sync/async,
minimum-allowance rejection, tier downgrade, model failover, outer cancellation,
and concurrent calls sharing a model. Use fake clocks/transports for request
budgets and releasable stalled workers for caller bounds; test interpreter exit in
a subprocess with an external test timeout. Plant the defect a new guard claims
to reject, then prove the valid centralized-policy case passes. Polling, cooldowns
and protocol constants are not request deadlines.

## Restricted collection

If pytest collection specifically fails while accessing ancestor guidance, retry
the same tests with `--rootdir=tests --confcutdir=tests`. This retains
`tests/conftest.py` and its credential-free isolation. It does not authorize access
to private configuration or suppress failures inside tests. Record the exact
remaining limitation instead of treating every sandbox collection failure as an
unrun suite.

## Detect macOS collection stalls

A focused file should normally start promptly. The full suite can take much longer,
with continuously advancing collection; that alone is not a hang. If a focused probe is
repeatedly slow or collection stops advancing, let pytest's macOS preflight hydrate
public inputs automatically. If it cannot, immediately run
`poetry run python scripts/check_macos_cloud_files.py --hydrate`, then rerun the exact
test once. The checker excludes ignored local configuration and every `.env*` file
except `.env.example`, even if accidentally tracked.
