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
- Logging and exception safety:
  `poetry run pytest tests/test_logging_consistency.py -v`.
- MCP behavior: `poetry run pytest tests/mcp/ -v`.
- Screening behavior: `poetry run pytest tests/scripts/test_find_gems.py -v`.
- Codex metadata or public documentation: `make docs-guards` and
  `make agent-metadata-guards`.

Use `make check-all` for the complete static surface and `make test-ci` for the
deterministic credential-free suite. Do not launch paid, networked, integration, or
destructive tests without the request and prerequisites that authorize them.

## Detect macOS collection stalls

A warm focused file should usually start and finish within about two seconds. The
roughly 10.3k-test full suite may take about 15 minutes, with several minutes of
continuously advancing collection; that alone is not a hang. If a focused probe is
repeatedly slow or collection stops advancing, let pytest's macOS preflight hydrate
public inputs automatically. If it cannot, immediately run
`poetry run python scripts/check_macos_cloud_files.py --hydrate`, then rerun the exact
test once. The checker excludes ignored local configuration and every `.env*` file
except `.env.example`, even if accidentally tracked.
