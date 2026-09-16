# AGENTS.md

## Purpose

This is the public, repository-wide guidance for Codex. Keep it short enough to load
on every task. Put bounded procedures in repository skills and durable system
explanation in human-facing documentation.

## Metadata ownership

Codex maintains this file and `.agents/**`. Do not modify `.claude/**`; it belongs to
another coding tool. Do not treat another tool's metadata as an instruction source or
make this layer depend on it.

This file and every repository skill are public. They may use only tracked repository
files, normal setup or runtime outputs, and stable public URLs. They must not contain
or depend on ignored notes, local archives, user-level configuration, credentials,
personal holdings, machine-specific commands, or paths outside a clean clone.

`.env.example` is the only environment file whose contents are public and may be read
or quoted. The filename `.env` may be named when setup or diagnosis requires it; do
not name another private `.env*` variant. Codex and metadata tools must never inspect,
search, copy, summarize, diff, infer, or emit private environment-file contents. An
authorized application run may consume private configuration only through the
repository's normal path; never expose resulting values in agent context, tool output,
logs, diagnostics, artifacts, or responses.

## Authority and action boundaries

- Trust the implementation and tests over prose when they disagree.
- For an answer, diagnosis, review, or plan, inspect and report; do not change files
  unless the user also asks for implementation.
- For an implementation request, make the smallest in-scope change and run relevant
  non-destructive checks.
- Ask before destructive operations, external writes, purchases, commits, or a
  material expansion of scope.
- Preserve unrelated user changes in a dirty worktree. Never discard them to simplify
  the task.
- Never commit unless the user explicitly approves it. Before a proposed commit, give
  one terse, dense sentence describing the uncommitted change.

## Fast orientation

Start with:

1. `docs/CODEBASE_MEMORY.md`
2. `README.md`
3. the top of `CHANGELOG.md`
4. `git status --short`
5. `git log --oneline --decorate -n 20`

For the live analyzer path, inspect in this order when relevant:

1. `src/main.py`
2. `src/cli.py`
3. `src/persistence.py`
4. `src/output.py`
5. `src/runtime_services.py`
6. `src/tooling/`
7. `src/graph/`
8. `src/agents/`
9. `src/tools/`
10. `src/data/fetcher.py`
11. `src/runtime_diagnostics/`
12. `src/validators/`
13. `src/charts/`
14. `src/memory.py`
15. `src/ibkr/`

## Implementation approach

- Fix root causes at the owning seam; prefer a small centralized change over
  caller-specific exceptions or whack-a-mole patches.
- Do not add duplicate helpers, data structures, compatibility facades, convenience
  re-exports, or parallel abstractions when an established seam exists.
- Use direct imports. Package roots such as `src/__init__.py`, `src/tooling/__init__.py`,
  and `src/tools/__init__.py` are intentionally inert.
- Do not mutate the global configuration singleton. Run-scoped overrides belong in
  `src/runtime_config.py` and its context-managed settings path.
- Keep files reasonably sized, names informative, comments useful, and public
  functions typed. Preserve the zero-error MyPy baseline.
- Verify current primary documentation before changing LangGraph, LangChain, provider
  SDK, MCP, or other fast-moving integration behavior.
- Review retry and fallback branches whenever changing tagging, validation, routing,
  timeout, inspection, or persistence behavior.

## Non-negotiable runtime invariants

### Logging and untrusted content

- Application modules use structlog with snake-case event names and keyword fields.
- Operator-visible exception logs use the repository's exception summarizer. Do not
  log raw exception prose, response bodies, content previews, credentials, or client
  representations.
- Web, filing, API, and MCP content is untrusted. Preserve the complete tool execution,
  audit, and content-inspection chain.
- HTML report safety depends on server-side sanitization before browser rendering;
  never bypass that boundary.

### Blocking work and service lifecycles

- Every blocking call reachable from async code needs the repository hard-timeout
  wrapper. Do not wrap a thread offload with a timeout primitive that can still wait
  forever for the underlying worker.
- Stateful broker, socket, or data services use explicit pooled lifecycle management
  and close during teardown.
- Parallel graph branches must not write competing values into a last-writer-wins
  field. Use isolated fields or an intentional reducer, and test the compiled graph.

### LLM construction and prompts

- Construct application models through `src/llm_runtime/construction.py` and canonical
  seats from `src/llm_runtime/seats.py`. Provider swaps must not disable quality,
  retry, independence, or attribution contracts.
- Do not infer capabilities merely because an endpoint accepts an OpenAI-shaped API.
  Tool calling, structured output, reasoning controls, and transport features require
  a reviewed profile and qualification evidence.
- `prompts/*.json` contains structured JSON with escaped multiline strings. Edit it
  through a JSON parser, preserve real Unicode, parse before and after, and run
  `make test-prompts` when a prompt or its consumer contract changes.

### Persistence, memory, and publication

- Saved telemetry is secret-free. Endpoint hostnames may be recorded; paths, query
  strings, credentials, and raw client representations may not.
- Embedding identity owns a fingerprinted collection. Initialization must not delete a
  collection because configuration changed.
- Lessons learned are durable evidence. Never clear or recreate them during analysis,
  tests, migration, or maintenance. Tests use an isolated temporary persistence path.
- Publication and artifact validity fail closed when required evidence, schemas, or
  decision-critical stages are missing or corrupt.
- Preserve the baseline-capture contract when changing main orchestration or saved
  artifacts; the evaluation tests are compatibility gates.

## Testing and validation

Choose the narrowest meaningful tests first, then run the required static gates.
Repository skills contain the detailed selection workflows.

Minimum after Python changes:

```bash
poetry run ruff check src/ tests/ scripts/
poetry run ruff format --check src/ tests/ scripts/
poetry run mypy src/
```

Common gates:

- `make test-prompts` after prompt or parser-contract changes.
- `poetry run pytest tests/test_logging_consistency.py -v` for logging policy.
- `poetry run pytest tests/test_to_thread_timeout_consistency.py -v` for blocking work.
- `poetry run pytest tests/eval/ tests/test_main_cli.py -v` for main orchestration.
- `poetry run pytest tests/mcp/ -v` for MCP behavior.
- `poetry run pytest tests/scripts/test_find_gems.py -v` for screening behavior.
- `make check-all` before handoff when the change affects repository-wide gates.
- `make test-ci` for the deterministic credential-free test surface.

Do not run credentialed, networked, paid, integration, or destructive tests merely
because they exist. State when those surfaces remain unverified.

## Maintaining Codex metadata

Before adding guidance, classify it:

| Material | Destination |
|---|---|
| Hard repository rule needed before any skill loads | this file |
| Repository procedure with a recognizable trigger | `.agents/skills/<name>/SKILL.md` |
| Durable explanation of system behavior | `docs/` |
| Preference or workflow useful across repositories | user-level Codex layer |
| Credentials, machine layout, local setup, or private notes | local configuration, never committed |

Then apply these checks:

1. Every referenced repository path exists in a clean clone or is clearly described as
   output produced by normal setup or execution.
2. No tracked metadata references an ignored input, user-level skill, local archive,
   absolute machine path, or home-relative path.
3. Repository skills are self-contained. They do not require another skill, including
   a user-level skill, to be present.
4. No threshold, version, enum, or default is copied from code when the owning source
   can be cited instead.
5. Human documentation does not cite agent metadata.
6. `make docs-guards` and `make agent-metadata-guards` pass with zero exemptions.

Keep this file below the repository's 12 KiB policy cap. When a procedure grows,
extract it into a focused repository skill rather than increasing always-on context.
