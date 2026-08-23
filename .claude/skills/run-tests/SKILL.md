---
name: run-tests
description: >
  Choose and run the right tests in this repository. Use before running pytest, while
  iterating on a failure, or when deciding whether a change needs the full suite.
  Carries this suite's directory ownership, timings, and Make targets.
---

# Running tests here

The full suite is roughly **9,100 tests across 412 files and about five minutes**.
That is the big gun: run it once at wrap-up, not while iterating. Iterating on the
owning subset takes seconds.

## Which directory owns the change

| Directory | Files | Owns | Rough runtime |
|---|---|---|---|
| `tests/agents/` | 56 | node factories, DATA_BLOCK sanitization, runtime and timeouts, consultant and auditor loops | ~30 s |
| `tests/ibkr/` | 48 | positions, reconciliation, disposition, watchlist, report rendering | ~40 s |
| `tests/financial/` | 38 | fetcher, merge policy, FX and currency, liquidity | ~90 s |
| `tests/` root | 71 | persistence, main CLI, provenance, retrospective-adjacent | ~40 s |
| `tests/advanced/` | 33 | retrospective, editor, article writer, cost rollups | ~10 s |
| `tests/tooling/` | 19 | hooks, content inspection | ~5 s |
| `tests/validators/` | 16 | red flags, supplemental flags | ~5 s |
| `tests/llm_runtime/` | 16 | seats, bindings, profiles | ~5 s |
| `tests/eval/` | 13 | capture contracts | ~5 s |
| `tests/prompts/` | 12 | prompt parity and contract round-trip | ~3 s |
| `tests/memory/` | 12 | ChromaDB isolation — **slowest per test** | ~60 s |
| `tests/charts/`, `tests/reports/`, `tests/web/`, `tests/config/`, `tests/mcp/` | 6–12 each | as named | ~5–10 s |

Practical loop: touch code, run its directory, run `poetry run ruff check src/`,
repeat. Then once, before reporting: `poetry run ruff format --check src/ tests/`,
`poetry run mypy src/`, the full suite, and `make test-prompts` if a prompt moved.
Every direct tool invocation goes through `poetry run`; the `make` targets already do.

## Run the full suite earlier when the change is cross-cutting

A changed prompt, or anything under `src/llm_runtime/` or `src/config.py`, is
cross-cutting by nature — the narrow run tells you almost nothing there.

**A green directory is not a green suite.** A change can pass its own directory while
breaking fixtures elsewhere that encoded the old behaviour.

## Make targets

| Target | What it does |
|---|---|
| `make test-ci` | deterministic tests, no network or provider credentials |
| `make test-prompts` | L0 static parity plus L1 contract round-trip, no LLM |
| `make replay` | L2 deterministic replay over frozen fixtures, no LLM |
| `make eval-semantic` | L3 semantic judge — **real LLM cost**, manual or nightly |
| `make check-all` | format check, lint, type check |
| `make pre-commit` | `check-all` plus `test-ci` |

`make replay` reads `tests/fixtures/frozen/`, and `make eval-semantic` reads the suite
manifests in `evals/prompt_check_suites/` — both tracked. Capture bundles under
`evals/captures/` are operator-local: they appear once you run a capture, and a fresh
clone has none.

## Diagnosing a run that looks stuck

Stream output to a file. **Never pipe a long run through `tail`** — it emits nothing
until the run ends, so an empty file is indistinguishable from a hang.

`tests/memory/` is a multi-minute serial block *by design*: the memory healthcheck
joins a daemon thread with a 15 s timeout and embedding calls are bounded at 30 s, so
the slowest cases sit at 10–15 s each and show ~0% CPU while waiting. Before
concluding deadlock, re-read the last test name printed; if it moved, the run is slow,
not hung.

Two environment notes that cost real time here: an unwritable Matplotlib cache
re-scans system fonts in every process, and a type-checker cache in a cloud-synced
directory produces an internal error with a sqlite disk I/O failure that has nothing
to do with the type check. Point `MPLCONFIGDIR` and `MYPY_CACHE_DIR` at a temp
location before a batch.

## Writing a test

Mark anything over five seconds `slow`, and anything needing the network
`integration`. Verify a new guard by planting the defect it exists to catch and
watching it fail — a guard that only passes on clean code proves nothing. Prefer an
AST scan to a text scan when the guard's subject is a code pattern, because a comment
explaining a retired pattern necessarily contains that pattern.

Full history: `docs/CODEBASE_MEMORY.md`.
