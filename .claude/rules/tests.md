---
paths:
  - "tests/**"
---

# Writing tests here

## Running under a sandbox

A bare `pytest` can abort during rootdir collection with a `PermissionError` on a
file outside the test tree that the sandbox denies. That is not a suite failure, and
the boundary is not to be bypassed. Keep collection off the repository root instead:
`poetry run pytest <paths> -q -p no:cacheprovider --rootdir=tests --confcutdir=tests`.
No conftest lives above `tests/`, so `tests/conftest.py` and its isolation still load.

**Keep the configured `addopts`; never pass `-o addopts=""`.** It carries
`--strict-markers`/`--strict-config`. (The old `--allow-unix-socket` and pytest-socket
are gone: the offline boundary below allows AF_UNIX and `socketpair` itself.)

## Subdirectory conftests and re-collection

**Do not delete the `pytest_make_collect_report` hook imported into
`tests/conftest.py`, and do not "fix" a `fixture ... not found` by moving the fixture
up.** pytest 9.1 binds a subdirectory conftest's fixtures to the first `Dir` object
it builds for that directory. A command that names a root-level test file between two
files of one subdirectory re-collects `tests/` and builds a second `Dir`. Without the
hook, tests under it lose that conftest: named fixtures error, and autouse isolation
is skipped silently. Whether either directory is a package plays no part.
`tests/test_conftest_recollection.py` guards it. Story: `docs/CODEBASE_MEMORY.md`.

## Markers and scope

Mark anything over five seconds `slow` and anything needing the network
`integration`; those are skipped unless you pass `--run-integration` (it also gates the
live SGX PDF canary). Mock model responses in unit tests. Run the directory that owns your
change while iterating; the full suite once at wrap-up.

## A guard must be proven against a real regression

**Verify a new guard by planting the defect it exists to catch and watching it fail.**
A guard that only passes on clean code proves nothing — several here were written,
passed immediately, and turned out to assert something weaker than intended.

Prefer an **AST scan to a text scan** when the subject is a code pattern: a comment
explaining a retired pattern necessarily contains that pattern, so a text scan
false-positives on its own documentation.

A test that keeps its **own copy** of the contract it guards does not guard it. The
copy drifts with the code and stays green. Read the real constant, the real registry,
the real prompt.

Hand-enumerated lists of spellings cannot catch an omission — that is structurally the
thing they are bad at. Scan for the shape instead.

## Isolation

Persistent stores must be redirected to a temporary directory for the session.
Anything that constructs its own client from configuration makes that configuration
the single chokepoint between a test and real data, so redirect there rather than
patching individual call sites. A cleanup helper called without a scope argument can
wipe everything.

Reset process-global mutable state between tests: capability caches, circuit breakers,
session singletons, and rate limiters all leak across tests otherwise, and the symptom
appears in an unrelated file.

Pin service tiers and model overrides for the session so an operator's environment
cannot flip construction paths under mock-based tests.

**Unit tests are offline, strictly.** The autouse `_offline_boundary` in
`tests/conftest.py` refuses sockets, DNS, libcurl and (except for `security` tests, which
drive the real SDK into wire fakes) the embedding SDKs, and **fails at teardown any
non-`integration` test that attempted the network**, naming targets and threads.
A failure means: inject a fake at the data source, or mark the test `integration`.
Never exempt a test from the boundary and do not widen the embedding exemption. To
assert a path sends nothing, or to attempt on purpose, call
`install_offline_network` from `tests/helpers/offline_network.py` in the test and
assert on its own list. Guarded by `tests/test_offline_boundary.py`.

**Unit tests never read the operator's `.env`.** Every `BaseSettings` subclass in
`src/` takes `env_file` from `src.env_file.default_settings_env_file()` (None under
`PYTEST_DISABLE_DOTENV`), and so does `_parse_env_file`, the reader behind
`get_env_value`. A new settings class that hard-codes `".env"` fails
`tests/config/test_env_file_policy.py`.

**Process-global waits and state are reset by root fixtures.** `_instant_process_limiters`
swaps every process-fallback limiter for `tests/helpers/instant_limiter.py` (a real one
paces ~5 s per acquire); a test of the real limiter takes
`@pytest.mark.real_rate_limiter`. `_reset_shared_memory_state` clears the class-level
embeddings and Chroma clients, so a memory test that needs `available=True` supplies
`tests/helpers/fake_embeddings.py` rather than relying on a leftover. Tests that drive
`FinancialSituationMemory`, `SmartMarketDataFetcher`, FX or `src.main` cut the Yahoo and
Tavily seams with `tests/helpers/offline_fetcher.py`, `tests/helpers/offline_fx.py` and the
`offline_main_flow` fixture. yfinance retries are 0 (`configure_yfinance_defaults`): the
application owns retries. Guarded by `tests/test_isolation_fixtures.py`.

## Subprocesses

Spawn so the runtime uses `posix_spawn`: pass `close_fds=False` and **never** `cwd=`,
which forces the fork path and can segfault in an atfork handler. Set the child's
import path through its environment instead. Prefer a subprocess to a spawning process
pool.

## Dates in fixtures

Golden rendered output needs **absolute** dates, or the goldens drift at midnight.
Anything gated on freshness needs a date **relative to now**, or it passes for a
fortnight and then fails on correct code. Choose by what the test is actually
asserting.

Full history: `docs/CODEBASE_MEMORY.md`.
