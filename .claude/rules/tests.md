---
paths:
  - "tests/**"
---

# Writing tests here

## Markers and scope

Mark anything over five seconds `slow` and anything needing the network
`integration`. Mock model responses in unit tests. Run the directory that owns your
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
