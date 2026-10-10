---
paths:
  - "src/**/*.py"
---

# Every blocking call needs a hard wall-clock bound

**Never leave an `asyncio.to_thread(...)` unbounded.** Two primitives, one canonical:

- **`run_with_hard_timeout(coro, *, timeout, label)`** from `src/async_utils.py` is
  canonical for `to_thread` of any blocking sync I/O. It is deadline-only: it raises
  `TimeoutError` when the timeout fires and **orphans** the task. That is what makes
  it work when the underlying C-level socket read cannot be cancelled.
- **`asyncio.wait_for(...)`** is acceptable *only* around a coroutine you wrote that
  honours cancellation cleanly.

**Never wrap `asyncio.to_thread(<blocking network call>)` in `asyncio.wait_for`.**
`wait_for` cancels the inner task and then *waits on it*; a thread parked in a socket
read cannot be cancelled, so it waits forever. This is the bug behind the historical
overnight hangs.

For a new blocking call site, prefer **`run_blocking_call(policy, fn)`** from
`src/blocking_io.py`, which co-locates the timeout policy with the call. Named
policies exist for the common cases; use `policy.with_label(f"...:{ticker}")` for a
per-call label.

**Sync callers: never `with ThreadPoolExecutor(...)` + `future.result(timeout=...)`.**
`__exit__` calls `shutdown(wait=True)`, so the timeout bounds nothing, and executor
workers are joined at interpreter exit even after `shutdown(wait=False)`. Use
`run_blocking_call_sync(policy, fn)` with an `InFlightLimit` on the policy: one daemon
thread per call, a slot held until the thread really ends, fail-fast
`BlockingCallSaturatedError` when full. An `InFlightLimit` on an async policy does the
same for `run_blocking_call` — use it where repeated stalls could pile up orphans.
None of this stops the stalled work; it bounds the caller and the orphan count.

LLM `ainvoke` is bounded the same way: `invoke_with_rate_limit_handling` in
`src/agents/runtime.py` wraps every call in `run_with_hard_timeout`. That wrap is the
load-bearing safety net against provider-side slow tails, where SDK-internal retries
could otherwise park a single call for the better part of an hour. Do not remove it in
a refactor; a test guards its presence.

**Admission over floors.** Inside that wrap, an LLM request is admitted against the
absolute deadline it inherits (`src/llm_runtime/timeouts.py`): it is sized to
`remaining - margin`, refused with `BudgetExhausted` when less than a useful
attempt remains, and every retry and backoff is re-admitted against the enclosing
deadline. Do not stretch a ceiling with a floor to make a slow request fit; give
the work an enclosing deadline (`enclosing_deadline`) and let admission truncate.

**The pipeline watchdog is the outermost deadline.** `scripts/pipeline_signals.sh`
hands each child `PIPELINE_TICKER_DEADLINE_EPOCH`. `run_deadline()` in
`src/llm_runtime/run_deadline.py` converts it once, less
`PUBLICATION_RESERVE_SECONDS`, and `src/main.py` binds it with
`enclosing_deadline_at` around the retrospective and the graph run. Never bind it
around render, persistence or shutdown: that work runs on the reserve. The stage
watchdog values are `Settings` fields, read by the script through
`python -m src.config_print`. Never put a literal default back in the script.
Operational scopes read it through the operational reader (which also sees a
tool runner) and never bind one. `run_blocking_call(_sync)` clamps every
blocking call to it. Work inside a hard cap sizes against a deadline kept back
by `deadline_margin(cap)`: equal timers race, and the cap's cancellation would
replace the work's own typed timeout.

**Start every child process through `src/subprocesses.py`.** Anywhere else it
forks (a bare name, `close_fds=True` or `cwd=` each force it), and `fork()` in this
threaded process can crash in Network.framework's atfork handler. The helper
also kills and reaps on every abnormal exit. `tests/architecture/test_subprocess_spawns.py`
enforces it.

**No literal deadline at an execution boundary.** A number passed to
`timeout=`, `*_timeout_seconds=`, `ClientTimeout`, `BlockingCallPolicy` or
`wait_for` belongs in a `Settings` field, clamped to the caller's deadline.
`tests/test_timeout_literals.py` scans for this shape. An allowlist entry is
keyed by scope and parameter, and it must say why a literal is correct there.

**A deadline is not a bound.** A checkpoint bounds nothing that follows it, and a call
starting just under the deadline can still run a full per-call timeout past it. Wrap
the whole workflow at one ceiling instead, and prove the node *returns promptly*
rather than that the work finishes.

`tests/test_to_thread_timeout_consistency.py` enforces this statically: every
`asyncio.to_thread` must be enclosed by `run_with_hard_timeout` or be allowlisted with
a justification, and `wait_for(to_thread(...))` is flagged as a regression. It
resolves asyncio aliases (`import asyncio as aio`, `from asyncio import ...`, injected
`asyncio_module=asyncio` parameters) — the injected spelling is how `src/data/gap_fill.py` hid a
`wait_for(to_thread(...))` from the literal-prefix version of the guard.

Full history: `docs/CODEBASE_MEMORY.md`.
