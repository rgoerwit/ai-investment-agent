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

LLM `ainvoke` is bounded the same way: `invoke_with_rate_limit_handling` in
`src/agents/runtime.py` wraps every call in `run_with_hard_timeout`. That wrap is the
load-bearing safety net against provider-side slow tails, where SDK-internal retries
could otherwise park a single call for the better part of an hour. Do not remove it in
a refactor; a test guards its presence.

**A deadline is not a bound.** A checkpoint bounds nothing that follows it, and a call
starting just under the deadline can still run a full per-call timeout past it. Wrap
the whole workflow at one ceiling instead, and prove the node *returns promptly*
rather than that the work finishes.

`tests/test_to_thread_timeout_consistency.py` enforces this statically: every
`asyncio.to_thread` must be enclosed by `run_with_hard_timeout` or be allowlisted with
a justification, and `wait_for(to_thread(...))` is flagged as a regression.

Full history: `docs/CODEBASE_MEMORY.md`.
