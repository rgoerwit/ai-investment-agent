---
paths:
  - "src/tooling/**"
  - "src/tools/**"
  - "src/mcp/**"
---

# Tools, hooks, and untrusted content

## Everything goes through the hook plane

Tool execution runs through `ToolExecutionService`, which is where audit logging,
argument policy, content inspection and budget enforcement all attach. A tool that
bypasses it gets none of them. MCP calls are no exception: they execute under a
canonical name through the same service, which is what puts them under inspection and
budget at one chokepoint.

Services resolve through a ContextVar with module singletons as fallback. Tests bind
scoped services rather than mutating the globals.

## Two tool deadlines, both load-bearing

The runner timeout in `src/tooling/runtime.py` does not cover the hooks; the
combined scope in `ToolExecutionService.execute` (runner plus
`TOOL_HOOKS_TIMEOUT_SECONDS`) is their only bound, for every caller, and the
only one: the graph node adds no wrapper. Tool scopes may *read* the caller's
enclosing deadline but must never *bind* one: the LLM judge fails open when
admission refuses it. Only the service binds the tool-call deadline
(`tool_call_deadline`, kept back from its cap by `deadline_margin`).

**Size all operational work with one reader.** `require_allowance`,
`clamp_to_operational_deadline`, `operational_time_left` or
`operational_deadline` in `src/llm_runtime/timeouts.py` see the caller *and*
the tool runner; `current_enclosing_deadline` is for LLM admission only.
`run_blocking_call(_sync)` clamps itself. A web-search, filing or document
timeout is a `Settings` field read at the call, resolved before the request,
coroutine or thread exists. Classify a timeout where it happens
(`_ScopedToolTimeout`), never relabel it. `tests/architecture/test_deadline_readers.py`
enforces the readers. See `docs/CODEBASE_MEMORY.md`.

MCP bounds go to the SDK's own parameters in `src/mcp/client.py`: the httpx2
client `timeout=`, `ClientSession(read_timeout_seconds=)` and
`call_tool(read_timeout_seconds=)`. Pass the call's bound explicitly, or it inherits
the handshake bound. The one wrapper is `asyncio.timeout(time left)` around the
public `list_tools()`, which has no per-request bound; never swap it for
`send_request`, which skips `_absorb_tool_listing`. Size every step and the retry
backoff with the operational readers, so the runner's deadline counts. Return a refusal or timeout from the session body and
raise it outside: inside, the SDK's task groups wrap it and `classify_mcp_error`
loses its category. See `docs/MCP.md`.

## Treat every external string as hostile

Search results, social feeds, retrieved memory, cached context, filings, and free-text
fields from financial APIs are all untrusted ingress and are inspected on the way in.
Content that reaches a prompt is wrapped in the trust-boundary formatter, and untrusted
context is delivered with reduced authority — as a human-authored message, never as a
system instruction.

Memory is filtered on the **write** path as well as the read path; otherwise a
poisoned document is persisted once and returned as trusted context forever.

## Conflict tolerance is not conflict blindness

A contract marked as having conflicting valid payloads is sticky for the rest of the
run, so the comparison that sets it must ignore genuinely volatile fields — a price
tick between two fetches is not a conflict. The exclusion list is verified disjoint
from the analysis-critical fields, so it can never mask a real disagreement. Widening
it needs the same proof.

## Verify a vendor's real tool surface

**Call `list_tools()` before adding a vendor.** Never infer tool names from REST docs
or a blog post: a hallucinated allowlist produces a runtime error on every call, and
the failure looks like a transport problem. Use the printed names and input schemas
verbatim.

One vendor's missing credential disables **that server**, not the plane. A missing key
is a distinct condition from a malformed registry: the first is an operator state and
logs at info, the second is an error. And a tool whose server did not resolve must
stop being offered — otherwise the model plans a call against it and the resulting
failure counts toward a partial-failure ratio, trading a clean absence for a
review-discarding error.

The registry template is `config/mcp_servers.example.json`; the live registry is
operator-local.

## MCP error classification fails silently

The SDK runs on `httpx2`, not `httpx`: hand it an `httpx2` client and match
`httpx2` exception types. It folds a non-2xx POST into a stand-in JSON-RPC error
and wraps errors in `ExceptionGroup`s, so the response hook that records the HTTP
status and the group unwrapping are both load-bearing. Without them, 401 and 429
lose their AUTH cooldown and backoff and nothing goes red. Run
`tests/mcp/test_errors.py`, which drives the real SDK, after any SDK bump or edit
to `src/mcp/errors.py` or `_open_session`.

Full history: `docs/MCP.md`.
