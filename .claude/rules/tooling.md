---
paths:
  - "src/tooling/**"
  - "src/tools/**"
  - "src/mcp/**"
---

# Tools, hooks, and untrusted content

## Everything goes through the hook plane

Tool execution runs through `ToolExecutionService`, which is where audit logging,
argument policy and content inspection attach. A tool that bypasses it gets none of
them. MCP calls are no exception: they execute under a canonical name through the
same service, which is what puts them under inspection at one chokepoint.

**Count a sent request where it is sent, never in an `after()` hook.** `after()`
runs only for a runner that succeeded with time left, so quota a vendor consumed
on a timeout, failure or discarded result went uncounted. `MCPRuntime` takes one
`BudgetTracker.reserve` per tools/call before sending it: a single `BEGIN
IMMEDIATE` transaction on a worker that checks and takes both limits. The count
is admitted reservations, never refunded; `has_room` is advisory only. A
server's `Retry-After` is honoured in full when the deadline fits it and refused
otherwise; the backoff cap applies only to the exponential backoff.

**A tool reads its own inspection through `approved_tool_content`.** It returns
the approved text, or raises `ToolCallBlocked` (`INSPECTION_BLOCKED`) or
`ToolCallFailed` (`INSPECTION_TIMEOUT`, `INSPECTION_FAILED`), from the enforced
disposition the configured mode produced, never the raw decision. A blocked
placeholder never comes back as evidence or a search result. A tool whose one
Tavily search is its whole answer passes `raise_on_block=True`; one that merges
sources drops a blocked one; a direct caller outside the service catches
`ToolCallBlocked` itself (the macro-context fetch). The content hook delivers
only what it inspected: past its input limit the inspected, truncated text, and
an output it cannot serialize fails as `INSPECTION_FAILED`. Reason codes are
defined once in `src/tooling/runtime.py`, and every surface tells the model the
code `tool_failure_reason` gives. The service checks hook admission before
every `before()` and `after()` hook and after the last of each. Admission cannot
undo a hook's side effect, so durable records go in an observer (below).
CPU-bound document parsing (PDF and HTML) runs on the shared extraction pool,
never on the event loop.

Services resolve through a ContextVar with module singletons as fallback. Tests bind
scoped services rather than mutating the globals.

## Two tool deadlines, both load-bearing

The runner timeout in `src/tooling/runtime.py` does not cover the hooks; the
combined scope in `ToolExecutionService.execute`
(`tool_execution_with_inspection_timeout_seconds`: the runner allowance plus
`TOOL_HOOKS_TIMEOUT_SECONDS`, clamped to the operational deadline) is their
bound for every caller, and the only one: the graph node adds no wrapper. Tool scopes may *read* the caller's
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
(`_ScopedToolTimeout`), never relabel it. Share a multi-phase tool's allowance
by fractions of it, never absolute reserves (the official-document
`_allocate`), and inspect inside a bound through
`InspectionService.check_within`, which fails closed and binds nothing. `tests/architecture/test_deadline_readers.py`
enforces the readers. See `docs/CODEBASE_MEMORY.md`.

MCP bounds go to the SDK's own parameters in `src/mcp/client.py`: the httpx2
client `timeout=`, `ClientSession(read_timeout_seconds=)` and
`call_tool(read_timeout_seconds=)`. Pass the call's bound explicitly, or it inherits
the handshake bound. `MCP_TOOL_CALL_TIMEOUT_SECONDS` ships unset and follows the
runner's timeout; read it only through `_configured_tool_call_timeout`. The
cooldown is rechecked immediately before `call_tool`, after listing and
reservation; a test that skips a sleep advances `FrozenWallClock`
(`tests/helpers/mcp_server.py`), or a retry meets its own cooldown. The one wrapper is `asyncio.timeout(time left)` around the
public `list_tools()`, which has no per-request bound; never swap it for
`send_request`, which skips `_absorb_tool_listing` and so the output-schema cache
(the `x-mcp-header` filter there is inert: `initialize()` negotiates only legacy
versions). Never send a call for a tool the listing lacks, or the SDK re-lists
after it under the stale session bound: refuse it (`tool_not_listed`). Size every step and the retry
backoff with the operational readers, so the runner's deadline counts. Return a refusal or timeout from the session body and
raise it outside: inside, the SDK's task groups wrap it and `classify_mcp_error`
loses its category. See `docs/MCP.md`.

## Outbound page fetches say who they are, and failures say why

**Never collapse transport failures into one reason.** The official-document tool
maps an exception to its reason by class over the whole cause chain
(`_TRANSPORT_REASONS`: TLS, timeout, connect, protocol); add a row, not a branch.
**Every client that fetches third-party pages sends `config.http_user_agent`**
(`HTTP_USER_AGENT`); never write a UA literal. sec.gov additionally needs
`SEC_CONTACT_EMAIL` (refuse before connecting without it; never send it elsewhere).
A tool call's outcome is recorded where the service decides it, not in each
caller: `execute` tells observers either `completed()` with the returned result or
`failed()` with the reason code, never both. **Write a durable record there, never
in `after()`**, which can run for a result the caller is then told timed out. Story: `docs/CODEBASE_MEMORY.md`.

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
