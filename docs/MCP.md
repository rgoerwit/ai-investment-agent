# Model Context Protocol (MCP) integration

Last updated: 2026-10-10 (budget counted at send, unlisted tools refused, legacy-only header filter).

The consultant agent can cross-validate analyst claims against vendor MCP
servers (currently the FMP MCP server). The consultant-facing MCP wrappers
route calls through the normal `ToolExecutionService` hook chain, so audit
logging, argument policy and content inspection observe those calls under the
canonical name `mcp__<server>__<tool>`. `MCPRuntime.execute_raw()` is the one
entry point; it runs as the service's runner and enforces the per-server budget
itself.

## Vendor surface notes (verified May 2026)

Call `session.list_tools()` against any new vendor before adding it to the
registry — assumed tool names are unreliable, and a vendor's REST documentation
routinely does not match its MCP surface. `scripts/mcp_smoke.py` exercises a
configured server through the full hook chain without an LLM in the loop.

- **FMP** uses a *dispatcher* pattern. Top-level tools (`statements`, `quote`,
  `analyst`, `chart`, etc.) take an `endpoint` enum argument that selects the
  actual operation. So fetching financial ratios looks like
  `tool="statements", arguments={"symbol": ..., "endpoint": "metrics-ratios-ttm"}`.
  The consultant wrapper's `_FMP_METRIC_DISPATCH` table in
  `src/consultant_tools.py` encodes the metric → (tool, endpoint) mapping.
  When adding a new metric, call `session.list_tools()` and pick from the
  advertised `endpoint_enum` for the relevant tool.
- **Twelve Data** is intentionally **not exposed** in this repo. Their public
  MCP only publishes `u-tool` (a free-form AI router) and `doc-tool`. Neither
  fits the consultant's narrow-allowlist + structured-payload contract; the
  registry entry is left with `enabled: false` until they ship structured
  per-metric tools.

## mcp 2.x migration (Oct 2026)

`pyproject.toml` pins `mcp = ">=2.2.0,<3.0.0"`. The repo stayed on 1.x through
Aug 2026 because several v2 changes fail **silently** for a client like ours;
it moved once openai 3, anthropic 1 and langsmith 0.12 had pulled in `httpx2`
anyway, which removed most of the extra dependency cost.

We are a **client only**: one server (FMP remote) over streamable HTTP, calling
`list_tools` and `call_tool`. The server-side parts of upstream's migration guide
do not apply. These did, and the silent ones are why `tests/mcp/test_errors.py`
drives the **real** SDK over an `httpx2.MockTransport` rather than mocking it:

| Site | v2 change | Failure mode if missed |
|---|---|---|
| `client.py` `_open_session` | `http_client=` must be an `httpx2.AsyncClient` | **Silent.** A plain `httpx` client "degrades in subtle ways (server-initiated messages stop arriving)" |
| `errors.py` `classify_mcp_error` | SDK raises `httpx2` exceptions | **Silent.** This repo also depends on `httpx`, so `import httpx` still works and `isinstance(exc, httpx.HTTPStatusError)` just never matches |
| `errors.py` | `McpError` → `MCPError` | **Silent** under the old `try/except ImportError` import |
| `errors.py` + `client.py` | A non-2xx **POST** no longer raises `HTTPStatusError`: the SDK turns it into a JSON-RPC error with a stand-in code (`-32603` "Server returned an error response") and **drops the status** | **Silent.** 401 and 429 became generic protocol errors. Fixed by an `httpx2` response hook that records the last POST error status (`HTTPErrorRecord`); 401/403/429/5xx then override the stand-in code. Other 4xx keep the server's JSON-RPC code, so a spec-correct 400 with an error body stays a protocol error |
| `errors.py` | Errors from `initialize`/`call_tool` arrive wrapped two `ExceptionGroup`s deep (anyio task groups) | **Silent.** The group fell through to the generic non-retryable fallback. `_unwrap_exception_group` picks the first `MCPError`/`httpx2.HTTPError` leaf |
| `client.py` | `streamable_http_client` yields a 2-tuple | Loud `ValueError` |
| `normalize.py`, `client.py` | `structuredContent`/`isError`/`inputSchema`/`outputSchema` → snake_case attributes | Loud `AttributeError`. The camelCase names remain valid **constructor** aliases, so test fixtures that build results with them still pass and prove nothing about attribute access |

The last two rows were not in the pre-migration audit; they surfaced only when
the end-to-end tests ran against the real SDK. Whether 1.x also wrapped errors in
groups was not checked, so do not assume the AUTH cooldown and 429 backoff ever
worked before this migration.

The `httpx2` fork also logs under its own logger names (`httpx2`, `httpcore2`);
quieting only `httpx` leaves one INFO line per LLM or MCP request.

v2 verifies TLS against the OS trust store via `truststore` instead of `certifi`.
The `python:*-slim` base image ships `ca-certificates`, so this should be
transparent in Docker, but no live MCP call has been made from the container
since the migration.

## Configuration

1. Set environment variables in `.env`:

   ```ini
   MCP_ENABLED=true
   MCP_SERVERS_PATH=config/mcp_servers.json
   MCP_USAGE_DB_PATH=runtime/mcp_usage.db
   CONSULTANT_MCP_ENABLED=true
   ```

2. Copy `config/mcp_servers.example.json` to `config/mcp_servers.json` and set
   `enabled: true` on the entries you want to use. Provide the API key envs
   referenced by `auth.env_var` (e.g. `FMP_API_KEY`, `TWELVE_DATA_API_KEY`).

If `MCP_ENABLED=true` and the registry file is missing, empty, or contains no
enabled servers, runtime startup logs a warning and consultant MCP wrappers stay
hidden.

## Server registry schema

Each entry in `config/mcp_servers.json` declares one MCP server:

| Field              | Required | Notes                                                       |
| ------------------ | -------- | ----------------------------------------------------------- |
| `id`               | yes      | Stable identifier; used as the `mcp__<id>__*` name prefix.  |
| `transport`        | yes      | `"streamable_http"` or `"stdio"`.                           |
| `base_url`         | http     | HTTPS URL for `streamable_http` servers.                    |
| `command`/`args`   | stdio    | Process spec for `stdio` servers.                           |
| `auth`             | no       | `none` / `query_api_key` / `header_bearer` / `header_static`. Non-`none` types must set `env_var`. |
| `enabled`          | no       | `false` keeps the entry in the registry but hides it.        |
| `scopes`           | yes      | Which agent scopes may use this server (e.g. `"consultant"`). |
| `tool_allowlist`   | yes      | Tools the consultant may invoke; everything else is rejected. |
| `daily_call_limit` | no       | 0 disables; otherwise upper bound per UTC day per server.    |
| `per_run_limit`    | no       | 0 disables; otherwise upper bound per analysis run.          |
| `trust_tier`       | yes      | `official_vendor` / `community` / `unknown`. The content inspector applies a lower threshold for `official_vendor` structured payloads. |

## Trust tier vocabulary

- `official_vendor` — first-party endpoint maintained by the data provider
  (e.g. Twelve Data's own MCP server). Structured-financial payloads from
  `official_vendor` sources receive a halved heuristic weight.
- `community` — third-party endpoint with a public maintainer.
- `unknown` — anything else; the inspector applies its default weights.

## Scope semantics

`scopes` is an allowlist. The consultant wrappers call
`MCPRuntime.execute_raw(..., scope="consultant")` through the shared tool hook
chain; servers without `"consultant"` in `scopes` raise
`MCPCallError(category=CONFIG)` before any network call.

## Budget interaction

`MCPRuntime` owns the budget, beneath the hook chain:

- **Admission.** Each attempt is admitted with `BudgetTracker.can_call` before
  a session opens. A spent `daily_call_limit` or `per_run_limit` (0 means
  unlimited) raises `MCPCallError(category=BUDGET)`, `failure_kind: "budget"`
  in the consultant JSON, with nothing sent. A retry the budget cannot cover is
  refused like a deadline refusal: the sent attempt's error, with
  `details.retry == "refused_budget"`.
- **Accounting.** Each tools/call is counted as it is handed to the SDK,
  whatever follows: a timeout, a 429 or 5xx, an `isError=true` payload, a
  result that fails its output schema, a cancellation, or a result the tool
  plane discards. A retried call counts twice. `initialize` and `tools/list`
  are not counted: vendor quotas meter data endpoints, and a request-rate limit
  on a listing surfaces as a 429 and its cooldown.

**Why not a hook (October 2026, round 3).** The budget used to be an
`MCPBudgetHook`, counting in `after()`. That hook runs only when the runner
succeeds and the combined deadline still has room, so a call the vendor
received and then timed out, failed or had its result discarded (the after()
chain stops once the deadline is spent) was never counted, and a retry counted
once for two sends. Counting at the send boundary is the only place that sees
every request. A call that fails locally inside the SDK before reaching the
wire is over-counted, which is the safe direction for a quota. Two concurrent
calls can still overshoot a limit by the number in flight.

Counts persist in the SQLite database at `MCP_USAGE_DB_PATH`. Rows older than
the retention window are swept on startup.

## Diagnosing a blocked call

The consultant tool `spot_check_metric_mcp_fmp` always returns JSON. On block
or failure it emits:

```json
{
  "error": "...",
  "ticker": "...",
  "metric": "...",
  "provider": "fmp",
  "failure_kind": "config|auth|transport|protocol|tool_error|inspection|budget|inspection_blocked",
  "retryable": true,
  "source": "fmp_mcp"
}
```

`failure_kind` mirrors `MCPErrorCategory` plus `inspection_blocked` for hook-
level blocks. Use the value to disambiguate: `config` (allowlist/scope/spec, or
a tool the server does not list: `details.refused == "tool_not_listed"`),
`budget` (the server's call budget is spent),
`auth` (401/403, vendor cooldown), `transport` (5xx/429/network/timeout, see
[Debugging a timed-out call](#debugging-a-timed-out-call)),
`tool_error` (vendor returned `isError=true`), `inspection` /
`inspection_blocked` (content inspector rejected the payload).

If the inspector sanitizes a payload into plain text instead of blocking it, the
consultant wrappers return a JSON object with `text_payload` and
`note="mcp_payload_sanitized_or_textual"` rather than mislabeling the result as
a protocol failure.

## Timeout model

Four settings. Each is read when the step starts and shrinks to the time left:
the caller's enclosing deadline (`src/llm_runtime/timeouts.py`) and, under the
tool service, the runner's `tool_call_deadline` (`src/llm_runtime/timeouts.py`),
kept back from the runner's cap by `deadline_margin`, read through the
operational clamp. MCP reads that deadline and never binds it, so with the
45 s call setting under the 45 s runner the call's own timeout fires first.

| Setting | Default | Applied as | Bounds |
|---|---|---|---|
| `MCP_HTTP_TIMEOUT_SECONDS` | 5 s | `httpx2.AsyncClient(timeout=...)` | each phase (connect, read, write, pool) of each HTTP request. It does not bound a whole request |
| `MCP_INITIALIZE_TIMEOUT_SECONDS` | 30 s | `ClientSession(read_timeout_seconds=...)`, and `asyncio.timeout(...)` around `list_tools()` | the wait for the initialize response, connecting included; then, re-read after the handshake, the tools/list catalogue response |
| `MCP_TOOL_CALL_TIMEOUT_SECONDS` | 45 s | `call_tool(read_timeout_seconds=...)` | the wait for one tool call's result |
| `MCP_RETRY_MIN_USEFUL_SECONDS` | 1 s | `wait_allowance(minimum=...)` before the retry backoff | the least time that must be left after the backoff for the one retry to be sent |

**The catalogue (October 2026, round 2).** `list_tools()` takes no per-request
timeout, and the session's read timeout is fixed when the session opens, so a
listing sent after a slow handshake used to get a fresh full bound. It now runs
under `asyncio.timeout(time left)` inside the session body (`_list_tools_in_time`).
A timeout or a spent allowance is captured there and raised after the SDK contexts
have exited normally, like every other MCP error. It stays the public
`list_tools()`: calling `send_request` directly would skip `_absorb_tool_listing`,
which caches the output schemas the SDK validates a result against. On a
2026-07-28 session it would also drop tools with an invalid `x-mcp-header`
annotation, but **that filter does not protect production**: `initialize()`
negotiates only legacy versions, where it is inert, and only `discover()`
reaches the modern one. Negotiating it was considered and declined: every
`execute_raw` opens a fresh session, so a `server/discover` probe (with a
legacy fallback on `METHOD_NOT_FOUND`) would add one or two round trips per
call; the SDK bounds that probe by its own fixed 10 s, not our settings; the
modern session changes header and caching semantics; and neither vendor's
support can be verified offline. A test pins both behaviours.

**Every call lists first.** The SDK validates a successful result against the
tool's output schema, and in a fresh session -- every `execute_raw` opens one --
it fetched that schema with a tools/list sent *after* the call, under the stale
session bound. `_call_in_time` now lists first, within the time left, so the
schema is cached and that post-call listing is never sent. A successful call
sends the same requests as before, in a different order; a call that fails
(timeout, error payload) now costs one listing it did not before. A listing that
fails means the call is never sent.

**A tool the server does not list is refused (round 3).** The SDK also re-lists
after a call whose tool is missing from its cache, under the same stale bound,
and then skips validation. That happened for an allowlisted tool the server
does not offer (a registry error) and for a tool on a later page, because the
listing read only the first. `_list_tools_in_time` now follows `nextCursor`,
all pages under its one bound, until the tool appears; it stops at the last
page or a repeated cursor. A tool still missing is refused unsent and uncounted
as `MCPCallError(category=CONFIG, details={"refused": "tool_not_listed"})`, with
no retry and no cooldown.

**The retry backoff is admitted.** `_execute_with_retry` asks
`wait_allowance` (the same rule as an LLM retry) whether a useful attempt still
fits after the backoff. A transient backoff is shortened to fit; a server's
`Retry-After` is not, so it is refused. It re-checks after the sleep. A refused
retry raises the last attempt's error (its category, HTTP status and JSON-RPC
code kept, and its 429 or AUTH cooldown still registered) with
`retryable: false` and `details.retry == "refused_deadline"`: sent, then failed,
and the retry never sent.

What the qualification found (mcp 2.2.0, httpx2 2.13.0):

- **The HTTP default was already 5 s, by accident.** `streamable_http_client`
  keeps a supplied client's timeout. The SDK's own 30 s, with a 300 s read
  (`mcp/shared/_httpx_utils.py`), applies only to a client it creates. Ours passed
  no timeout, so it ran on httpx2's default of `Timeout(5.0)`. The setting keeps
  that. A vendor that takes more than 5 s to start a response fails here first,
  as a retryable `transport` error.
- **The protocol layer had no bound.** `ClientSession`'s read timeout defaults to
  `None`. `initialize()` and `list_tools()` take no per-request timeout and use
  the session's. `call_tool()` accepts its own, and falls back to the session's
  when it gets none. That is why the call passes its bound explicitly; otherwise
  it would inherit the 30 s handshake bound. A stdio server that never answered
  `initialize` hung forever. So did an HTTP server that sent bytes slowly enough
  to beat the per-phase timeout.
- **Why 30 s for the handshake.** It is the SDK's own default for general
  operations. It is also far above a healthy HTTP handshake, which the 5 s
  per-phase limit already keeps short. It is a new bound, not a re-tuning.
- **Why 45 s for the call.** The Consultant's only MCP path runs inside two
  tool-runner scopes. One is the outer `spot_check_metric_mcp_fmp` call (45 s);
  the other is the inner `mcp__<server>__<tool>` call, which since October 2026
  (round 2) gets only what the outer runner has left, never a fresh 45 s: the
  service binds each call's deadline for any call nested inside it
  (`tool_call_deadline` in `src/tooling/runtime.py`). Both bound the whole call,
  session open included, so the runner fires first and the outcome is
  unchanged. If you raise `TOOL_EXECUTION_TIMEOUT_SECONDS` for a slow vendor,
  raise this one too.
- **Cleanup is the SDK's job, and it does it.** When a read timeout fires, the
  dispatcher raises `MCPError(REQUEST_TIMEOUT)`. For a tool call, it first sends
  `notifications/cancelled` so the server stops work. Leaving the session then
  cancels the transport's task group, which cancels the in-flight POST, and closes
  the httpx2 client. stdio shutdown closes stdin, waits, then kills the process
  tree; it is shielded and every wait is bounded (`mcp/client/stdio.py`).
  `tests/mcp/test_mcp_timeouts.py` drives the real SDK over a stalled mock server
  and checks that the parked request is cancelled, the client is closed, and no
  task outlives the session.

### Debugging a timed-out call

| What you see | Cause |
|---|---|
| `failure_kind: "transport"`, `retryable: true`, message `Request 'initialize' timed out` | The handshake outlived `MCP_INITIALIZE_TIMEOUT_SECONDS`. `tools/list` reads the same |
| the same with `Request 'tools/call' timed out` | The call outlived `MCP_TOOL_CALL_TIMEOUT_SECONDS`. The server was sent a cancel |
| `transport`, `retryable: true`, an httpx2 `ReadTimeout` / `ConnectTimeout` message | One HTTP phase outlived `MCP_HTTP_TIMEOUT_SECONDS`; usually a vendor slow to start its response |
| `transport`, `retryable: false`, `details.timeout == "enclosing_deadline_spent"` | The caller's or the tool runner's deadline had already passed. `details.scope` names the step: `mcp_initialize` means no session was opened, `mcp_list_tools` that one was opened but no listing (and so no call) was sent, and `mcp_tool_call` that the listing was sent but no call |
| `transport`, `retryable: true`, `details.timeout == "time_left_elapsed"`, `details.scope == "mcp_list_tools"` | The catalogue was sent and outlived what the handshake left. Under `execute_raw` the retry is then usually refused (next row) |
| any category, `retryable: false`, `details.retry == "refused_deadline"` | The attempt was sent and failed retryably, and the deadline left no room for the backoff plus `MCP_RETRY_MIN_USEFUL_SECONDS`. The rest of the error is that attempt's own |
| `TOOL_ERROR: TimeoutError` in the Consultant's tool output, or `tool_call_timeout` in the log | The log's `scope` names what fired: `runner` (`TOOL_EXECUTION_TIMEOUT_SECONDS`; work outside the MCP bounds, since those end a margin earlier), `with_inspection` (the combined runner-plus-`TOOL_HOOKS_TIMEOUT_SECONDS` scope, usually a slow inspection), `before_hooks`/`after_hooks` (a hook's own timeout) or `hooks_allowance_spent` (no time left to inspect the result) |

A retryable failure is retried once in `_execute_with_retry`,
with a fresh session and a re-read allowance, when the backoff is admitted. So
a stalled call costs at most two bounds plus the backoff, and never more than
the time left. Before October 2026 a
timed-out request classified as `protocol`.

## Smoke testing the integration

`scripts/mcp_smoke.py` is the canonical way to answer "is MCP actually moving
bytes right now?" without burning tokens or waiting for a full analysis. It
bypasses the consultant LLM and calls `spot_check_metric_mcp_fmp` directly
through the full hook chain — same canonical `mcp__<server>__<tool>` name,
same audit/policy/inspection hooks and the runtime's budget, real vendor traffic.

```bash
# Default: 6914.T currentPrice
poetry run python scripts/mcp_smoke.py

# Pick any (ticker, metric) covered by _FMP_METRIC_DISPATCH
poetry run python scripts/mcp_smoke.py --ticker AAPL --metric trailingPE

# Quieter — suppresses SDK/hook INFO chatter on stderr
poetry run python scripts/mcp_smoke.py --quiet
```

Exit codes are CI-friendly:

| code | meaning                                                          |
| ---- | ---------------------------------------------------------------- |
| `0`  | success — vendor returned a numeric `value`                      |
| `1`  | vendor- or hook-level failure — see `failure_kind` in the JSON   |
| `2`  | config / runtime not available (MCP disabled or registry unread) |
| `3`  | unexpected internal error (architectural failure, not vendor)    |

Structured JSON is always printed to stdout — pipe through `jq` for further
inspection. A typical free-tier FMP run produces `failure_kind="tool_error"`
with a 402 message — that means the integration is healthy and the limitation
is purely the FMP plan tier.

## Adding a new server

1. Add an entry to `config/mcp_servers.json` matching the schema above.
2. Restrict `tool_allowlist` to the smallest useful surface — the consultant
   should not see vendor admin or write tools.
3. Choose a `trust_tier` honestly; default to `unknown` for new entries.
4. Pick conservative `per_run_limit` / `daily_call_limit` so the consultant
   cannot exhaust vendor quota during a single analysis.
5. Set `enabled: true` once the API key envs are populated.

For `stdio` servers, auth-backed environment variables are injected into the
subprocess environment automatically. Extra non-secret environment variable names
can still be listed under `env_vars`.

The runtime validates all of the above eagerly; a bad entry raises
`ValueError` at startup rather than at first call.
