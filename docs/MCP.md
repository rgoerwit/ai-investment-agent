# Model Context Protocol (MCP) integration

Last updated: 2026-09-30 (mcp 2.x migration).

The consultant agent can cross-validate analyst claims against vendor MCP
servers (currently the FMP MCP server). The consultant-facing MCP wrappers
route calls through the normal `ToolExecutionService` hook chain, so audit
logging, argument policy, content inspection, and per-server budget can all
observe those calls under the canonical name `mcp__<server>__<tool>`.

`MCPRuntime.call_tool()` still exists as a convenience/testing path, but it is
not the recommended agent-facing integration surface for this repo.

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

`MCPBudgetHook` runs as part of the shared hook chain and gates each call:

- `before()` blocks the call when `daily_call_limit` or `per_run_limit` is
  exhausted (raises `ToolCallBlocked`, surfaced as a `TOOL_BLOCKED:` sentinel
  in the consultant tool's structured failure JSON).
- `after()` records one upstream consumption for any non-blocked result —
  including upstream `isError=true` payloads, which still consume vendor
  quota.

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
level blocks. Use the value to disambiguate: `config` (allowlist/scope/spec),
`auth` (401/403, vendor cooldown), `transport` (5xx/429/network),
`tool_error` (vendor returned `isError=true`), `inspection` /
`inspection_blocked` (content inspector rejected the payload).

If the inspector sanitizes a payload into plain text instead of blocking it, the
consultant wrappers return a JSON object with `text_payload` and
`note="mcp_payload_sanitized_or_textual"` rather than mislabeling the result as
a protocol failure.

## Smoke testing the integration

`scripts/mcp_smoke.py` is the canonical way to answer "is MCP actually moving
bytes right now?" without burning tokens or waiting for a full analysis. It
bypasses the consultant LLM and calls `spot_check_metric_mcp_fmp` directly
through the full hook chain — same canonical `mcp__<server>__<tool>` name,
same audit/policy/inspection/budget hooks, real vendor traffic.

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
