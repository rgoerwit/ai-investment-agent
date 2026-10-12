# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Debt-to-equity hardening. Statement-derived D/E keeps its provenance through
  the merge, and the total-debt/book-equity cross-check only logs a ×100
  disagreement instead of replacing the value: the unit is fixed per source,
  and book value, share count and debt can come from different periods or share
  classes. The cross-check also guards underflow, overflow and malformed inputs.
  `DE_RATIO` is read from the last DATA_BLOCK as one complete token, so
  `8.05 %` is a percentage and `8.05e-2`, grouped or suffixed values are
  rejected rather than truncated. The `DE_DISCREPANCY` tolerance is a named
  constant.

- A response the provider stopped at its output cap is classified
  `output_cap_exhausted` from its stop reason before the token count is
  compared with the cap. 0995.HK's Senior stopped on `MAX_TOKENS` reporting
  19,111 of 19,115 tokens, so it was classified `incomplete_structured_output`
  and the new `RETRY-CAP` recovery never ran.

- `debtToEquity` is one canonical ratio from the source boundary to every
  consumer. `src/data/debt_equity.py` owns each source's verified unit and
  basis: yfinance and yahooquery report total debt / equity in percent and are
  converted before the multi-source merge compares values; FMP, Alpha Vantage,
  EODHD and web-extracted values, whose unit or basis is unverified, contribute
  nothing. The magnitude and EV-versus-market-cap guess in the fetcher is gone
  (net cash is not evidence of a unit); a cross-check against total debt / book
  equity logs `debt_equity_unit_conflict` on a ×100 disagreement and adopts the
  derived ratio. A unitless `DE_RATIO` is read as a ratio (fundamentals prompt
  11.17 declares it), unitless legacy `D/E:` lines no longer gate, and
  `EXTREME_LEVERAGE` and the refinancing-risk D/E clause gate on the canonical
  raw ratio, with a non-blocking `DE_DISCREPANCY` when the report disagrees.
  TOT.TO's 8.047 (8.05%) had been read as 805% and auto-rejected in Energy; all
  six Aug–Oct `EXTREME_LEVERAGE` rejects were this misread, and replaying them
  on the canonical ratio passes each one.

- An `ACTIVE_TENDER` without a positive tender price is demoted to `RUMORED`
  before anything acts on it, through `effective_m_and_a_status` in
  `src/data_block_utils.py` (the metric extractor, the retrospective snapshot and
  both value-trap call sites); a stake percentage, a date, or a value that opens
  with `N/A` is not a price. The Foreign Language Analyst now reports the
  analyzed company's Subject Role (TARGET / ACQUIRER / OTHER), and the Senior
  sets `ACTIVE_TENDER` only for a priced target (prompts: foreign language 1.24,
  fundamentals 11.16). 6670.TW, the acquirer in a tender for 51% of 6904.TW with
  no price, had been tagged `ACTIVE_TENDER`, which exempted its health and growth
  gates. Snapshots already in the analysis index keep their stored status until
  the ticker is re-analyzed.

- An analyst output that exhausted its output cap is recovered at a lower
  reasoning value, not a higher one. The structural retry re-ran every contract
  failure at the retry seat's raised effort, so when thinking had consumed the
  cap (0995.HK's Senior Fundamentals on `gemini-3.8-flash`, 19,111 of 19,115
  tokens) the retry failed the same way and the run failed closed. Such
  failures now rerun as `(RETRY-CAP)` on the same binding at the FAST intent's
  reasoning value, through `build_model_for_seat(reasoning_intent=...)`; other
  failures keep `(RETRY-HIGH)`, and the legacy schema, which cannot lower
  reasoning, keeps it with a logged fallback.

- A pydantic `ValidationError`, such as a model's invalid tool argument, is
  classified as `application_error` with no host, before any message-marker
  scan. It used to fall through to `unknown_provider_error` with host
  `errors.pydantic.dev`, taken from the documentation URL pydantic appends to
  its message. The `LLM_MODEL_FAILOVERS` example now names `gemini-3.6-flash`,
  since `gemini-3.7-flash` was deprecated on 2026-10-08.

- An MCP server's cooldown is rechecked immediately before a `tools/call` is
  sent, after the listing and the budget reservation, so a caller paused there
  while another caller hit a 429 or 401 no longer sends during the cooldown; its
  reservation stays counted.

- A cancelled caller never publishes an article. `publish_article` refuses from a
  task that is being cancelled, as well as after an expired deadline, and
  `ArticleWriter.write` drafts through `run_article_workflow_step`; a caller
  cancelled before the deadline whose draft completed anyway used to overwrite
  the destination.

- The content-inspection hook delivers only what it inspected. A result over its
  50k-character input limit that the inspector allows is returned as the
  inspected, truncated text with a finding, not the original, whose tail used to
  reach the model uninspected; a result it cannot serialize fails as
  `INSPECTION_FAILED` instead of passing through uninspected.

- `make env-align` (`scripts/align_env_file.py`) writes a candidate copy of an
  existing `.env` laid out like `.env.example`: each of your settings sits where the
  template names its key, template defaults you never set are commented out, and
  anything the template does not mention is kept in trailing sections. Values are
  copied verbatim and the result must load to exactly the same values, raw and
  interpolated, or nothing is written; the report is counts only. The candidate
  replaces the output atomically and owner-only (0600 before it is visible), and
  an output that is the source or the template under any name, a hardlink or
  symlink included, is refused.

- The CLI publishes an approved article or its `.draft` through `publish_article`,
  the one publication step, instead of a bare atomic write, so it logs
  `article_saved` like every other publication. It runs after the workflow scope
  has closed, so a `.draft` kept after a workflow timeout is still saved; a failed
  write leaves an existing file intact and is reported like any other article
  failure.

- `MCP_TOOL_CALL_TIMEOUT_SECONDS` is unset by default and then follows
  `TOOL_EXECUTION_TIMEOUT_SECONDS` (still 45 s), resolved in one place in
  `src/mcp/client.py` for both the call allowance and the retry's minimum useful
  attempt; an explicit value is honoured as before. It repeated the runner's literal,
  so raising the runner's cap left the MCP call bound at 45 s.

- A tool's own content inspection is read one way. `get_official_filings`,
  `search_foreign_sources`, `extract_guidance_sources`, `search_claim` and
  `fetch_reference_content` returned the `TOOL_BLOCKED:` placeholder as their text
  (as `STATUS: EVIDENCE_FOUND` or as search results), so the ledger recorded a block
  as a success; they now go through `approved_tool_content`, as the official-document
  tool does: a block is a `BLOCKED` record, a timeout or backend failure a `FAILED`
  call. A blocked Tavily result is no result for callers that merge several sources;
  single-source tools raise it (`search_claim`, `extract_guidance_sources`,
  `search_legal_tax_disclosures` and `get_macroeconomic_news`; the macro-context
  fetch treats a blocked region as thin so the GLOBAL fallback still runs). Every
  surface (graph node, Consultant, Auditor, editor) now tells the model the code the ledger records (`TOOL_TIMEOUT`, the typed
  code, or `TOOL_EXECUTION_FAILED`), never an exception class or exception prose;
  the editor loop used to send the exception text. Reason codes are defined once in
  `src/tooling/runtime.py`: `INSPECTION_BACKEND_ERROR` is now `INSPECTION_FAILED`
  and `OFFICIAL_DOCUMENT_BLOCKED` is `INSPECTION_BLOCKED`. Memory writes and the
  macro cache log a backend failure as `*_inspection_failed`, not as a block.
  Each tool call now has one ledger outcome: the evidence recorder used to record in
  its `after()` hook, so an allowance that ran out during or just after it left the
  call both `SUCCEEDED` and `FAILED` while the caller was told it timed out. The
  service now tells `ToolCompletionObserver`s the result it returns, or failure
  observers why there is none, never both; the recorder and the baseline capture's
  `after` event record there. A `before()` policy block is now recorded as
  `BLOCKED`, like a block the tool raises.

- The macro-context cache and the retrospective evaluation memo are written through
  `atomic_write_text`: the macro cache used a fixed `.tmp` name, so two tickers in
  one region raced on the same temp file. `LLM_ATTEMPTS` now caps an explicit
  `max_attempts` as well (an explicit 3 under the default 2 made three calls); a
  caller's count can only lower it. `budget_timeout_class` and
  `request_cap_override` are declared once on `DispatchBudgetFields` instead of in
  each of the three transports. The MCP total attempt count is named
  `_MAX_ATTEMPTS`, the memory probe slot has one owner, and the last four tests that
  released a stalled worker without joining it now do both inside their patches.

- Documentation matches the code. The README names the seats pinned to standard tier
  under `--quick` (Portfolio Manager included) and states that integration tests need
  `--run-integration` and an `INTEGRATION_`-prefixed key; the agentic-AI primer
  describes the current routing (Sync Check, fast-fail, one or two debate rounds,
  optional review seats), the BUY / HOLD / DO NOT INITIATE verdicts and the owners of
  thresholds and seats instead of copying their numbers; the `.env.example` timeout
  notes distinguish the request ceiling, invocation cap and flex floors; and the
  codebase note derives the minimum useful attempt from `MIN_USEFUL_ATTEMPT_FRACTION`.

- One `DEFAULT_WORKER_SLOTS` in `src/blocking_io.py` sizes the gap-fill, MCP
  budget-store and document-extraction worker pools (each still its own pool, so a
  stall in one never starves another), and a guard rejects the number written
  inline. pypdf extraction now shares the HTML parse's pool: it had no in-flight
  limit, so repeated stalls could pile up threads; `pdftotext` stays a bounded
  subprocess. `atomic_write_text` flushes and fsyncs before the replacement, and the
  three IBKR JSON writers (analysis index, refresh scheduler state, conid cache)
  use it instead of each repeating the temp-file pattern. Stalled-worker tests
  across the suite release and join through `tests/helpers/stalled_workers.py`.

- No new work after the deadline. `ToolExecutionService` checks the operational
  deadline before every `before()` and `after()` hook and after the last of each, so
  a hook that overran, including one that swallowed cancellation, is never followed
  by a later hook, the runner or a returned result (it used to check only before each
  `after()` hook). Articles are written only after the bounded draft step returns
  while the workflow deadline holds: `generate_article` and
  `ArticleWriter.write(output_path=...)` no longer write from inside the bounded
  work, so a late or cancellation-resistant writer writes nothing and an existing
  file is kept. Articles and the analysis report are written through one atomic
  writer, `src/atomic_files.py::atomic_write_text`; the report was a plain
  `open(..., "w")` that a failure could leave half-written.

- An official document's HTML parse, link ranking and text extraction now run as one
  worker call under the extraction allowance and an in-flight limit instead of on
  the event loop, where an 80 ms parse overran a 20 ms runner bound and stalled
  every other task. A stall, a saturated limit or a spent allowance returns
  `EXTRACTION_TIMEOUT`, a late result is discarded before inspection, and a stalled
  worker keeps its slot until it ends. `OFFICIAL_DOCUMENT_EXTRACT_TIMEOUT_SECONDS`
  now bounds the HTML parse as well as pypdf and pdftotext.

- Test infrastructure: the pytest re-collection workaround applies only where the
  fixture manager has the deferred-conftest mechanism and the keyword
  `parsefactories` signature it patches, so other supported pytest versions are
  untouched. Stalled-worker tests release and join their workers in a `finally`
  through `tests/helpers/stalled_workers.py`. Live integration tests take provider
  keys from `INTEGRATION_<NAME>` (never a pytest-env dummy) and otherwise skip naming
  the variable to set, because pytest-env overwrote the real key before the old
  capture read it, so the live memory tests would have run against the dummy key.

- The "one SDK attempt" guarantee is qualified: it means no SDK retry policy, not one
  physical request. google-genai's aiohttp path resends once after a disconnect or
  connector error inside the attempt, without re-sizing the timeout, so one Gemini
  dispatch can make two sends bounded only by the hard deadline; a request-layer
  test pins it. MCP stdio servers launch through the mcp SDK's own launcher
  (`anyio.open_process`, `cwd` honoured), outside `src/subprocesses.py` and its
  spawn guard; `docs/MCP.md` records the boundary.

- The literal-deadline guard resolves deadline callees through the name they are
  bound to (a local def, an import alias such as `enclosing_deadline as ed`, or
  `module.function`), follows constant chains across one import, flags a fixed
  value inside `max()` unless it is a floor of 0 or a computed bound, checks the
  default of every deadline-shaped parameter, and respects lexical scope (nested
  functions, loop and `with` targets no longer leak or shadow wrongly). Each missed
  pattern is planted in a test that proves it is now rejected.

- `invoke_with_rate_limit_handling` has one attempt count. `max_attempts` (default
  `LLM_ATTEMPTS`) caps every retry, rate-limit and transient alike; a hidden
  `max_attempts=3` beside `max_transient_attempts` let `LLM_ATTEMPTS=5` stop at
  three calls on repeated 503s and `LLM_ATTEMPTS=1` make three calls on repeated
  429s. Under the default `LLM_ATTEMPTS=2` a 429 now gets one retry instead of two.
  The Consultant, PM verdict recovery and APAC direct retry pass `max_attempts=1`,
  so Consultant 429s are no longer retried. The hard timeout is now
  `min(cap, remaining)`: a fixed 0.1 s floor padded a spent deadline, and dispatch
  admission already refuses an attempt below the minimum useful allowance.

- MCP: each tools/call takes one atomic budget reservation (daily and per-run, one
  SQLite `BEGIN IMMEDIATE` transaction on a bounded worker thread) immediately
  before it is sent, so concurrent callers can no longer exceed a limit, and a
  locked or broken budget store refuses the call unsent instead of stalling the
  event loop or escaping as a raw sqlite error. The count is admitted reservations,
  so it can exceed actual sends. A server's `Retry-After`, now also parsed as an
  HTTP-date, is honoured in full or the retry refused (it was capped at 5 s); a 429
  without one waits the 30 s default cooldown for its own retry as well as for
  fresh calls, and the cooldown is registered before the retry, so a fresh caller
  cannot send during it.

- The embeddings startup probe has one owner per binding. Concurrent sync and async
  callers each built a client and sent a probe; now they share one client and one
  probe, each waiting under its own deadline. A cancelled owner no longer forces a
  re-probe, a superseded binding's late probe can no longer publish, and a success
  after the probe bound is discarded. `start_blocking_call` is added to
  `src/blocking_io.py`, and probes share one in-flight slot.

- An explicit seat client timeout now reaches every request. A writer built with a
  111 s `client_timeout_seconds` sent about 298 s, because dispatch took the
  registered seat's policy from the invocation scope and never saw the model's own
  override. The override is stamped on the transport at construction
  (`request_cap_override`) and resolved at every dispatch by `policy_for_provider`:
  the model's override, then the canonical seat's, then the receiving vendor's
  ceiling, always bounded by the deadline; a shorter ceiling lowers the minimum
  useful attempt with it. OpenAI-compatible vendors now refuse a `service_tier`
  passed at invocation or through bound kwargs before sending. The
  `article_writer.create_writer_llm` seam no longer takes `timeout`, and the legacy
  writer fallback chain refuses any override other than temperature.

- An official document that content inspection blocked is no longer recorded as
  primary evidence. The block placeholder used to come back as the document text,
  wrapped in `STATUS: EVIDENCE_FOUND` with primary authority. Inspection now returns
  one enforced disposition (allowed, sanitized, blocked, inspection_failed) from
  the decision, the configured mode and the fail policy, so a warn-mode `block` is
  still allowed. A blocked document is a `BLOCKED` ledger record. An inspection that
  failed or timed out is a `FAILED` tool call: the model sees
  `TOOL_ERROR: INSPECTION_FAILED` or `TOOL_ERROR: INSPECTION_TIMEOUT` (previously
  `STATUS: INSUFFICIENT_DATA`, `REASON: INSPECTION_TIMEOUT`). Like any tool failure,
  it counts against the forensic budget and the graph tool circuit. The same rule
  applies to every tool output under fail-closed inspection, and a Tavily search
  whose inspection failed returns no result. The fail-closed placeholder carries
  the code `INSPECTION_BACKEND_ERROR` in place of the backend's exception text.

- Unit tests are offline by rule, and integration tests are opt-in. One autouse
  `_offline_boundary` fixture refuses sockets, DNS, libcurl and the embedding SDKs for
  every non-`integration` test and fails any test that tried, naming targets and
  threads, because refusal alone let leaking tests pass while waiting out timeouts.
  `integration` tests now need `pytest --run-integration` (it also gates the live SGX
  PDF canary, replacing `RUN_LIVE_EXCHANGE_PDF`). pytest-socket and its
  `--allow-unix-socket` addopt are removed, and the unused
  `src.tools.shared.fetch_with_timeout` is deleted.

- yfinance no longer retries inside the library. `configure_yfinance_defaults` pinned
  `yf.config.network.retries = 3`, so one refused or failing Yahoo request slept
  1 + 2 + 4 s inside the library, invisible to every deadline and to the source
  fallbacks that already answer a transient Yahoo failure; it now pins 0 (the library
  default) and the application owns retries. Settings classes and the `.env` reader
  take their file from one policy (`src.env_file.default_settings_env_file`): the
  IBKR and dashboard settings hard-coded `.env` and `get_env_value` parsed it
  directly, so under pytest an operator's IBKR credentials and `.env`-only values
  reached unit tests. The unit suite's remaining Yahoo, Tavily and IBKR requests are
  cut at their own seams, the process-fallback rate limiter and shared memory clients
  are reset per test, and the full suite no longer waits on yfinance backoff or the
  real Google limiter.

- A failed official-document retrieval now says why. The catch-all that collapsed
  every transport failure into `DOCUMENT_RETRIEVAL_FAILED` (11 of ~1,619 network
  attempts in saved runs: six TLS certificate failures, a server disconnect, a read
  timeout, ...) is replaced by one data-driven mapping, by exception class over the
  whole cause chain: `DOCUMENT_TLS_FAILED` (an `ssl.SSLError` anywhere in the chain),
  `DOCUMENT_DOWNLOAD_TIMEOUT` (`TimeoutError`, `httpx.TimeoutException`),
  `DOCUMENT_CONNECT_FAILED`, `DOCUMENT_PROTOCOL_ERROR` (`RemoteProtocolError`,
  `LocalProtocolError`, `ReadError`, `DecodingError`), `DOCUMENT_ALLOWANCE_SPENT`
  (refused before any client opens); the residual keeps the old name and now means a
  fault outside the network. `DOCUMENT_HTTP_ERROR` adds `HTTP_STATUS: <code>`. An
  https-to-http redirect is `DOCUMENT_INSECURE_REDIRECT` (still refused), not
  `UNAPPROVED_DOCUMENT_HOST`. `official_document_failed` is now logged for every
  failed retrieval with `reason`, `phase` (dns, connect, redirect, body, extract,
  inspect), `document_scheme`, `document_host` and `http_status`, never the path or
  query. The auditor prompt (v2.16) lists the new codes.
- One `HTTP_USER_AGENT` setting (default: the editor's former string) is sent by the
  official-document client, the editor's reference fetch and StockTwits; the
  document client sent none before, and `www.sec.gov` returned 0 successes in 262
  saved calls. EDGAR requires a declared contact: new optional `SEC_CONTACT_EMAIL`
  (no default; empty is unset) is required for sec.gov hosts in the official-document
  client only: its requests there carry `<product token> <address>`, and unset, it
  refuses sec.gov URLs before connecting with `DOCUMENT_SEC_CONTACT_REQUIRED` naming
  the setting. The address goes to no other host, and no other code path adds it
  (the editor's reference fetch and StockTwits send the shared string alone). Setting
  it makes sec.gov retrievals possible; it does not by itself establish that EDGAR
  accepts them. **Behaviour change:** StockTwits sends the shared string, not
  `TradingBot/1.0`.
- A tool call that times out or raises now leaves an evidence record
  (`execution_status="FAILED"`, `reason` `TOOL_TIMEOUT` / `TOOL_EXECUTION_FAILED`,
  requested URL, classified timeout scope) through a `failed()` observer the service
  calls at its one chokepoint, so the graph tool node, consultant, auditor and
  preflight all leave it; before, the URL was lost and the model relabelled the
  timeout `EXTRACTION_TIMEOUT` in prose. The consultant, legal-counsel and
  consultant-loop paths now return `TOOL_ERROR: TOOL_TIMEOUT` (as the graph node did)
  instead of `TOOL_ERROR: TimeoutError`. Every timeout leaves `ToolExecutionService`
  as a `_ScopedToolTimeout` (adds `not_started` and `with_inspection` scopes).

- The literal-deadline guard (`tests/test_timeout_literals.py`) follows a numeric
  constant through an import (`from m import T [as U]`, `m.T`, one re-export hop; a
  config owner's constant counts once imported elsewhere), a 3-argument
  `getattr(obj, "x", 30)` fallback, and every `src/` function whose positional or
  keyword-only parameter is deadline-shaped (`enclosing_deadline`, `require_allowance`,
  `run_program`, `check_within`, ...), derived from the signatures. It found six
  non-deadline literals in `src/` (breaker cool-off report fields, a refused-retry
  backoff record, an orphan-age filter, an auth cooldown, a consultant placeholder),
  all allowlisted with reasons; no real deadline literal remained. Test-only change.
  Residual hole: a function that returns a literal is not followed.

- One minimum-useful-attempt rule and one margin, both derived. A minimum is
  `MIN_USEFUL_ATTEMPT_FRACTION` (new, 0.04, at most 0.5, chosen not measured) of the
  plane's configured allowance for one attempt: the LLM per-request ceiling (or the
  invocation cap when shorter) before any flex floor, the embedding call timeout, the
  MCP tool-call timeout (2.4-12 s, 1.2 s, 1.8 s at the shipped defaults; 4.8 s for a
  full-mode OpenAI request, 12 s at most). `LLM_MIN_USEFUL_ATTEMPT_SECONDS`, `MEMORY_EMBEDDING_MIN_USEFUL_SECONDS`
  and `MCP_RETRY_MIN_USEFUL_SECONDS` and their cross-field validators are removed (all
  were branch-only). The LLM budget policy's margin is `deadline_margin(cap)`, the tool
  plane's rule. **Behaviour change:** a quick-mode LLM request's minimum moves from 5 s
  to 0.04 x its attempt allowance (2.4 s at the 60 s standard quick cap, 4.8 s full
  mode), and the old `cap / 2`
  ceiling is now the field's 0.5 bound; margins are unchanged.
- Settings are read, never defaulted at the call site. Every `getattr(settings, "<field>", <literal>)`
  is now a plain attribute read, one `resolve_settings` in `src/config.py` replaces four private
  "explicit settings or the live singleton" helpers, and a hygiene test scans `src/` for the shape.
  **Behaviour change:** the BUY stability gate (`buy_stability_enabled`) and the review/regional
  independence requirements defaulted OFF for any config object lacking the field while `Settings`
  ships them ON; such an object now raises `AttributeError` instead of silently disabling the gate.
  `RETROSPECTIVE_MAX_EVALUATIONS_PER_RUN` has one default (the setting; the duplicate module constant
  is gone). The Consultant's node deadline is measured on the shared deadline clock, shortened by the
  run deadline, and bound as the LLM enclosing deadline for each request (never across tools).
- LLM retries have one owner, the runtime loop. Every client sends exactly one SDK
  attempt per request on every vendor; the transport classes enforce it, and no
  constructor or factory takes an attempt count any more
  (`SeatExecutionPolicy.transport_attempts` and its deprecated alias
  `sdk_max_retries` are removed). **Behaviour change:** the macro-context,
  injection-judge and signal-processor seats on OpenAI, compatible vendors or
  Anthropic were built with `LLM_ATTEMPTS` SDK attempts while also retried by the
  loop, so a failing call sent up to `LLM_ATTEMPTS²` requests (4 at the shipped
  2), each SDK re-send past the deadline it was sized from; it now sends at most
  `LLM_ATTEMPTS`. Google calls, the health check and every runtime-owned seat are
  unchanged. The injection judge's seat is renamed "Prompt Injection Judge" (the
  label its token accounting already used), and it and the signal processor now
  resolve their seat's budget instead of the generic one. Dispatch records no
  longer carry `configured_attempts`.
- Memory: `--no-memory`, and a graph built without a ticker, construct no memory
  object (the five shared `legacy_*` memories are gone, and with them a Chroma
  client, a startup probe and a cross-ticker read). The embeddings startup probe
  starts only when its full `MEMORY_EMBEDDING_HEALTHCHECK_TIMEOUT_SECONDS` fits
  before the run deadline (otherwise it is deferred, uncached), and the CLI runs
  it off the event loop before building memories. Embeddings take their rate
  limiter from the embedding provider (`limiter_for_binding`), so OpenAI
  embeddings are no longer paced at Google's RPM; the unthrottled setup fallback
  and `embedding_rate_limiter_fallback` event are removed.
- Rate limits: every process limiter starts with one token, so a process's first
  request no longer waits about 5 s at 15 RPM; the pace afterwards is unchanged.
  The Google ceiling is read in one place (`configured_google_rpm`), so the
  unscoped fallback honours `GEMINI_RPM_LIMIT` under the legacy schema, and a
  run-scoped xAI binding is now throttled at `XAI_RPM_LIMIT`.
- MCP call budgets are now enforced by `MCPRuntime` itself: each attempt is
  admitted against `daily_call_limit` / `per_run_limit` before a session opens,
  and each tools/call is counted as it is sent. `MCPBudgetHook` is gone; it
  counted in `after()`, so a call that timed out, failed, or whose result the
  tool plane discarded was never counted, and a retry counted once for two
  sends. **Behaviour change:** a retried call now uses two of `per_run_limit`,
  and a retry the budget cannot cover is refused (`details.retry ==
  "refused_budget"`). The dead `MCPRuntime.call_tool`, `list_tools_for_scope` and
  `src/mcp/catalog.py` were removed.
- An MCP tool the server does not list is now refused before the call is sent
  (`failure_kind: "config"`, `details.refused == "tool_not_listed"`), and the
  listing follows `nextCursor` within its one bound. Before, the call was sent
  anyway and the SDK re-listed after it under the stale session bound, then
  skipped output validation. The `x-mcp-header` filter is documented as inert in
  production, where `initialize()` negotiates only legacy protocol versions.
- Every child process under `src/` now starts through `src/subprocesses.py`, on
  CPython's `posix_spawn` path (absolute executable, `close_fds=False`, no
  `cwd`), with kill-and-reap on every abnormal exit. pdftotext and the run
  fingerprint's git probe forked before, which in the threaded analysis process
  risks a crash in Network.framework's atfork handler. The reap log events are
  now `subprocess_reap_pending/failed/cancelled` (were
  `official_document_subprocess_reap_*`), with a `pid` field.

- The official-document tool now shares its allowance by fractions instead of
  absolute reserves: the download may use `OFFICIAL_DOCUMENT_DOWNLOAD_FRACTION`
  (0.55) of it, and `OFFICIAL_DOCUMENT_INSPECTION_FRACTION` (0.25) of the rest is
  held back from PDF extraction for the content inspection of the evidence. These
  replace `OFFICIAL_DOCUMENT_DOWNLOAD_TIMEOUT_SECONDS`,
  `OFFICIAL_DOCUMENT_DOWNLOAD_TOTAL_TIMEOUT_SECONDS` and
  `PDF_EXTRACTION_RESERVE_SECONDS`, added earlier in this release. A runner at or
  below the old 20 s reserve fetched no document at all; any runner value now
  works. **Behaviour change:** under the 45 s runner the download gets about 23.6 s
  (was 25 s) and extraction about 14.5 s (was 20 s); inspection, which had no
  share, gets the rest, and evidence it cannot finish inspecting is reported as
  `INSPECTION_TIMEOUT` instead of the runner's bare `TOOL_TIMEOUT`. Both defaults
  are chosen, not measured. Tavily and the document tool now inspect through one
  bounded `InspectionService.check_within`.

- Operational work inside a tool call now sizes itself against what the tool
  runner has left, not only the caller's deadline: Tavily, market data, FX,
  filings, the editor's reference fetch and every blocking call read one
  operational deadline (MCP and the official-document tool already did). Before,
  a second search inside a 45 s runner could be given its full 30 s and be cut off
  as a bare `TOOL_TIMEOUT`. Every blocking call (`run_blocking_call` and its sync
  twin) now applies that clamp itself and refuses, before any thread starts, once
  the deadline has passed. Work inside a hard cap now gets a deadline kept back by
  a margin, the new `DEADLINE_MARGIN_FRACTION` (5 % of the cap, at most
  `LLM_DISPATCH_MARGIN_SECONDS`), so its own typed timeout fires before the cap's
  cancellation instead of racing it. **Behaviour change:** an MCP call or document
  extraction that runs to the runner's end now returns its own timeout result a
  margin early rather than `TOOL_TIMEOUT`.
- The combined tool scope is now derived: `TOOL_EXECUTION_TIMEOUT_SECONDS` plus the
  new `TOOL_HOOKS_TIMEOUT_SECONDS` (75 s; 120 s in all, unchanged). It replaces
  `TOOL_EXECUTION_WITH_INSPECTION_TIMEOUT_SECONDS`, added earlier in this release,
  and its startup check: raising the runner no longer requires raising a second
  setting. Tool-call timeouts are now logged once with the scope that produced
  them (`runner`, `before_hooks`, `after_hooks`, `hooks_allowance_spent`,
  `with_inspection`, `not_started`); a hook's own timeout was reported as the
  combined scope. The graph tool node no longer wraps the service in a second,
  identical bound, so cancellation reaches a running tool one loop turn sooner.

- Tests: subdirectory conftest fixtures now survive pytest 9.1 re-collecting a
  directory. Naming a root-level test file between two files of one subdirectory
  used to drop that subdirectory's conftest: named fixtures errored and autouse
  isolation was skipped silently. Unit tests can no longer send embedding requests:
  the memory startup probe reached Google 57 times in one full run. A new test runs
  the real CLI flow under an already-spent run deadline and checks that it fails
  closed. The README macro-context test now asserts the current diagram.
  No production code changed.
- An OpenAI, OpenAI-compatible or Anthropic client that its callers invoke directly
  is now built with `LLM_ATTEMPTS` total attempts, like Google already was; it was a
  fixed 3. **Behaviour change:** at the shipped `LLM_ATTEMPTS=2` such a client makes
  two requests instead of three (both construction schemas). Seats that pin their
  attempts keep them (health check, writer, macro classifier: one). The OpenAI
  request ceiling is the new `OPENAI_REQUEST_CAP_SECONDS` setting (120 s, unchanged).
  The APAC client ceiling is now `APAC_NODE_TOTAL_TIMEOUT_SECONDS` (240 s, unchanged)
  and the same value applies inside an invocation, where it was 300 s; lowering or
  raising the node total now moves both.
- MCP steps now stay inside the time left, and read the tool runner's deadline as
  well as the caller's (a nested MCP call no longer sizes its handshake and call
  against the full settings while the runner has less). The tools/list catalogue
  is bounded by what the handshake left (it got the session's full bound again).
  Every tool call now lists first, inside that bound, so the SDK validates the
  result against a cached schema instead of listing *after* the call under the
  stale bound; a successful call sends the same requests in a different order, and
  a failed one costs one extra listing. The retry backoff is admitted like an LLM
  retry: shortened to fit, refused when less than the new
  the minimum useful attempt would not be left, and re-checked after the
  sleep. A refused retry raises the last attempt's error with `retryable: false`
  and `details.retry == "refused_deadline"`; a 429 still registers its cooldown.
- An official-document download now has a total deadline,
  `OFFICIAL_DOCUMENT_DOWNLOAD_TOTAL_TIMEOUT_SECONDS` (35 s), around the whole
  download, redirects and DNS checks included; httpx's timeout bounded only each
  phase, so a trickling body could run on. The download also leaves PDF extraction
  its reserve, `PDF_EXTRACTION_RESERVE_SECONDS` (unset means
  `OFFICIAL_DOCUMENT_EXTRACT_TIMEOUT_SECONDS`, 20 s): it may use what the caller and
  the 45 s tool runner have left, less the reserve, so under the runner a download
  gets at most 25 s (was up to 35 s per phase), and with less than the reserve left
  no download starts. Extraction backends size to the tool runner's remaining time
  too, and a skipped pdftotext logs `official_document_extraction_skipped`. A
  pdftotext child cut short by cancellation is now killed and reaped (it was left
  running); the reap is shielded and bounded by `SUBPROCESS_CLEANUP_TIMEOUT_SECONDS`
  (5 s).
- The `--article` workflow now has its own deadline, `ARTICLE_WORKFLOW_TIMEOUT_SECONDS`
  (900 s, about twice the longest measured workflow), around draft, review and
  revisions, and the writer runs asynchronously through the runtime loop instead of a
  synchronous call that blocked the event loop. Writer clients send one SDK attempt
  per request (was three Anthropic requests of `API_TIMEOUT` each, per tier); the loop
  retries up to `LLM_ATTEMPTS`, each attempt capped at `API_TIMEOUT` in both run modes.
  A writer fallback tier is no longer built once the allowance cannot fit it. When the
  allowance runs out after a draft exists, the draft is saved for review with
  `editor_review_error` and the new `workflow_deadline_exceeded` flag, never as an
  approved article; a step that overruns is discarded, never written. The writer's
  fact-check search now uses the async Tavily entrypoint. `create_writer_llm` honours
  the temperature, timeout and attempts it is given, and the writer's configured
  temperature now also reaches the fallback tiers (the provider-scoped Gemini floor
  wrote at 0.1, not the writer's 0.7). `generate_article` is now a coroutine.
- A memory embedding attempt now has one deadline covering the rate-limiter wait
  and the request (the wait had no bound, and the request got a fresh
  `MEMORY_EMBEDDING_CALL_TIMEOUT_SECONDS` after it). The unthrottled fallback is
  taken only when no rate limiter exists; it no longer re-sends the request after
  a provider error or timeout. Retry backoff is clamped to the caller's deadline,
  retries stop when less than the minimum useful attempt
  (`MIN_USEFUL_ATTEMPT_FRACTION` of the call timeout) is left, and the last attempt's own error is raised instead of a `RetryError`.
  OpenAI embeddings send one SDK attempt per request (was three).
- A model invoked outside the runtime loop is admitted and sized against an
  enclosing deadline when one is bound; with none, nothing changes.
- The portfolio manager's macro-event classifier invokes through the runtime loop
  (bounded attempts, owned retries, one SDK attempt per request) instead of an
  unbounded synchronous call.
- `TOOL_EXECUTION_WITH_INSPECTION_TIMEOUT_SECONDS` now bounds every tool call made
  through the tool service, its `before()`/`after()` hooks included, not only graph
  tool calls: the Consultant, Auditor, Legal Counsel, article editor and evidence
  preflight had no bound on their hooks. A tool call made from inside another (the
  Consultant's FMP MCP spot check) gets only what the outer runner has left. A
  result whose allowance runs out before or during its `after()` hooks is
  discarded, never returned or recorded in the evidence ledger, and a call whose
  allowance is already spent now runs no hook at all (a call a `before()` hook
  would have blocked times out instead). The async Tavily entrypoints now bound
  search and inspection with one deadline, as the sync path did: a late search
  result is discarded before inspection, and an inspection that overruns returns
  nothing rather than the uninspected payload.

- Enforce reliable financial-health failures through the existing verdict correction, preserving the canonical deep-value/current-ratio exception and active-tender exemption; project the same health assessment into saved policy, PM context and reports.
- Close every completed Junior acquisition with its named terminal message, including empty or malformed failures, preserving current failure provenance and preventing a redundant blocked tool round.

### Changed

- Every OpenAI, OpenAI-compatible and Anthropic client is now sized at dispatch
  for the vendor it names. Before, only the flex-tier OpenAI client and Gemini
  were. The canonical OpenAI adapter without a failover, the DeepSeek, z.ai,
  Moonshot and xAI clients, the Consultant's alternate transport, the legacy
  Consultant, Auditor, Editor and writer-fallback factories, the APAC factory
  and both Anthropic writers sent their construction-time timeout. Every
  constructor now passes its provider explicitly, and a missing one is an error.
  **Behaviour change inside an invocation:** a compatible vendor is sized against
  its `OPENAI_COMPATIBLE_CLIENT_TIMEOUT_SECONDS` ceiling instead of the 120 s
  OpenAI cap, and its model-failover copy no longer sends `service_tier`. The
  Consultant's alternate transport on a compatible base is built with that
  ceiling too. Requests outside an invocation keep their built timeout. Streaming
  on these clients and on Gemini is refused: `.stream()` runs as a sized
  non-streaming call, and `streaming=True` or a `stream` argument raises.
- The per-ticker pipeline watchdog is now a hard ceiling inside the analysis.
  `STAGE1_TICKER_TIMEOUT_SECONDS` (600) and `STAGE2_TICKER_TIMEOUT_SECONDS` (2400)
  are settings that `scripts/run_pipeline.sh` reads through
  `python -m src.config_print`. An exported value still wins. A value that is not
  a positive whole number now stops the pipeline. Each child receives an absolute
  `PIPELINE_TICKER_DEADLINE_EPOCH` and binds it, less the new
  `PUBLICATION_RESERVE_SECONDS` (30), around the analysis. **Behaviour change
  under the pipeline:** an LLM request, retry or backoff that cannot finish
  30 s before the watchdog fires is refused rather than sent. The injection
  judge is included and falls back to its heuristic decision. Every
  operational deadline that clamps to its caller now shrinks to the ticker's
  remaining time, which until now was a no-op outside the APAC node. A ticker
  that runs out of time now ends as an unpublishable `ANALYSIS FAILED` report
  (`no_analysis`) rather than a SIGTERM (`child_exit_124`) more often.
  Report rendering and persistence run outside the cap, on the reserve. A
  startup `deadline_ladder_truncated` log reports when the configured retry
  ladder cannot fit, which is always true in `--quick` at the shipped defaults.
  Manual runs are unchanged. `tests/test_timeout_literals.py` now rejects a
  numeric literal passed to a deadline parameter outside the configuration
  owners.
- Attempt counts now mean total attempts on every vendor. `API_RETRY_ATTEMPTS` is
  renamed `LLM_ATTEMPTS` (the old name is still read; the new one wins) and
  `SeatExecutionPolicy.sdk_max_retries` is renamed `transport_attempts` (the old
  field still works, translated per vendor). Under the application retry loop each
  request sends one SDK attempt and the loop owns retries, defaulting to
  `LLM_ATTEMPTS`. **Behaviour change on OpenAI and Anthropic:** at the same
  configured value, clients make one fewer request (the writer and its
  fallbacks: 3 requests instead of 4; the editor now sends one per request under
  the loop, below);
  the Auditor, APAC specialist and the Consultant's alternate transport go from
  2-4 SDK requests per call to one; the OpenAI-bound Health Check makes exactly
  one request instead of two. The loop's transient-retry budget, previously a
  fixed 2, now defaults to `LLM_ATTEMPTS`: at a configured 3, a full-mode call can
  make three attempts of up to the 1,350 s flex-floored cap instead of two, until
  an enclosing deadline bounds it.
- The Article Editor's review, the retrospective lesson and the semantic judge now
  run under the runtime retry loop, like the graph seats: each call is capped by
  its seat's budget, every request is sized from the remaining allowance, and
  retries are owned by the loop (one SDK attempt per request; these seats are now
  built with one attempt on OpenAI and Anthropic). **Behaviour changes:** the
  editor review was uncapped and is now capped per call at the review-plane budget
  (`CROSS_CHECK_QUICK_LLM_CALL_HARD_TIMEOUT_SECONDS`, 180 s, in `--quick`;
  `LLM_CALL_HARD_TIMEOUT_SECONDS` in full mode), and a timeout degrades the review
  to REVISE as any other editor failure does; the editor seat moves from the
  editorial to the cross-check budget class. The `--quick` retrospective lesson
  and judge caps follow their class (`QUICK_LLM_CALL_HARD_TIMEOUT_SECONDS`, 60 s)
  instead of the removed `min(LLM_CALL_HARD_TIMEOUT_SECONDS, 120)` clamp, which no
  longer exists; `LLM_CALL_HARD_TIMEOUT_SECONDS` is read only in full mode and is
  no longer reported as quick-mode drift. A refused judge response is still a
  HARD_FAIL. The analyst structural retry reads its overall budget from the same
  resolver (unchanged values). Consultant budgets are read from config at call
  time, and their quick/full choice follows the run config rather than the
  graph-build flag. Inside an invocation, a seat's explicit client timeout (the
  Health Check's 10 s) is now the request ceiling, and a request takes its
  ceiling from the vendor that receives it.
- The two tool deadlines are now settings, defaults unchanged:
  `TOOL_EXECUTION_TIMEOUT_SECONDS` (45 s, the tool runner) and
  `TOOL_EXECUTION_WITH_INSPECTION_TIMEOUT_SECONDS` (120 s, a graph tool call
  including the content-inspection hooks that run after the runner's bound is
  released). Configuring the second below the first is rejected at startup. Both
  shrink to a caller's remaining enclosing deadline when one is bound, and a call
  whose caller deadline has already passed fails as a timeout without starting
  the tool. **Behaviour change:** the graph tool node's outer scope is now
  deadline-only (`run_with_hard_timeout`) rather than `asyncio.wait_for`, so a
  stalled inspection hook that ignores cancellation returns `TOOL_TIMEOUT` on time
  and its late, uninspected result is discarded instead of being returned.
- The market-data deadlines are now settings, one per scope, defaults unchanged:
  the whole metrics fetch (`MARKET_DATA_AGGREGATE_FETCH_TIMEOUT_SECONDS`, 30 s),
  each parallel source (`MARKET_DATA_PER_SOURCE_TIMEOUT_SECONDS`, 15 s), a vendor
  HTTP request (`MARKET_DATA_VENDOR_HTTP_TIMEOUT_SECONDS`, 10 s), the lightweight
  EODHD lookup (`MARKET_DATA_ANCHOR_LOOKUP_TIMEOUT_SECONDS`, 5 s), ticker
  resolution (`MARKET_DATA_TICKER_RESOLUTION_TIMEOUT_SECONDS`, 10 s), price
  history (`MARKET_DATA_PRICE_HISTORY_TIMEOUT_SECONDS`, 15 s), each company-name
  source (`COMPANY_NAME_LOOKUP_TIMEOUT_SECONDS`, 5 s), a live FX rate
  (`FX_RATE_FETCH_TIMEOUT_SECONDS`, 8 s) and the Consultant's spot check
  (`CONSULTANT_SPOT_CHECK_TIMEOUT_SECONDS`, 8 s). Each is read at the call and
  shrinks to a caller's remaining enclosing deadline when one is bound; a call
  whose caller deadline has already passed takes its existing timeout fallback
  without starting.
- The research deadlines are now settings, one per scope, defaults unchanged: a
  Tavily web search (`TAVILY_SEARCH_TIMEOUT_SECONDS`, 30 s), a Tavily extract
  (`TAVILY_EXTRACT_TIMEOUT_SECONDS`, 30 s), the gap-fill search inside a metrics
  fetch (`GAP_FILL_SEARCH_TIMEOUT_SECONDS`, 5 s), the wait for a DuckDuckGo search
  (`DDG_SEARCH_TIMEOUT_SECONDS`, 8 s) and its client's HTTP timeout
  (`DDG_HTTP_TIMEOUT_SECONDS`, 5 s), the official-filings lookup
  (`OFFICIAL_FILINGS_LOOKUP_TIMEOUT_SECONDS`, 20 s), one filing fetcher
  (`FILING_FETCH_TIMEOUT_SECONDS`, 30 s), an official-document download
  (`OFFICIAL_DOCUMENT_DOWNLOAD_TIMEOUT_SECONDS`, 35 s), its DNS check
  (`OFFICIAL_DOCUMENT_DNS_TIMEOUT_SECONDS`, 5 s) and PDF extraction
  (`OFFICIAL_DOCUMENT_EXTRACT_TIMEOUT_SECONDS`, 20 s), and the editor's reference
  fetch (`REFERENCE_FETCH_TIMEOUT_SECONDS`, 10 s). Each is read at the call and
  shrinks to a caller's remaining enclosing deadline when one is bound, including
  an explicit per-call Tavily timeout and the sync Tavily worker's shared
  deadline; a call whose caller deadline has already passed takes its existing
  timeout fallback without starting.
- The MCP deadlines are now settings, one per scope, each handed to the SDK's own
  parameter: the HTTP client's per-phase timeout (`MCP_HTTP_TIMEOUT_SECONDS`, 5 s,
  unchanged: it is the httpx2 default the client already ran on because it set
  none), the session read timeout on the initialize handshake and a tools/list
  catalog request (`MCP_INITIALIZE_TIMEOUT_SECONDS`, 30 s), and the read timeout
  on one tool call (`MCP_TOOL_CALL_TIMEOUT_SECONDS`, 45 s). Each is read when the
  session opens or the call is sent and shrinks to a caller's remaining enclosing
  deadline; a spent allowance opens no session and sends no call, and fails as a
  non-retryable `transport` error. **Behaviour change:** the handshake, the
  catalog request and the tool call had no protocol-level bound, so a stdio
  server that never answered, or an HTTP server trickling bytes, could hold a
  session for as long as its caller allowed (the Consultant's path is capped by
  the 45 s tool runner, which still fires first, so its outcome is unchanged; a
  direct `MCPRuntime.call_tool` or `list_tools_for_scope` caller had no bound). A
  timed-out request (`REQUEST_TIMEOUT`, -32001) now classifies as a retryable
  `transport` error rather than `protocol`.
- The operational deadlines are now settings, one per scope, defaults unchanged:
  the health check's LLM probe (`HEALTH_CHECK_CONNECTIVITY_TIMEOUT_SECONDS`,
  15 s), the embeddings startup probe
  (`MEMORY_EMBEDDING_HEALTHCHECK_TIMEOUT_SECONDS`, 15 s), one embedding request
  (`MEMORY_EMBEDDING_CALL_TIMEOUT_SECONDS`, 30 s), closing a Google GenAI client
  at shutdown (`GENAI_CLIENT_CLOSE_TIMEOUT_SECONDS`, 2 s) and the Langfuse flush
  (`LANGFUSE_FLUSH_TIMEOUT_SECONDS`, 2 s). Only the embedding request shrinks to
  a caller's remaining enclosing deadline; a spent allowance sends no request
  and is not retried. The health check, the startup probe (whose failure is
  cached for the process) and the shutdown paths are not clamped.
- Six more market-data deadlines are settings, defaults unchanged, each clamped
  to a caller's enclosing deadline: each yfinance ownership section
  (`OWNERSHIP_SECTION_TIMEOUT_SECONDS`, 8 s), a StockTwits request
  (`STOCKTWITS_HTTP_TIMEOUT_SECONDS`, 10 s), the retrospective's price fetch
  (`RETROSPECTIVE_PRICE_FETCH_TIMEOUT_SECONDS`, 15 s), the advisory market note
  and price snapshot (`MARKET_CONTEXT_FETCH_TIMEOUT_SECONDS`,
  `PRICE_SNAPSHOT_FETCH_TIMEOUT_SECONDS`, 10 s each) and the report-header name
  lookup (`OUTPUT_COMPANY_NAME_TIMEOUT_SECONDS`, 6 s; each candidate symbol uses
  `COMPANY_NAME_LOOKUP_TIMEOUT_SECONDS`).
- **Fixed:** the synchronous report-header company-name lookup ran each
  candidate on a `ThreadPoolExecutor` worker. The caller returned on time, but
  executor workers are joined at interpreter exit, so a stalled Yahoo read could
  hold the process open after the report was written, and each stalled call
  left another worker. It now runs on a slot-capped daemon worker
  (`run_blocking_call_sync`, two slots); a full cap falls back to the ticker.
- The APAC specialist's 240 s is now its node total
  (`APAC_NODE_TOTAL_TIMEOUT_SECONDS`), shared by the first call and the direct
  retry, rather than a fresh 240 s per call.
- Seat timeout budgets are chosen by a `TimeoutClass` on each registered seat
  rather than by matching display names; new knobs `FLEX_QUICK_ATTEMPT_FRACTION`
  (0.4), `MIN_USEFUL_ATTEMPT_FRACTION` (0.04) and `LLM_DISPATCH_MARGIN_SECONDS` (2).
- Treat generic consultant conditions, numerical discrepancies, and suspicious
  OCF/NI ratios as zero-weight review findings rather than issuer-risk penalties.
- Block BUY when the Consultant reports MAJOR_CONCERNS: unresolved material
  analysis flaws withhold initiation authority without adding issuer-risk points.
  Growth exceptions permit at most HOLD, with the active-tender override preserved.

### Fixed

- Size every LLM request at dispatch from the invocation's absolute deadline
  instead of a timeout fixed on the client at construction. The flex-attempt
  timeout (24 s quick, 900 s full) no longer rides along on the standard re-issue,
  on calls sent at standard after flex degrades, or on the model-failover copy;
  a request with less than `MIN_USEFUL_ATTEMPT_FRACTION` of the cap left is refused before
  any transport call, and each call attempt now records its `dispatches`
  (tier, model, timeout sent, remaining allowance, configured attempts, outcome).
  The full-mode latency fallback to standard is reachable again.
- Pin the quick-mode Portfolio Manager to the standard Gemini tier again: its
  2026-10 `504 DEADLINE_EXCEEDED` failures were the 24 s quick flex-attempt deadline,
  enforced server-side and inherited by the standard re-issue and failover. The Google
  adapter now honors the run-scoped `--quick` retry clamp, so quick PM, Senior
  Fundamentals and Analyst Retry calls drop from 3 to 2 SDK attempts when
  `API_RETRY_ATTEMPTS` is 3; explicit per-seat retry policies are unchanged.
- Bound the caller's wait in the gap-fill company-name lookup (previously
  `wait_for(to_thread)`, which waits on an uncancellable thread) and the sync Tavily
  search (whose `with ThreadPoolExecutor` exit joined the stalled worker). Each now
  runs on a daemon thread capped by an `InFlightLimit`, so stalls cannot pile up
  threads or hold interpreter exit, and saturation returns the usual fallback at once.
  The Tavily worker shares the caller's absolute deadline: a late search result is
  discarded before inspection, and inspection gets only the remaining time. Neither
  client exposes a transport timeout, so a stalled request still runs until it ends or
  the process exits. The `to_thread` guard now resolves asyncio aliases.
- Keep EPS rubric lineage on eligible per-share bases, honor verified majority
  ownership ahead of unrelated independence text, preserve Junior originals from
  inspected ingress, and bind one-cell-per-line comparative results tables.
- Recognize complete half-year labels, abbreviated English dates, retained Japanese
  fiscal Q1 context, and equivalent currency-bound reporting-unit declarations while
  preserving comparative table and earnings-scope checks. Keep monetary amount
  grammar separate from reporting units.
- Freeze additional source-adjudicated latest-results regressions and distinguish
  incomplete claims, retained unbound evidence, citation authority, and attempted
  acquisition in offline replay. Report flex fallbacks and degradation episodes in
  batch summaries without changing routing or pricing.

- Reject failed evidence and failed result envelopes before ownership/capacity
  normalization; bind envelope authority to its declared source and apply the same
  checks to preflight fallback while preserving valid neighboring results.
- Bind ownership percentages to a shareholder row or ownership clause instead of
  unrelated amounts elsewhere in the cited record; preserve unknown control.
- Bind prior-first comparative statements and declared percentage columns without
  relaxing period, currency, scale, or earnings-scope checks. Preserve recorder-owned
  source authority during offline replay and record explicit evidence truncation.
- Give complete, unsupported quick-mode latest-results claims one budgeted fetch of
  their approved citation, with no repeat of an already attempted source.
- Preserve verified independent initiation failures across growth exceptions, and
  withhold deterministic HOLD upgrades when English-language coverage or measured
  liquidity is unresolved. Render and save the same gate assessments, distinguish
  indeterminate growth inputs, and label risk-zone defaults before initiation gates.
  Missing health withholds growth-exception authority without implying verified
  low health.
- Record effective standard Google service-tier attribution without changing the
  deliberate standard transport policy or flex fallback pricing.

- Recognize supported currency-code and compound-symbol prefixes, signed and
  accounting amounts through the shared monetary grammar. Preserve minor currency
  denominations in FX resolution, caches, OCF comparisons, and reconciliation;
  share stable currency spellings with chart formatting. Extract market-cap values
  after parenthesized date qualifiers. Accept explicit ISO currency amounts in
  either order for capital exposure, and stop free-text scanners at invalid
  monetary candidates instead of substituting later comparative amounts.
- Relay Junior Fundamentals data from its inspected tool results in code, retaining
  useful partial results and exposing the same final artifact to prompt checks and
  baseline capture. Bound currency and sign/bracket prefixes in the amount scanner.
  Preserve legacy HTML extraction and render canonical initiation gates, measured
  values, uncertainty, and the hard-fail aggregate in reports.
  Share the Portfolio Manager's structural and semantic correction allowance;
  require eligible decision facts or active gates and retain publication failure
  when the final policy trace is invalid.
- Bind observed comparative issuer statements to exact metric rows, period columns,
  currency, reporting scale, and earnings scope. Preserve rejection across adjacent
  tables and ambiguous layouts; persist finite normalization and withheld-field
  reasons in the retained report.
- Remove legal-response fragments from parser-fallback logging.
- Parse currency suffixes separately from monetary magnitudes, preventing KRW,
  BRL, and MYR amounts from creating false cash-conversion warnings. Share monetary
  token recognition across metric extraction, OCF corroboration, filing conflicts,
  capital-exposure scaling, and web-extracted market capitalization; preserve signed
  K/M/B/T and spelled magnitudes. Reject unsupported units and ambiguous grouping,
  including grouped-number backtracking to a smaller numeric prefix.
- Bind capital claims to exact source URLs and scaled same-currency amounts; retain
  unresolved absence coverage ahead of accounting qualifiers. Include inspected,
  successful Legal Counsel follow-up evidence in normalization. Align the legal
  prompt with unresolved no-findings coverage until absence is code-verifiable.
- Preserve both debate sides and canonical versus advisory score explanations in
  consultant context. Keep missing evidence distinct from issuer risk in review
  and portfolio-manager prompts.
- Recover invalid full-mode legal JSON once through the existing text-only recovery
  seat, retain cost attribution, validate regulatory-risk entry shapes, and fail
  closed when recovery is unsuccessful or changes retained legal assessments.
- Leave more output room for Value Trap reasoning, retrieve unfetched approved
  guidance documents when extraction is unavailable, and report evidence coverage
  gaps separately from publication status in the batch health digest.

## [4.0.0] - 2026-09-16

### Added

- **Auditable analysis controls** — Canonical decision-policy records, evidence
  promotion, tool-execution accounting, research budgets, recovery telemetry, and
  decision-evidence reporting now make verdict changes and degraded runs traceable.
- **Broader, safer screening** — Add official BVB Regulated Market coverage and RON
  handling, plus resilient current feeds for ASX, JPX, Nasdaq, XETRA, and paginated
  StockAnalysis sources.
- **Optional balanced debate handoffs** — Full runs can pass bounded Round-1 rationale
  capsules symmetrically to both second-round researchers and Research Manager.
- **Portable agent guidance** — Add public, secret-safe Codex and Claude guidance,
  focused repository skills, and automated metadata/dependency guards.

### Changed

- **Fail-closed outcome semantics** — Eligibility rejection, analysis failure, and
  publishability are now distinct canonical states; deterministic growth, liquidity,
  legal-coverage, and evidence rules remain authoritative across provider choices.
- **Bounded provider-neutral execution** — Shared research loops, structural recovery,
  timeout classification, token reserves, and cost attribution now follow code-owned
  contracts in quick and full modes.
- **More efficient batch runs** — The pipeline reuses recent complete same-mode
  artifacts, excludes US-domiciled secondary listings, deduplicates issuers, and keeps
  candidate order stable across restarts.
- **Reviewed dependency refresh** — Update the LLM, orchestration, data, observability,
  and development stacks while deliberately retaining MCP 1.x pending its incompatible
  client and wire-format changes.

### Fixed

- **Portfolio and refresh integrity** — Quarantine unresolved broker identities,
  preserve incomplete-inventory accounting, withhold unsafe BUY/ADD authority, and fix
  refresh fairness, backoff, settled-rejection, closed-position, and FX edge cases.
- **Source and artifact completeness** — Enabled exchange-source failures now stop
  publication before replacing prior output; malformed structured responses, missing
  required evidence, and incomplete artifacts fail visibly instead of appearing valid.
- **Regression diagnostics** — Longitudinal comparisons now separate quick from full
  runs and flag lost contracts, evidence, publishability, or material cost efficiency.

### Security

- Pin fixed releases for `cryptography`, `h2`, `pyasn1`, `pypdf`, `starlette`, and
  `urllib3`, including protections for the Auditor's externally sourced PDF path.

## [3.15.0] - 2026-08-18

### Added

- **Provider-neutral LLM seats** — Named application seats now resolve through
  independently configurable base, review, and regional provider planes, with
  validated capabilities, explicit independence waivers, migration tooling, and
  provider-accurate model/tier cost attribution.
- **Retrospective audit tooling** — A read-only evidence/disposition report and
  bounded, archive-aware evaluation make prospective learning coverage and any
  legacy-sweep value observable before prices or lessons are written.

### Changed

- **Evidence-scoped lessons learned** — Retrospectives recover stored bear
  evidence from source artifacts and render deterministic records from an
  explicit disposition policy; only verified, market-contextual outcomes can
  become injectable lessons.
- **Consultant and evidence workflows** — Consultant tools now use bounded
  concurrent batches, while evidence disposition, output validation, and shared
  parser contracts make incomplete evidence explicit rather than inferred.

### Fixed

- **International pricing and portfolio decisions** — Preserve minor-unit
  denominations (including GBp/GBX), compare prices only when currencies and
  scales are compatible, and reject incoherent derived price levels.
- **Runtime resilience and reporting** — Correct flex fallback/cache behavior,
  provider failure classification, optional MCP credentials, degraded-run
  status, and the Auditor/IBKR execution paths.
- **Cross-consumer parser drift** — Centralize repeated report patterns and
  protect key verdict, metric, risk, macro-alert, and ticker parsers with
  regression contracts.

## [3.14.0] - 2026-07-30

### Added

- **Canonical evidence infrastructure** — Registered tool payloads enter through a typed, fail-closed contract before LLM rendering (`src/tooling/evidence_recorder.py`, `src/claim_policy.py`); material claims retain source and period provenance; positive score credits require transitive fact lineage (`src/score_lineage.py`); corrected facts and scores project from one canonical snapshot (`src/analysis_snapshot.py`); and invalid analyses or unapproved articles are retained as diagnostic/draft artifacts rather than silently discarded.
- **Capital-structure evidence checks** — New deterministic checks on capital-structure claims (`src/agents/capital_structure.py`) feeding the Legal Counsel prompt and red-flag detector.
- **Management-guidance & earnings-durability classification** — Deterministic management-guidance retrieval plus an earnings-baseline durability classifier (`TEMPORARILY_BOOSTED`, etc.) that suppresses moat/capital-efficiency bonuses earned on a transient cyclical-peak print.
- **Live-first FX rate cache** — `FxRateCache` (`src/fx_normalization.py`) resolves currency rates live-then-fallback, batched per unique currency per run and cached with a 1-hour TTL, replacing the previous always-fallback path in `src/ibkr/portfolio.py` and `src/ibkr/reconciliation_rules.py`; `FALLBACK_RATES_TO_USD` refreshed to current spot rates.
- **Per-agent/provider/model/tier cost accounting** — Cost rollups by provider, model, and service tier; unpriced-model surfacing; an order-independent pricing matcher (prices `kimi-k3`); retry-cost re-attribution to the originating agent; and a new `scripts/cost_report.py` for ranking spend and diffing baseline-vs-candidate runs.
- **OpenAI-compatible endpoint routing** — `OPENAI_API_BASE` lets the Consultant/Auditor route through OpenAI-compatible providers (e.g. Kimi/Moonshot).

### Changed

- **BUY rationale discipline** — A BUY must cite at least one eligible supporting fact; passing health/growth scores alone no longer establish the forward case.
- **Ownership and evidence provenance** — Further hardening of ownership-structure and MRQ statement-period provenance, with analysis-quality gaps now scored separately from issuer risk instead of being conflated into one penalty.
- **Value Trap Detector** — Thinking level bumped one notch in full mode specifically to reduce fabrication.
- **Gemini model support** — Updated model health checks and pricing for current-generation Gemini models.

### Fixed

- **False rejections from the new evidence gates** — Raw-read `N/A` in guidance validation and rounded-precision citation matches no longer trip false rejections; live-data conflict fast-fails no longer fire on false positives; FLA output budget and FX timeout widened.
- **Forensic-auditor host-rejection observability** — Rejected document hosts are now logged deterministically (previously visible only in the LLM's prose) alongside multi-source issuer-website discovery and raised byte/timeout caps.
- **False `N/A` collapse on `ADJUSTED_*_SCORE`** — `GROWTH_ADJ`/`HEALTH_ADJ` no longer collapse to a false `N/A` when a structurally-unbacked rubric criterion (one that can never be corroborated by design) vetoed an otherwise fully-corroborated scorecard.
- **Article-relative chart paths** — Corrected chart path resolution when generating articles.

## [3.13.0] - 2026-07-19

### Added

- **Concentration-aware watchlists** — IBKR recommendations now select BUY-ready watchlist entries and dip adds within configurable exchange and sector limits, account for cash and working orders, protect existing holdings, and keep a safe non-empty watchlist floor.
- **Business-quality decision context** — Investment memos surface code-derived ROIC quality and moat signals, while gate-passing `DO_NOT_INITIATE` outcomes are explicitly marked for review.
- **Bounded forensic evidence** — The Auditor can use approved official disclosures in HTML/text or bounded PDF form, with deterministic freshness, period/scope, currency, and auditor-opinion gates plus shared budget telemetry.

### Changed

- **Safer held-position decisions** — Verdict rejects no longer become automatic sells; exits are confirmation-gated, while entry-constraint and price-weakness cases are routed to review or monitoring to reduce churn.
- **Unified portfolio workflow** — CLI reconciliation and the Flask dashboard now share an action plan and consistent treatment of watchlists, concentration breaches, cash, turnover, freshness, live-order failures, and degraded broker state.
- **Valuation and market-data handling** — Added GPT-5.6 model/pricing support and strengthened multi-exchange position identity, FX normalization, local-price valuation, and portfolio reporting.

### Fixed

- **Forensic and accounting reliability** — Current and audited evidence are kept separate and compared only when periods, scopes, and currencies align; bounded retrieval and explicit insufficiency states prevent unsupported evidence from being treated as clean.
- **Portfolio/report consistency** — Watchlist candidate accounting, action sections, dashboard drilldowns, concentration explanations, and freshness/degraded-state warnings now use the same underlying decisions.

## [3.12.0] - 2026-07-12

### Added

- **APEX Model Tier** — Gate-critical seats (Senior Fundamentals Analyst, Portfolio Manager) are pinnable to a dedicated `APEX_MODEL` with deep-tier thinking, separate from the flash data-gathering tier; in `--quick` they pin to the standard service tier with a larger per-call budget so the block-emitting work finishes before the pipeline watchdog.
- **Flex Service Tiers** — Gemini/OpenAI flex inference at ~50% token price with error-discovered capability caching, standard-tier fallback on capacity/latency, load-bearing timeout floors (quick-mode exempt), and flex/cache-adjusted cost tracking (corrects months of 3–4× cost under-reporting on gemini-3.x/gpt-5.x).
- **Writer Fallback Chain** — The article writer degrades through an ordered Claude → `EDITOR_MODEL` (OpenAI) → Gemini floor chain with truthful `article_writer_fell_back` stamping and family-neutral log events.
- **IBKR Advisory Data Source** — Opt-in (`IBKR_DATA_SOURCE_ENABLED`) sixth merge source supplying point-in-time ratios/price (quality 9.4, overrides Yahoo/FMP, gap-fills the ex-US `trailingPE` vacuum); a no-op that auto-recovers when unentitled.
- **Batch Health Digest** — `scripts/scan_batch_health.py`, auto-run at each stage end of `run_pipeline.sh`, summarizes anomalies (non-publishable, `llm_failures`, consultant `ERROR`/`UNPARSED`, same-mode verdict flips), scoped by file mtime for cross-day-resume robustness.
- **Prompt-Drift Harness** — Tiered L0–L3 prompt↔parser contract tests (`make test-prompts` / `replay` / `eval-semantic`) catching silent prompt/parser format drift at commit time.
- **Score-Consistency & Drawdown Flags** — Deterministic health/growth score validation (`SUSPECT` + arithmetic repair, `*_SCORE_UNRELIABLE` gate guard, per-criterion rubric breakdown audit) and an `UNEXPLAINED_DRAWDOWN_NEWS_GAP` flag with a News-Analyst price-drawdown protocol.

### Changed

- **Portfolio-Manager BUY Discipline** — `VALUATION_INPUT_RELIABILITY` quarantine (distrusted multiples can't back a BUY), suppression of peak/transient moat bonuses, distortion-before-catalyst and material-op-signal BUY blocks, a now-default BUY-stability gate (decoupled from `src.agents` via neutral `pm_decision_parser`), and weak-asymmetry BUY qualification.
- **Consultant/Auditor Wiring** — Verdict scoped to the FINAL CONSULTANT VERDICT section; explicit `MANDATE_BREACH:`/`HARD_STOP:` tokens (prompt v2.12) with negation-aware fallback; a ≤50% partial-tool-failure passes a tagged review rather than discarding it; auditor decoupled from the consultant on the shared OpenAI plane; gate-bypass labeled `SKIPPED` not `UNPARSED`.
- **PM Sizing Reconciliation** — No-initiation verdicts (HOLD/DO_NOT_INITIATE/SELL) clamp both the machine `POSITION_SIZE` token and the human-facing prose to zero, applied after all deterministic verdict modifiers.
- **ex-US Data Quality** — Statement-derived FY revenue/EPS growth overrides stale yfinance scalars (`GROWTH_DATA_STALE`); OCF FILING-authority gated on exact extraction with an independent auditor cross-check; exchange-code canonicalization (TSEJ/TSE/TSXV) with a ground-truth canary.
- **Report Rendering** — Compliance visuals for P/E, health, and growth follow the PM gate verdict; "undiscovered" is softened via a banner caveat; unresolved-auditor stubs render as prose; HOLD memos carry a monitor-only clarifier.

### Fixed

- **Full-Suite pytest Hang** — Memory embedding calls are hard-bounded (google-genai's per-request `timeout=None` disabled all httpx timeouts, hanging an SSL read forever).
- **`--quick` child_timeout Kills** — Flex timeout floors are exempted in quick mode with a standard-tier fallback on queued-flex latency, and APEX seats are pinned standard with a larger budget so gate-critical seats aren't guillotined before emitting their block.
- **DuckDuckGo Fallback Death** — The single-worker search executor is replaced with a constructor-locked multi-worker pool so one hung socket read no longer kills fallback for the rest of the process.
- **Cloudflare 52x** — Consultant transient 520–524 responses are classified retryable instead of dying non-retryable.
- **IBKR Session Lifecycle** — One pooled connection per process with retry on brokerage-session init, watchlist fail-closed, and multi-exchange held-position `conid` resolution.
- **Currency/Valuation** — GAMA.L GBp→GBP double-scale, `format_iv` false cents on KRW/JPY, and scenario intrinsic-value normalization on peak/one-off EPS.

## [3.11.0] - 2026-06-11

### Added

- **Korean Equities (KRX/KOSDAQ)** — Enabled end-to-end: config gate, 6-digit yfinance padding, KRW→`.KS` fallback, KSE alias in IBKR↔yfinance translation, and a screener P/E fallback (`marketCap / netIncomeToCommon`) so Korean rows survive yfinance's null `trailingPE`.
- **Brazilian Equities (B3)** — Tickers/exchanges added with BDR filtering at the scraper (round-trip translation preserved) and a pagination fix for `.SA` listings.
- **Special-Situation Exits** — Active tender offers surface as `SPECIAL_SITUATION_EXIT` with a market-vs-tender spread narrative and a PM event-driven override of the hard-fail framework; LOW/MODERATE screen rejects on held positions route to REVIEW (`SCREEN_REJECT`) instead of executable SELLs.
- **Macro-Event Override Rebuild** — Correlated-sell detection now fires on three triggers (14d verdict window, 35% cumulative with stop-breach evidence, refresh-immune price drawdown breadth); demotions are sustained from stored events with rolling expiry; truthful per-trigger flag wording; extensive edge-case regression suite.
- **Ticker-Override Registry & Data-Vacuum Gate** — Operator-curated `config/ticker_overrides.json` for confirmed listing migrations (applied on both analysis and IBKR position sides), `ticker_migration_suspected` logging, and a pre-LLM abort for total-data-vacuum tickers (`--force-data-vacuum` to bypass).
- **Entity Governance Cards** — Canonical entity identity persisted and threaded across prompts, metrics, and serialization.
- **Macro-Regime Threading** — Structured regime context across prompts, persistence, retrospectives, and DIP WATCH scoring.

### Changed

- **Prompt Corpus** — `prompts/*.json` is the single canonical prompt source with fail-loud loading; the stale 3,100-line inline fallback in `src/prompts.py` was removed.
- **Thesis Constants** — Centralized in `src/thesis_constants.py` with prompt↔code parity tests (also corrected stale $500k liquidity documentation to the implemented $100k/$250k policy).
- **Logging Standards** — Six statically-enforced invariants: structlog-only, snake_case events, no f-strings, sanitized exception summaries (`summarize_exception`), redacted content previews, kwargs-only calls; tool/LLM-visible errors are typed, never raw exception prose.
- **PFIC Cross-Check #11** — High cash/asset ratio no longer overrides Legal Counsel; HIGH requires `pfic_status=PROBABLE` or ratio ≥50% with a passive-profile sector.
- **Agent Input Hygiene** — Trader/risk/research prompt assembly reads valid artifacts only, so failure stubs never enter downstream prompts.

### Fixed

- **Cross-Exchange Ticker Matching** — Ambiguous base symbols (e.g. `AGS.SI` vs `AGS.BR`) are poisoned and base-symbol fallback requires currency agreement, ending wrong-exchange analysis attachment.
- **Logger-Level Flattening** — `Settings` construction no longer force-levels every registered logger, which silently destroyed CLI noise suppression (httpx/ibind/genai chatter).
- **Data-Hygiene Repairs** — Deterministic fixes for invalid 52-week ranges, nonsensical ADR routing, and low-P/E anomalies; ADR fields confined to a single canonical DATA_BLOCK source.
- **pm_block Circular Import** — Chart extractors are importable standalone; fresh-subprocess import-order tests guard regressions.
- **Privacy** — Personal portfolio exports untracked and the real IBKR account ID replaced with a synthetic placeholder across docs, scripts, and tests.

## [3.10.1] - 2026-05-22

### Added

- **Investment Memo Section** — New memo-first report section with bear `KILL_CRITERIA` parser, APAC verdict rubric defaulting to SUPPORT, source-confidence table, and APAC/Auditor PM fallback resolution blocks.
- **Valuation Scenarios** — `VALUATION_SCENARIOS` block with Python-computed bear/base/bull intrinsic values and probability-weighted mean, surfaced in the memo and used as a PM stop-loss anchor.
- **Football-Field Scenario Overlay** — Bear/base/bull/weighted scenario markers on the football-field chart, with compacted labels and a constrained layout to avoid legend/caption overlap.
- **Research Manager Variant Perception** — Variant-aware perception pass with placeholder-variant filtering.
- **Report Quality Judge** — New CLI (`src/eval/report_quality_judge.py`) that resolves rendered markdown via `--markdown-dir` and buckets zero-feature legacy artifacts separately from FAIL.
- **Integration Suite** — 48-test end-to-end suite with fixture builders parameterized across runtime and saved-JSON shapes, exercising all five quality-feature data flows plus strange-input regression guards.

### Changed

- **Saved-State Wiring** — Memo, PM payload, and football-field chart now read saved-JSON shapes, real `DATA_BLOCK` field names, and derived EPS so scenario valuation actually fires in production.
- **PM Summarizer** — Preserves previously worked-out fenced blocks through summarization; auditor fallback gated on named forensic checks rather than a generic "no red flags" string.
- **Auditor/Consultant/Valuation Artifacts** — Persisted alongside other artifacts for downstream consumers.

### Fixed

- **Article Path Resolution** — Article path is now resolved relative to the current working directory rather than the directory where `--output` places its file.
- **Football-Field Layout** — Resolved legend/caption overlap and tightened scenario label rendering.
- **Saved-State Access** — Fixed valuation and source-access paths when reading from saved JSON.
- **Overlay Error Sanitization** — New logging routed through `summarize_exception` to avoid leaking raw tracebacks.

## [3.10.0] - 2026-05-17

### Added

- **APAC Regional Specialist** — Optional no-tools regional audit node after Research Manager, before Consultant. Sends a minimized DeepSeek-compatible payload, lets the agent's silence protocol decide when no material APAC exposure exists (listing, supply chain, customers, regulation, FX, or commodities), and threads the audit into Consultant, Trader, and Portfolio Manager context.
- **Adversarial Prompt-Scanning Corpus** — Added injection-payload corpus, judge-replay fixtures, refresh scripts, and corpus-driven tests for heuristic/LLM-judge inspectors, argument policy, content propagation, and memory-write filtering.
- **MCP Consultant Tooling** — Added MCP runtime/auth/budget/client modules, example server config, smoke script, docs, and consultant-tool integration tests for controlled external tool access.
- **Runtime and Inspection Hardening** — Expanded sanitized logging, prompt-boundary wrapping, tool argument policy checks, escalating/LLM-judge inspectors, runtime service scoping, hard-timeout coverage, and network/circuit-breaker protections.
- **Pipeline Health Tooling** — Added run-health extraction, batch-summary, signal, and slow-tail scripts to make long screening runs easier to diagnose.

### Changed

- **Large Ownership Refactor** — Split the CLI/output/persistence spine, data fetcher, IBKR reconciler, and red-flag validator into smaller ownership modules while preserving public behavior.
- **Metrics and Prompt Adjustments** — Tightened prompt and metric handling around derived metrics, coverage gaps, entity/period mismatches, local coverage, capital allocation, valuation context, and APAC regional review.
- **IBKR and Dashboard Surfaces** — Improved reconciliation ownership modules, portfolio presentation, refresh services, serializers, live dashboard behavior, and security data/quote rescue paths.
- **Dependency and Runtime Updates** — Updated Python dependencies and runtime initialization paths, removed legacy toolkit facades, and kept package roots intentionally lean.
- **README/Test Organization** — Refreshed architecture documentation, example artifact locations, adversarial test documentation, and regression coverage across agents, MCP, tooling, IBKR, validators, scripts, and web surfaces.

### Fixed

- **Prompt-Injection Regression Coverage** — Added corpus floors and replay tests to catch regressions in tool-output inspection, argument policy, content propagation, and memory-write filtering.
- **Runtime Failure Surfaces** — Hardened timeout, network, import-laziness, provider-failure, and sanitized error-reporting paths.
- **Portfolio/Pipeline Edge Cases** — Fixed several screening, cash-summary, interrupted-run, ticker-resolution, and dashboard-refresh edge cases covered by the expanded test suite.

## [3.9.11] - 2026-04-19

### Added

- **Cached Regional Macro Context** — Added a pre-graph macro analyst so individual equity evaluations can include better regional macro context.
- **Cash-Hoarding Value-Trap Checks** — Enhanced idle-cash and cash-hoarding checks so large cash balances without credible deployment plans are treated as capital-allocation/value-trap risk.

## [3.9.10] - 2026-04-12

### Changed

- **Debugging / Consolidation Pass** — Hardened Langfuse observability, cleaned up runtime and article/editor tracing seams, tightened regression coverage, and improved IBKR portfolio reporting so cash-blocked watchlist candidates are surfaced explicitly instead of disappearing silently.

## [3.9.7] - 2026-03-31

### Added

- **Simple Flask UI** — Added a simple Flask-based dashboard UI for IBKR portfolio and analysis review.

## [3.9.6] - 2026-03-28

### Added

- **Prompt Regression Workflow** — Added baseline capture suites, deterministic live prompt checks, and Stage 3 semantic regression checks with an LLM judge under `src/eval/`.

### Changed

- **Shared Default Eval Basket** — `smoke` now uses a globally mixed default basket and both baseline capture and prompt checks run it with no extra arguments.
- **Portfolio / Runtime Cleanup** — Large internal refactor across portfolio-manager, IBKR services, runtime helpers, and supporting tests/docs.

## [3.9.5] - 2026-03-21

### Added

- **Content Inspection Hook** — `src/tooling/inspector.py` / `inspection_service.py` / `inspection_hook.py`: policy layer that potentially sanitises every tool output before it reaches LLM context. `NullInspector` default (zero overhead); swap backend via `configure_content_inspection()` at startup.
- **Per-Agent Output Budgets** — `src/llm_budgets.py` centralises fractional `max_tokens` caps per agent; `src/agents/output_validation.py` coerces and validates DATA_BLOCK numeric fields.

### Fixed

- **`DO NOT INITIATE` Restart Parsing** — `run_pipeline.sh` now correctly recognises `DO NOT INITIATE` verdicts when resuming interrupted batch runs; previously they were re-queued as unresolved.
- **Exchange-Qualified Ticker Propagation** — `src/ticker_policy.py` centralises exchange-suffix rules; fixes cross-exchange base-symbol clashes in the data fetcher and pipeline verdict parser.
- **Trivy CI Pin** — `trivy-action` bumped `0.28.0` → `0.35.0`; the old tag did not exist, silently blocking all Trivy jobs before any scan ran.

### Changed

- **Container Hardening** — `Dockerfile`, `docker-compose.yml`, and `scripts/check-environment.sh` updated for Podman-first runtime; `README.md` local-container setup docs expanded.

## [3.9.4] - 2026-03-19

### Changed

- **Internal Hardening / Cleanup Release** — Broad regression-test expansion, efficiency work (especially lazy initialization and runtime overhead reduction), package-level refactoring for clearer ownership, and more consistent IBKR / portfolio-manager reporting. This release is mostly cosmetic and operational rather than feature-driven.

## [3.9.3] - 2026-03-04

### Fixed

- **Consultant C1/C2 Scope** — C1/C2 penalties now apply only to NUMERICAL metric discrepancies. Qualitative governance claims (affiliates, related-party relationships, M&A history) documented in `VALUE_TRAP_BLOCK` (CROSS_HOLDINGS, MAJORITY_HOLDER, M&A_HISTORY) are pre-verified by the Value Trap Detector and must not attract C1/C2 charges (`consultant.json` v2.5).
- **PFIC Asset Test False Pass** — When Cash, STI, or Total Assets are unavailable, the fundamentals analyst now sets `PFIC_ASSET_RATIO = N/A` and writes `QUANTITATIVE_TEST: INSUFFICIENT_DATA` instead of claiming "asset test passed" without a computed ratio (`fundamentals_analyst.json` v9.8).
- **DATA_BLOCK Authority Inversion** — Bull, Bear, and Risky analysts no longer frame DATA_BLOCK figures as "the claim" when a conflicting news/annual figure appears. All three now cite both figures with explicit periods (e.g., "DATA_BLOCK: -33.6% [TTM]; news: -6.7% [FY2024]") and anchor analysis on the DATA_BLOCK value (`bull_researcher.json` v2.7, `bear_researcher.json` v2.9, `risky_analyst.json` v5.3).

## [3.9.2] - 2026-03-04

### Fixed
- **D/E False Positive** — DATA_BLOCK `D/E: 6.92%` was escalated to 692% by the `<10 → ×100` normalisation heuristic after the `%` sign was stripped. Now captures the `%` across all six regex patterns and bypasses the multiplier.
- **UNRELIABLE_PEG False Positive** — PEG < 0.05 penalty suppressed when `revenue_growth_ttm ≥ 50%`; high-growth companies were incorrectly penalised.
- **D/E Scaling (1.5–10 range)** — Edge-case heuristic corrected for ratios in the ambiguous 1.5–10 band.
- **`--skip-scrape` with `--stage 1`** — Flag was silently ignored; now handled correctly.
- **Consultant Derived Metrics** — EV/EBITDA, ROE, and FCF Yield in FINANCIAL HEALTH DETAIL were incorrectly flagged as hallucinated; Derived Metrics Rule added (`consultant.json` v2.3).

### Changed
- **Analyst Hallucination Guardrails** — `NO FABRICATION` added to Bull and Bear KEY INSTRUCTIONS (v2.6/v2.8); `NO NEW NUMBERS` preamble and `PERIOD AWARENESS` note added to Research Manager (v5.1). Agents must write `[figure not in data]` rather than invent figures; 10–30% TTM vs FY divergence documented as normal.
- **Forensic Auditor PERIOD Labeling** — META block now outputs `PERIOD: FY/H1/H2/Q1–Q4`; balance sheet date extracted in fetcher (`auditor.json` v2.6).
- **Consultant COVERAGE_GAP Rule** — Neutral rule broadened from Taiwan/Japan micro-caps to all ex-US markets (`.AX`, `.TO/.V`, `.OL`, `.BR/.AS/.PA/.DE`) (`consultant.json` v2.4).
- **PFIC Detection** — Short-term investments (STI) added to cash definition; balance sheet PFIC fields extracted directly.
- **Stage 1 quick screen** — Now runs in non-strict quick mode; strict thesis enforcement is reserved for Stage 2 full analysis.
- **Taiwanese Equity Gap Handling** — Data gaps no longer treated as integrity failures; construction-sector debt thresholds and extra equity-match checks added.
- **Cross-Day Run Resumption** — Interrupted batch runs can resume mid-ticker-list without reprocessing completed tickers.

## [3.9.1] - 2026-02-22

### Added
- **IBKR Portfolio Integration** - Full support for Interactive Brokers REST API via `ibind`. Includes `IbkrClient` for rate-limited OAuth 1.0a sessions and a `portfolio_manager.py` tool for position-aware reconciliation against AI verdicts.
- **Settled Cash Awareness** - Portfolio logic now distinguishes between total cash and settled (spendable) cash, accounting for T+2 settlement delays in trade recommendations.
- **Automated Portfolio ADD/TRIM** - Reconciler now calculates precise order quantities to scale into underweight positions (ADD) or reduce overweight positions (TRIM) based on target weights.
- **Pipeline Rejection Records** - Stage 1 pipeline rejections (HOLD/DNI) are now stored in the global `lessons_learned` collection. These factual "Negative Evidence" records are injected into future analyses of the same ticker to prevent redundant debate and ensure consistent rejection reasoning.
- **Independent Forensic Channel Visualization** - Updated documentation and Mermaid diagrams to reflect the decoupled data flow from the Forensic Auditor to the External Consultant for independent cross-validation.

### Changed
- **Safety-First Execution** - The `--execute` flag in `portfolio_manager.py` is currently disabled to ensure manual verification of AI recommendations.

## [3.9.0] - 2026-02-20

### Added
- **Multi-Horizon Growth Analysis** - Data fetcher now computes three distinct growth horizons from quarterly statements: FY (annual), TTM (trailing twelve months), and MRQ (most recent quarter YoY). Deterministic `growth_trajectory` field (ACCELERATING/STABLE/DECELERATING) added to DATA_BLOCK. Catches deterioration hidden in rearview-mirror annual figures.
- **Retrospective Learning System** (`src/retrospective.py`) - Compares past analysis verdicts to actual market outcomes using excess return vs local benchmark. Generates lessons via Gemini Flash (~$0.001/lesson) and stores them in a global `lessons_learned` ChromaDB collection with geographic boost at retrieval time. Top-3 relevant lessons injected into Bull/Bear researcher prompts on future runs. Eight failure mode taxonomy aligned with Bear pre-mortem analysis.
- **Company Name Verification** (`src/ticker_utils.py`) - Multi-source resolution chain (yfinance → yahooquery → FMP → EODHD) prevents hallucinations when tickers are delisted or ambiguous. Unresolved names inject an explicit warning into all agent system instructions.
- **New Red Flags** - `GROWTH_CLIFF` fires when TTM revenue drops >15%. `THIN_CONSENSUS` fires when total analyst coverage <3, flagging unreliable PEG and target prices.
- **Batch Screening Pipeline** - `scripts/find_gems.py` consolidates scraper and filter into a single two-phase script. `scripts/run_pipeline.sh` orchestrates three-stage screening (scrape → quick analysis → full analysis on BUYs) with resumability and `--force`/`--stage`/`--cooldown` options.
- **Script Tests** - 61 new tests covering `find_gems.py` filters, scraping, CLI parsing, and `run_pipeline.sh` verdict extraction, filename conventions, and resumability logic.

### Changed
- **GICS 11-Sector Alignment** - Red flag detector and fundamentals analyst now use standard GICS taxonomy (Energy, Materials, Industrials, Consumer Discretionary, Consumer Staples, Health Care, Financials, Information Technology, Communication Services, Utilities, Real Estate) instead of legacy ad-hoc sector names. Three threshold profiles: Financials (D/E disabled), Capital-intensive (D/E >800%), Standard (D/E >500%).
- **English vs Total Analyst Coverage** - `ANALYST_COVERAGE_ENGLISH` (yfinance) now distinguished from `ANALYST_COVERAGE_TOTAL_EST` (supplemented by Foreign Language Analyst's local coverage estimate). Prevents false "undiscovered" signals on stocks with heavy local-language coverage.
- **Test Directory Reorganization** - 86 test files moved from flat `tests/` into 10 domain subdirectories: `agents/`, `memory/`, `validators/`, `charts/`, `reports/`, `financial/`, `config/`, `prompts/`, `advanced/`, `scripts/`. All `pytest` commands continue to work unchanged.

### Removed
- Obsolete scripts: `scripts/filter_tickers.py`, `scripts/ticker_scraper.py`, `scripts/run-analysis.sh`, `scripts/SCRIPTS_QUICK_REFERENCE.md` (replaced by consolidated pipeline).

## [3.8.0] - 2026-02-08

### Added
- **Historical Data Weighting** - Agents now prioritize multi-year financial trends and historical red flags to better identify cyclical peaks and unsustainable growth.
- **Tool-Equipped Consultant** - The Consultant node can now execute independent tool calls to verify metrics directly against primary filings.
- **Forensic Validation Layer** - Expanded red-flag detector with deep-dive governance and accounting anomaly checks.
- **Ticker Discovery Suite** - New scripts for automated ticker scraping and multi-factor screening across international exchanges.

### Changed
- **Lean Prompt Engineering** - Massive consolidation and shortening of all agent prompts to reduce token overhead and improve instruction adherence.
- **Robust Connection Management** - Implemented session-per-request patterns and global timeouts for all network operations to eliminate stranded connections.
- **Unicode-Category Sanitization** - Replaced character whitelists with category-based validation for superior international company name handling.

### Fixed
- Fixed EDINET domain resolution and implemented caching for 4xx responses.
- Enhanced truncation detection to be agent-aware, eliminating false positives from structured output references.

## [3.7.0] - 2026-02-01

### Added
- **International Currency Symbols** - Football field charts now display correct local currency (¥, £, €, ₩, HK$, etc.) based on exchange suffix
  - Supports 40+ exchanges including suffix currencies (Polish zł, Swedish kr, Czech Kč)
  - `CurrencyFormat` class handles both prefix ($100) and suffix (100 zł) conventions
- **PFIC Quantitative Asset Test** - Fundamentals Analyst now calculates cash/market-cap ratio
  - R ≥ 50%: Flags as HIGH PFIC risk
  - R ≥ 45%: Flags as MEDIUM PFIC risk
  - 35% price decline would trigger 50%: Flags as PFIC_CASH_TRAP latent risk
- **Agent Output Constraints** - Added word limits and anti-bloat rules to all 13 agent prompts to reduce truncation risk

### Changed
- **Log Levels** - Investment detection messages (red flags, legal flags, value trap flags) changed from WARNING to INFO since they indicate the system working correctly, not errors

## [3.6.0] - 2026-01-18

### Added
- **Universal Data Attribution** - System now tracks the exact source (API) of every financial metric throughout the pipeline
  - `SmartMarketDataFetcher` attaches `_field_sources` metadata to all metrics
  - New `DATA SOURCE ATTRIBUTION` table injected into Consultant, Research Manager, and Portfolio Manager contexts
  - Enables "Glass Box" reasoning: Agents can now distinguish between primary exchange data (e.g., "eodhd") and fallback estimates (e.g., "yfinance")
  - Consultant now explicitly verifies "Provenance" in the Hierarchy of Truth check
- **Robust Parallel Execution** - Fixed information flow for agents running in parallel
  - **Value Trap Detector (v1.3)**: Now correctly marked as parallel-independent; no longer attempts to read `DATA_BLOCK` from Fundamentals (which isn't ready yet). Uses qualitative signals for capital allocation rating instead.
  - **Sentiment Analyst (v5.2)**: Removed dependency on `fundamentals_report` for "Undiscovered" status check to ensure safe parallel execution.
  - **Research Manager (v4.6)**: Removed direct dependency on `auditor_report` (now adjudicated solely by Consultant) to streamline graph flow.

### Changed
- **Trader Prompt** - Now receives Valuation Parameters chart data context
- **Research Manager Prompt** - Now receives a "Data Provenance Note" to help resolve Bull/Bear conflicts based on data source quality and timeliness
- **Test Suite** - Added extensive tests for attribution extraction (`tests/test_attribution.py`) and fixed integration tests for parallel node execution (`tests/test_quantitative_validation_integration.py`)

## [3.5.0] - 2026-01-11

### Added
- **FORENSIC_DATA_BLOCK Structured Output** - Forensic Auditor now produces standardized accounting data block
  - Includes META (report date, currency, auditor opinion), EARNINGS_QUALITY, CASH_CYCLE, SOFT_ASSETS, SOLVENCY, CASH_INTEGRITY metrics
  - Multilingual terminology guide for international financial statements (Japanese, Chinese, Korean, German)
  - Calculation formulas for NI_TO_OCF, Paper Profit, DSO/DIO/DPO, Zombie Ratio, Altman Z-Score, Ghost Yield, Trash Bin ratios
  - Date validation to prevent stale data usage (>18 months triggers penalty)
  - Consultant validates forensic findings against Senior Fundamentals for cross-model verification
  - Portfolio Manager applies forensic penalties (+0.5 to +2.0 risk points) based on auditor opinions, RED_FLAGs, and data age
  - All forensic findings are advisory (no hard fails) - contribute to risk scoring only
  - See `tests/test_forensic_data_block.py` for 27 comprehensive tests
- **Moat Detection** - Red-flag validator now detects durable competitive advantages
  - Identifies pricing power, switching costs, network effects, regulatory barriers
  - Applies negative risk penalties (-0.5 to -1.0) to offset qualitative risks when moats detected
  - Flags appear in pre-screening results as MOAT_DURABLE_ADVANTAGE, MOAT_PRICING_POWER, etc.
- **Capital Efficiency Analysis** - Pre-screening now calculates ROIC and detects leverage engineering
  - Flags value destruction (negative ROIC), engineered returns (D/E >200% + ROIC>ROE), suspect returns (D/E 100-200%)
  - Applies risk penalties (+0.5 to +1.5) for capital structure concerns
  - Bonus for genuinely capital-efficient companies (-0.5 when ROIC >12% + conservative)
  - See `tests/test_capital_efficiency.py` for calculation logic

### Changed
- **Auditor Prompt (v2.1 → v2.2)** - Added FORENSIC_DATA_BLOCK template with international terminology and thresholds
- **Consultant Prompt (v1.0 → v1.1)** - Added forensic validation section for cross-checking accounting flags
- **Portfolio Manager Prompt (v7.1 → v7.2, Thesis v7.3 → v7.4)** - Added forensic penalties and capital efficiency flags to risk scoring

### Added (internal 3.4.0 version never released)
- **Value Trap Detector** - Agent for identifying value traps via ownership structure analysis
- **XML Security Boundaries** - Tavily search results now wrapped in `<search_results>` tags with `data_type="external_web_content"` attribute for prompt injection mitigation
- **Configurable Batch Cooldown** - `COOLDOWN_SECONDS` environment variable for `run_tickers.sh` (default 60s for free tier, 10s for paid)
- **FY Hint in Date Injection** - Agents now receive fiscal year context to prevent future-dated annual report searches

### Changed
- **Rate Limit Handling** - Added random jitter (1-10s) to exponential backoff to prevent thundering herd on parallel agent retries
- **Tavily Truncation** - Now cuts at `</result>` boundaries to preserve valid XML structure instead of arbitrary character positions

### Fixed
- Fixed undefined function reference in news formatting (`_truncate_tavily_result` → `_format_and_truncate_tavily_result`)
- Fixed module-level import for `random` in rate limit handling
- Removed unnecessary `html.escape` that broke Markdown formatting in search results

## [3.3.0] - 2026-01-05

### Added
- **Global Forensic Auditor** - Optional independent agent that runs in parallel with other analysts to validate financial data
  - Uses OpenAI (same as Consultant) for cross-model verification
  - Searches foreign sources, financial metrics, and news independently
  - Output feeds into Consultant for comprehensive cross-validation
  - Enabled automatically when `ENABLE_CONSULTANT=true` and `OPENAI_API_KEY` is set
  - Configurable model via `AUDITOR_MODEL` (defaults to `CONSULTANT_MODEL`)

### Fixed
- Fixed graph compilation failure when Auditor conditionally disabled
- Fixed toolkit attribute access error in auditor node creation
- Fixed router/graph mismatch for auditor enable state

### Changed
- Updated mermaid architecture diagram to show Forensic Auditor
- Added `_is_auditor_enabled()` helper for consistent enable-state checking across routers

## [3.2.0] - 2026-01-01

### Added
- **6-Axis Thesis Alignment Radar** - New visualizer showing Health, Growth, Value, Undiscovered status, Regulatory risks, and Jurisdiction stability.
- **Structured Data Model (v7.4)** - Updated prompt schema and extractors to use deterministic fields for D/E ratios, ROA, and specific jurisdictional identifiers, eliminating fragile narrative parsing.
- **Automatic Path Expansion** - Integrated `os.path.expanduser` into the configuration system to prevent the creation of literal `~` directories in the project root.
- **Robust Bash Cleanup** - Added `trap` and `cleanup_temp_files` logic to `run_tickers.sh` to ensure workspace hygiene even after interrupted runs.

### Changed
- **Plotting Infrastructure** - Refactored all chart generators (`football_field.py`, `radar_chart.py`) to use the Matplotlib Object-Oriented API for improved thread-safety and state isolation.
- **Theme-Agnostic Accessibility** - Charts now automatically adjust colors in `--transparent` mode to ensure legibility on both dark and light Markdown readers.

## [3.1.0] - 2025-12-18

### Added

- **Junior Analyst and Foreign Language Analyst Parallel Chain** - Redesigned the fundamental analysis stage into a parallelized research architecture
  - **Junior Analyst**: Handles standardized financial metrics, yfinance/yahooquery data fetching, and core profitability scoring.
  - **Foreign Language Analyst**: Dedicated agent for analyzing local-language (non-English) financial news, filings, and regional sentiment.
  - **Senior Fundamentals Analyst (Synthesis)**: Acts as a gatekeeper that waits for both Junior and Foreign analyst outputs before synthesizing the final data block and growth score.
  - **Information Arbitrage**: Enables the system to identify discrepancies between global English-language consensus and local-language operational realities.
- **External Consultant Node** - Optional cross-validation using OpenAI ChatGPT to detect biases and validate Gemini analysis
  - Uses different LLM (OpenAI) to catch groupthink and confirmation bias that single-model systems miss
  - Positioned post-debate, pre-risk-assessment for maximum context
  - Fully backwards compatible - system works identically with consultant disabled
  - Configurable via `ENABLE_CONSULTANT` and `OPENAI_API_KEY` environment variables
  - Comprehensive test suite (29 new tests: 12 integration + 17 edge cases)
  - See `docs/CONSULTANT_INTEGRATION.md` and `docs/CONSULTANT_CONSISTENCY_REVIEW.md` for details
- GitLeaks and Trivy security scanning to CI/CD pipeline
- Red-flag financial validator for pre-screening (extreme leverage, earnings quality, refinancing risk)
- Currency normalization for liquidity calculations (FX rate conversion)
- Comprehensive documentation structure with docs/ directory
- LICENSE file (MIT)
- CONTRIBUTING.md with development guidelines
- SECURITY.md with vulnerability disclosure policy
- CODE_OF_CONDUCT.md for community standards
- Issue templates for bug reports and feature requests (.github/ISSUE_TEMPLATE/)
- Pull request template for consistent contributions
- Repository topics for GitHub discoverability (agentic-ai, langgraph, equity-analysis, etc.)
- Example script for single-ticker analysis with setup validation
- Repository gap analysis documenting best practices alignment
- GitHub Rulesets configuration with admin bypass actors
- Automated GitHub repository configuration script (scratch/configure_github_settings.sh)

### Changed

- **Report Generator** - Now includes consultant review section when available (intelligent filtering excludes errors/N/A)
- **Token Tracker** - Added OpenAI pricing (gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4) with correct model ordering
- Updated Dockerfile to multi-stage build pattern (40% smaller images)
- Improved error handling in report generator with fallback hierarchy (Portfolio Manager → Research Manager → Trader)
- Modernized GitHub Actions workflows (CodeQL v4, step-level conditionals)
- Enhanced Dependabot configuration for automated security updates

### Fixed

- **Consultant Node** - Fixed crash on `None` debate state with defensive null-checking
- **Consultant Logging** - All consultant logging properly respects `--quiet` flag (structlog suppression)
- **Portfolio Manager** - Fixed missing consultant review in decision context
- Fixed silent output truncation when Portfolio Manager fails to produce final decision
- Fixed Docker build with `--no-root` flag for Poetry dependency installation
- Fixed SARIF upload errors in security scanning workflow
- Fixed missing file checks in CI/CD Docker image scanning
- Fixed GitLeaks false positives in Terraform example files (Azure storage account names detected as Finnhub keys)

### Security

- Added GitLeaks secret scanning with custom rules for API keys (Gemini, Tavily, FMP, EODHD, Finnhub)
- Added Trivy vulnerability scanning for repository, Python dependencies, and Docker images
- Enabled SARIF report uploads to GitHub Security tab
- Added daily scheduled security scans

## [1.0.0] - 2025-12-01

### Added-01

- Initial public release
- Multi-agent analysis system using LangGraph 1.x
- Support for international ticker formats (Hong Kong, Japan, Taiwan, South Korea, Europe)
- GARP (Growth at a Reasonable Price) investment thesis enforcement
- Multi-source data pipeline with fallback logic (yfinance → YahooQuery → FMP → EODHD → Tavily)
- Ticker-isolated ChromaDB memory to prevent cross-contamination
- Versioned prompt system with metadata tracking
- Adversarial debate pattern (Bull vs Bear researchers)
- Multi-perspective risk assessment (Conservative, Neutral, Aggressive analysts)
- Comprehensive test suite (37 test files with unit, integration, and edge case coverage)
- Docker support with health checks
- Terraform examples for Azure Container Instances deployment
- Batch analysis support via run_tickers.sh script
- Rate limiting and retry logic for API calls
- LangSmith integration for observability

### Documentation

- Comprehensive README.md with architecture diagrams (Mermaid)
- Developer guide for AI assistants
- Honest limitations section ("Not a Get-Rich-Quick Bot")
- Performance benchmarks and cost estimates
- Troubleshooting guide for common issues

### Core Architecture

- Parallel data gathering (Market, Fundamentals, News, Sentiment analysts)
- Financial validator pre-screening with deterministic red-flag detection
- Research synthesis by Research Manager
- 1-2 round adversarial debate between Bull and Bear researchers
- Risk assessment from three perspectives
- Executive decision synthesis by Portfolio Manager

### Investment Thesis

- Hard requirements: Financial Health ≥50%, Growth ≥50%, Liquidity ≥$500k, Analyst Coverage <15
- Soft factors: P/E ≤18, PEG ≤1.2, P/B ≤1.4, US Revenue 25-35%
- Automatic SELL on thesis violations

---

## Release Notes

### Version 1.0.0 - Initial Release Highlights

This release establishes the foundation for a production-grade multi-agent investment analysis system. The architecture demonstrates advanced agentic AI patterns including:

- **State Machine Orchestration** via LangGraph with conditional routing
- **Memory Isolation** using ticker-specific ChromaDB collections
- **Tool Use** with structured schemas and error handling
- **Adversarial Debate** to reduce confirmation bias
- **Deterministic Validation** gates to prevent emotional decision-making

The system is designed for retail investors seeking institutional-quality analysis of international equities without the $24,000/year Bloomberg Terminal cost. It enforces a disciplined GARP strategy while maintaining full transparency of reasoning.

**Performance**: 5-10 minutes per ticker (standard mode), 2-4 minutes (quick mode), negligible cost on free-tier Gemini API.

**Limitations**: Historical data only, free APIs have gaps, backward-looking analysis, manual trade execution required.

**Use Case**: Generate shortlist of candidates for deep due diligence, not for automated trading.

---

## Migration Guides

### Upgrading to 1.0.0

This is the initial public release. No migration required.

Future breaking changes will include detailed migration guides in this section.

---

## Deprecation Notices

None currently. Future deprecations will be announced here with timeline and alternatives.
