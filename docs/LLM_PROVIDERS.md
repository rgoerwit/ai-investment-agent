# LLM Provider Qualification and Operations

Last updated: 2026-10-11 (an explicit seat client timeout reaches every request through one precedence chain; compatible vendors refuse an invocation tier; one minimum-useful-attempt fraction and one margin rule; one retry owner: every client sends one SDK attempt
and the runtime loop owns retries; the injection judge and signal processor
resolve their seats; one Google RPM reader, embeddings on their provider's
rate-limit bucket, a first token in every new bucket; 2026-10-09:
direct-caller attempts follow `LLM_ATTEMPTS`, the
OpenAI request cap is a setting, and the APAC ceiling is its node total; the
article workflow owns a deadline and the writer
runs on the runtime loop; a direct caller is sized from an enclosing deadline;
every OpenAI-family and Anthropic client is sized at
dispatch for the vendor it names; the ticker run deadline caps every enclosing
deadline under the pipeline; 2026-10-08: editor review, retrospective lesson and semantic judge
moved onto the budget policy; seat client timeout and vendor ceiling inside an
invocation; per-dispatch request budgets and application-owned retries; quick PM
504 re-attribution; PM standard pin).

## Support claims

Provider support has three distinct levels:

1. **Constructible** — offline client construction and parameter contracts pass.
2. **Contract-capable** — recorded or live tool calls, structured output, artifact
   validation, and usage telemetry pass for the exact seats in scope.
3. **Production-qualified** — credential-gated multi-ticker runs finish with
   publishable artifacts within budget and are compared with one pinned Semantic
   Judge.

Do not describe a provider as supporting the base fleet from construction alone.
The credential-free suite currently establishes offline construction and reviewed
capability contracts for Google and OpenAI in reversible base/review roles. It does
not substitute for live API evidence. No live production-qualification evidence was
created by the migration implementation because that requires operator credentials
and incurs provider cost.

Transport capability, runtime application permission, and evidence qualification are
separate contracts. For example, current Claude profiles accurately record tool use,
structured output, and version-specific effort controls, but Anthropic remains
writer-only until its analytical seats pass the qualification procedure below.
Binding validation checks capability and permission and reports a disallowed
provider/group assignment before model construction.

Runtime provider/group validation is an **allowlist**, not a qualification award.
An allowed assignment may be exercised for testing while still having only
constructible evidence. Promotion to contract-capable or production-qualified
requires the procedure and recorded evidence below; the runtime policy does not
silently confer either status.

Output budgets separate the application's visible-output intent from any API-side
reasoning reserve. Saved diagnostics record both caps and classify an invalid response
that consumes the API cap as `output_cap_exhausted`; an incomplete structured block
below that cap remains `incomplete_structured_output`, while a complete response that
omits a required contract field is `output_contract_violation`. Raising
`LLM_BASE_OUTPUT_TOKENS` scales fractional seat budgets and is therefore an experiment,
not a substitute for correct block-completion detection.

Reserve class is selected once from the canonical request contract: `reasoning`,
`critical`, and `escalation` intents use the deep reserve even when a provider resolves
their effort to `medium`; fast/classifier/prose requests use the effort as the fallback.
An adapter may still decide that a particular model generation does not count hidden
reasoning against the completion cap, in which case the configured reserve is zero.
Per-call telemetry persists visible/thinking output and the intent, API, and reserve
caps so provider comparisons can distinguish useful output from hidden-reasoning spend.

Provider retries and structural recovery are also distinct. The invocation runtime may
repeat a transient partial response when no explicit terminal reason is available. It
must not repeat an otherwise identical request after an explicit provider output-cap
stop: that attempt is recorded as `output_cap_exhausted` and returned to the owning
validator. Structural recovery happens only after a provider-success response fails
validation and changes the request: it is text-only, uses the base provider's reasoning
binding, inherits the originating seat's visible-output budget, and regenerates from
already gathered evidence. Quick mode arms this bounded recovery only for Senior
Fundamentals and Portfolio Manager. Cost rollups keep recovery usage under the
originating analyst for human interpretation and also expose the canonical
`analyst_retry` billing seat plus `originating_seat_id` for precise comparison.

Deterministic Portfolio Manager trace reconciliation and verdict-policy enforcement do
not use this recovery seat. Only missing, truncated, or otherwise structurally invalid
PM output is eligible for a model-based correction.

Capability facts should be checked against primary provider documentation. The
current Claude entries use Anthropic's official [tool-use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview),
[structured-output](https://platform.claude.com/docs/en/build-with-claude/structured-outputs),
and [effort](https://platform.claude.com/docs/en/build-with-claude/effort)
contracts; those citations do not count as application qualification evidence.
Claude Opus 4.8 has its own reviewed adaptive-thinking profile. A later version
does not inherit the broad Claude 4 profile automatically; add or update a reviewed
profile before using it.

The compatible z.ai and DeepSeek adapters are restricted to APAC's no-tool,
single-shot contract. Moonshot Kimi K3 is separately registered for the review
plane because this repository already exercises that exact family in Consultant,
Auditor, and Editor flows; it is not enabled as a base provider. Anthropic remains
writer-only until analytical-seat qualification is recorded. An arbitrary custom
OpenAI-compatible endpoint never inherits OpenAI or Kimi capability claims from
its wire shape.

xAI Grok 4.6 is registered for the review plane at **level 1, constructible**, and
is the current example of the gap between the levels: its capability row is drawn
from vendor documentation, and no live evidence exists. Capability sources are
xAI's [model page](https://docs.x.ai/developers/grok-4-6),
[reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning),
[structured-output](https://docs.x.ai/developers/model-capabilities/text/structured-outputs),
and [pricing](https://docs.x.ai/developers/pricing) contracts, verified
2026-08-14; as with the Claude entries, those citations are **not** qualification
evidence. Run the procedure below before describing Grok as review-qualified in
any stronger sense.

One xAI fact deserves restating because it inverts the usual failure mode:
`reasoning_effort` defaults to `high` and **reasoning cannot be disabled**. Since
`budgets.resolve_generation_budget` enables a reasoning reserve only when an
effort was resolved, a Grok profile with an empty `reasoning_ladder` would pair
guaranteed deep reasoning with zero reserve — reproducing the 1088.HK Consultant
starvation deterministically. The registered ladder is load-bearing, and a Grok
version without a reviewed profile (4.5, which documents no `xhigh`) is expected
to fail closed rather than inherit 4.6's.

Grok also wants a cache-affinity hint. Without an `x-grok-conv-id` header, xAI
documents that related requests reach cache-cold servers and bill full input rate
instead of the cached rate; `provider_policy.provider_default_headers` supplies a
per-process token for this, and it is a cache hint only — never a correlation key.

## Live qualification procedure

Use a fixed `LLM_JUDGE_PROVIDER` and model for both comparison arms. For each newly
qualified provider and role:

1. Run the adapter/construction and offline conformance tests.
2. Run credential-gated tool and structured-output tests for every reachable seat.
3. Exercise refusal, truncation, quota, timeout, and malformed-output handling.
4. Run quick and full smoke tickers spanning US, APAC, and another international
   listing.
5. Run `make test-prompts` and the semantic comparison with the same judge model.
6. Record date, exact models/endpoints, seats, test command, result artifacts, and
   judge identity below. Never record keys or endpoint paths/query strings.

## Evidence record template

```yaml
provider: openai
level: production_qualified
verified_on: YYYY-MM-DD
models: []
endpoint_host: api.openai.com
seats: []
commands: []
artifact_refs: []
judge:
  provider: google
  model: gemini-3.1-pro-preview
limitations: []
```

### Recorded evidence

```yaml
provider: xai
level: constructible          # capability row from vendor docs; no live runs yet
verified_on: 2026-08-14
models: [grok-4.6]
endpoint_host: api.x.ai
seats: [consultant, forensic_auditor, forensic_auditor_escalation,
        article_editor, article_writer_review_fallback]
commands:
  - poetry run pytest tests/llm_runtime/test_xai_review_plane.py
artifact_refs: []
judge:
  provider: google
  model: gemini-3.1-pro-preview
limitations:
  - No live API evidence; steps 2-5 of the qualification procedure are unrun.
  - Cost is modeled at the <200k-prompt tier only. xAI's >=200k tier doubles
    every token in the request, including output and cached, and the flat
    pricing table cannot express it.
  - Verbose-output reputation is unmeasured against this repo's per-seat
    budgets; watch consultant_review length on the first runs.
```

## Binding telemetry and cost

Saved analysis JSON and baseline configuration snapshots include the secret-free
resolved binding plan: seat, group, authority stage, vendor, lineage, adapter,
sanitized endpoint host, model, quick model, capability requirements, optional-mode
status, and independence state/waiver. Token usage prefers the resolved identity
and falls back to model-name inference only for legacy or external records.

**The pricing table drifts from the vendor's page, and nothing goes red.** On the
2026-10-01 re-verification, `gemini-3.6-flash` was billed at twice Google's
published rate even though the comment above it said the promotion still applied,
and `gemini-3.8-flash`, generally available for a month, had no row, so it would have
been billed at the default rate. Both errors only skew the cost figures used to
compare models, which is exactly when accuracy matters most. Each Gemini Flash
promotion ends on a fixed date (2026-12-31 for 3.6–3.8), after which the rows
double. Re-verify every row against its source page on any configured-model
change, any vendor model announcement, monthly, and when a recorded promotion
lapses. To confirm a
model ID exists for your key, list models through the SDK; a vendor announcement is not proof (Gemini 4
Argon was announced 2026-09-30 with no public ID).

Every shipped default model must have explicit pricing. Unknown custom models are
listed under `unpriced_models`, contribute zero to dollar totals, and emit one
diagnostic when encountered in legacy/external usage; the new binding schema
rejects models without a reviewed capability profile. There is deliberately no
global `ALLOW_UNREVIEWED_MODEL` switch: a future operator extension should be a
typed profile containing identity, capabilities, reasoning values, qualification
scope, and a persisted justification—not a boolean that bypasses all contracts.

**Unknown *versions* resolve asymmetrically, on purpose.** Profile lookup is
longest-prefix, so what an unrecognized version inherits depends on how the family
names itself. `gpt-5.7` and `gemini-3.9-flash` fall back to the `gpt-5` and
`gemini-3` rows, which model those families generically and conservatively —
usable. Anthropic encodes the generation in the prefix (`claude-opus-4-6`,
`claude-opus-4-8`), so the shortest matching row describes the *oldest* generation
and inheriting it would understate a newer model's capabilities; unreviewed later
Claude versions therefore fail closed until a row is added. When adding a family,
decide which of these two shapes it has before writing the prefixes.

## Runtime policy

Rate-limit buckets are independent by effective vendor and sanitized endpoint
host. Configure the native defaults with `GOOGLE_RPM_LIMIT` (`GEMINI_RPM_LIMIT`
under the legacy schema; `configured_google_rpm` is the one reader),
`OPENAI_RPM_LIMIT`, `ANTHROPIC_RPM_LIMIT`, `DEEPSEEK_RPM_LIMIT`, `ZAI_RPM_LIMIT`
and `XAI_RPM_LIMIT`, plus `MOONSHOT_RPM_LIMIT` for Kimi review bindings. Memory
embeddings use their provider's bucket too. A new bucket holds one token, so a
process's first request is not delayed by a refill interval. Every
shipped provider has a conservative application-side default; operators may raise
it only after checking the account's real quota. Direct model construction outside
a bound `RuntimeServices` scope uses the same provider-specific process fallback
instead of silently becoming unthrottled. `GOOGLE_SERVICE_TIER` and
`OPENAI_SERVICE_TIER` apply only to their provider, while the content-inspection
judge is always pinned to standard tier. In particular, Moonshot review bindings
use their own limiter, endpoint, compatible-client timeout, and seat retry policy;
they do not inherit OpenAI flex behavior merely because their transport class
implements the OpenAI chat protocol.

**A service tier and a client timeout are separate settings, in both schemas.**
Until Aug 2026 they were fused: `_apply_openai_service_tier` set the tier *and*
floored the SDK client timeout, so `OPENAI_SERVICE_TIER=flex` was the only way to
give a compatible vendor more than the OpenAI-shaped per-seat default. An operator
pointing `OPENAI_API_BASE` at Moonshot therefore had to enable a pricing product
Moonshot does not sell in order to buy a timeout — and the flag's real effect was
invisible from its name. Now `OPENAI_SERVICE_TIER` is ignored whenever
`OPENAI_API_BASE` names a non-OpenAI host (no tier, no flex fallback, no floor),
and `OPENAI_COMPATIBLE_CLIENT_TIMEOUT_SECONDS` (default 300) is the one knob for
how long a compatible client may wait, read by the legacy consultant/auditor/editor
path and the provider-scoped compatible adapter alike. The OpenAI-proper flex path
is unchanged. Note the ordering this restores: with the old 900 s floor and a
600 s `LLM_CALL_HARD_TIMEOUT_SECONDS`, the SDK timeout could never fire first, so
its latency-fallback branch was unreachable.

**Embeddings are not routed by `OPENAI_API_BASE`.** That setting selects the
review *chat* plane's compatible vendor; those vendors serve no embeddings API, so
inheriting it pointed every memory read/write at an endpoint that cannot answer —
and memory failures degrade quietly rather than raising. `build_embeddings` honors
an explicit `api.openai.com` base and ignores a compatible host. If an
OpenAI-compatible *embeddings* gateway is ever needed, add a dedicated setting
rather than reusing the chat one.

Seat execution policy owns stable sampling, timeout, SDK-retry, reasoning-control,
output-override, service-tier, and quick-mode availability facts. The permanent
offline parity matrix compares the normalized transport class, intent/API caps,
reasoning reserve and effort, Responses/compatible payload fields, timeout, and
SDK retries across every reachable Google base/operational/judge seat and the
OpenAI review plane. Auditor escalation is the documented exception: the
provider-scoped path intentionally raises it from legacy `medium` to the strongest
reviewed effort supported by its escalation model.

Independence uses one enforcement boolean plus audit metadata, not two opposing
booleans. A waiver reason is mandatory when enforcement is disabled. If a
historical reason remains configured after enforcement is re-enabled, it is
inactive, omitted from runtime telemetry, and does not make otherwise safe config
invalid.

### Model failover: the degradation ladder

On 2026-10-01 the quick-mode Portfolio Manager on `gemini-3.8-flash` hit
`504 DEADLINE_EXCEEDED` in two tickers, each time *after* the transport had already
dropped from flex to standard. The only fallback then was a tier change on the same
model, and the retry loop's next attempt repeated that same model. A newly released
model being short of capacity is a property of the model, so the remedy has to change
the model.

**Correction (2026-10-08).** Those quick-mode PM 504s were not a capacity problem
alone: our short deadline exposed slower service. The deadline is the 24 s quick
flex-attempt cap (0.4 × the 60 s quick hard cap). A flex-built
client carries it as its SDK timeout, google-genai sends it as `X-Server-Timeout` so
Google enforces it server-side, and the standard re-issue and the failover copy inherit
it from the instance. The deadline exposed slower `gemini-3.8-flash` service from
2026-10-06. Containment: the PM is pinned to standard in quick mode again (the 2026-09
unpin assumed it fit the flex cap), and the Google adapter now takes SDK retries from
the run-scoped config, so the `--quick` clamp of 2 attempts applies to CRITICAL and
REASONING seats too, not only FAST. The systemic fix -- a per-dispatch deadline
instead of one fixed at construction -- landed the same day; see *Request budgets*
below. The pins on the PM, Junior and Senior Fundamentals stay until measured latency,
cost per publishable completion and failure rate justify flex. The failover ladder
below still stands for genuine model-health failures.

A struggling call now goes down one ladder, each rung owned by the layer that already
owns that concern:

```
(primary, flex) --transport--> (primary, standard) --retry loop--> (failover, standard)
```

- **What** to fail over to is the model-keyed map `LLM_MODEL_FAILOVERS`, so every seat
  bound to the source model inherits it (PM, Senior Fundamentals, debate and risk
  alike). The binding plan validates it at startup. A target must have a reviewed
  profile and the same vendor, it must sit on a transport that implements
  `ModelFailoverMixin` (`google_native`, `openai_native`), it is one hop only, and it
  cannot map a model to itself. The failover copy keeps the source's request
  settings, so the target must also accept the reasoning value each seat actually
  sends, and share the source's `reasoning_api_mode`, `temperature_policy` and
  `token_parameter`. Otherwise the failover fails at exactly the moment it is
  needed. For example, `gpt-6.1-sol -> gpt-5.4` is rejected for a seat pinned to
  `max`. The target appears in the binding telemetry as `failover_model`.
- **When** is decided in `invoke_with_rate_limit_handling`. Only `server_error` and
  `timeout` count. A 429, a bad request or a safety block says nothing about the
  model's health. The failover attempt *replaces* a retry rather than adding one, and
  it does not back off, because a different model is not waiting out this one's
  outage.
- **How**: the transport serves the attempt from a `model_copy` bound to the target
  at the standard tier. Billing follows the response's `model_name`.

**The failover never targets the model that just failed.** Several separate
mechanisms guarantee this, and no runtime comparison is needed:
- A self-map is rejected at startup.
- An alias (`*-latest`) has no reviewed profile, so it cannot mask the same model
  under another name.
- The tier rung runs first, so a lone flex 504 recovers on the primary at standard
  without spending the failover.
- Once on the failover model, later retries stay there. They neither bounce back to
  the primary nor take a second hop.

Each guarantee has a test in `tests/llm_runtime/test_model_failover.py`.

The structural recovery path (`SeatId.ANALYST_RETRY`, for malformed output) is
separate and unchanged. Cross-vendor failover is out of scope, because it would
change the independence guarantees above.

**`failover_target` must walk only real `RunnableBinding`/`RunnableSequence`
nodes.** The first version probed `bound`/`first`/`last` with `getattr`. On a mock
every attribute access returns a new mock, so the walk never terminated. It runs on
every LLM invocation, so it hung any test that passed a mock LLM. That hang is what
killed the session that wrote it.

### Request budgets: one policy, an absolute deadline, per-dispatch sizing

The construction-time timeout was the shared defect behind the October PM 504s,
and it was not PM-specific: every flex seat and the OpenAI flex path built their
client with a flex-attempt timeout (24 s quick, 900 s full), while the tier is
chosen *per call*. Re-issue, degraded-standard calls and the failover copy all
inherited it. It is gone; `src/llm_runtime/timeouts.py` replaces it.

- **`TimeoutClass` on every `SeatSpec`** (`STANDARD`, `APEX`, `CROSS_CHECK`,
  `REGIONAL`, `EDITORIAL`, `OPERATIONAL`) replaces the hand-kept display-name sets
  that chose a seat's quick cap. `AUXILIARY` is reserved for contexts no seat
  claims and cannot be registered.
- **`BudgetPolicy`** is resolved from (class, run mode, provider): the per-attempt
  invocation cap (full-mode flex floor included), the flex request ceiling (quick:
  `FLEX_QUICK_ATTEMPT_FRACTION` x cap; full: `FLEX_LLM_TIMEOUT_SECONDS`), the
  standard request ceiling (the value clients were already built with), the minimum
  useful attempt and the margin, and `LLM_ATTEMPTS`. Both are derived: the
  minimum is `MIN_USEFUL_ATTEMPT_FRACTION` (0.04, chosen, not measured) of one
  attempt's configured allowance, `min(invocation cap, per-request ceiling)`
  before any flex floor, so neither a long invocation cap (an operator's 600 s)
  nor a flexed cap demands a huge minimum; the margin is `deadline_margin(cap)`, the same rule
  the tool plane uses (`DEADLINE_MARGIN_FRACTION` of the cap, at most
  `LLM_DISPATCH_MARGIN_SECONDS`, which stays as the absolute ceiling so a
  1350 s flexed cap keeps a 2 s margin, not 67 s). Shipped defaults: quick
  standard (cap 60 s) 2.4 s; quick apex or cross-check (cap 180 s) 4.8 s on
  OpenAI and 4.8-7.2 s elsewhere; full mode (cap 120 s) 4.8 s; editorial on
  Google or Anthropic (300 s) 12 s, the largest; embeddings (30 s) 1.2 s; MCP
  (45 s) 1.8 s. `tests/llm_runtime/test_min_useful_attempt.py` pins the whole
  class x provider x mode table. Because
  both flex and the minimum are shares of the quick cap, a quick flex request is
  skipped for lack of room only if `FLEX_QUICK_ATTEMPT_FRACTION` is set below
  `MIN_USEFUL_ATTEMPT_FRACTION`.
- **The deadline is absolute and lives in a ContextVar**, never on the model.
  `invoke_with_rate_limit_handling` sets `min(now + cap, enclosing)` per attempt,
  where *enclosing* is `overall_timeout_seconds` or a node scope
  (`enclosing_deadline`; the APAC node's `APAC_NODE_TOTAL_TIMEOUT_SECONDS` is one),
  itself capped by the pipeline's ticker run deadline when the watchdog supplies
  one (`enclosing_deadline_at`; see the watchdog contract in
  `docs/CODEBASE_MEMORY.md`).
  A retry or failover attempt gets a fresh allowance; nothing renews the enclosing
  deadline.
- **Every dispatch is admitted and sized**: flex, standard re-issue, a call sent at
  standard because flex is degraded, and the failover copy (which reads the same
  ContextVar, so it never gets a fresh clock). `remaining = deadline - now - margin`;
  below the minimum useful attempt the request is refused with `BudgetExhausted`
  before any transport call. A flex share below the minimum goes straight to
  standard. `BudgetExhausted` is a `TimeoutError` but is never a flex-health signal
  and never counted by the circuit breakers.
- **One retry owner.** Every request is one SDK attempt, on every vendor and
  every construction path. It is a property of the transport classes, not a
  constructor argument: the Gemini transport has `max_retries=1`
  (langchain-google-genai maps it to `HttpRetryOptions(attempts=1)`; `0` would
  mean the SDK's default), the OpenAI-family and Anthropic transports zero
  retries, and any other value fails construction. A per-call `max_retries` is
  refused at dispatch. The loop's one attempt count defaults to `LLM_ATTEMPTS`
  and covers every retry, rate-limit and transient alike (a caller that must not
  retry passes `max_attempts=1`), and every retry and backoff is re-admitted against the enclosing deadline: a
  transient backoff is clamped to fit, a rate-limit wait that leaves no useful
  attempt refuses the retry. The loop's backoffs (5 s × attempt plus 1-3 s jitter
  for a transient error, 60 s × attempt plus 1-10 s for a rate limit, the
  providers' per-minute quota window) stay literals: admission already clamps
  them, so a setting would add surface without adding a bound.
- **What "one SDK attempt" does not mean.** It means no SDK-level retry policy,
  not one physical request. google-genai's aiohttp path
  (`_api_client.py`, `_async_request_once`, installed 2.24.0) catches
  `aiohttp.ServerDisconnectedError`, connector, DNS and OS errors, sleeps 1-10 s
  and sends the same request again, outside `HttpRetryOptions`. One Gemini
  dispatch can therefore make two physical sends (never more: the resend is not
  repeated and no policy retry follows). The resend reuses the request's timeout,
  so it is bounded only by the caller's hard deadline, not re-sized against the
  remaining budget; the pre-resend sleep is cancellable.
  `tests/llm_runtime/test_google_disconnect_resend.py` pins both facts through
  the real chat transport with a fake at `aiohttp.ClientSession._request`; the
  embedding path has the same behaviour
  (`tests/memory/test_embedding_attempt_deadline.py`). Read request counts from
  the wire, not from the attempt count.
- **Why one owner (October 2026).** Until then a client's SDK attempts were
  decided by its seat's class: runtime-owned classes were built with one, the
  rest with `LLM_ATTEMPTS`, on the assumption that their callers invoked the
  model directly. None did any more: macro context, the injection judge and
  the signal processor all ran under the loop. On OpenAI, compatible vendors and
  Anthropic each of their loop attempts therefore sent up to `LLM_ATTEMPTS`
  requests (`LLM_ATTEMPTS²` in all), and each SDK re-send carried the timeout
  sized for the first request, so it ran past the attempt deadline (orphaned by
  the hard timeout, still billed). Google was unaffected inside a scope, where
  each dispatch sent `max_retries=1`. The per-seat `transport_attempts` (and its
  deprecated alias `sdk_max_retries`), the class membership that chose it, and
  every factory's attempt parameter are gone. The judge's context was
  "Prompt Injection Judge" while its seat said "Content Inspector", and the
  signal processor's "SignalProcessor": neither resolved to its seat, so both
  ran under the generic AUXILIARY policy. Both now pass the seat's display name,
  and the seat is named "Prompt Injection Judge", the label its token
  accounting always used. The judge still fails open on `BudgetExhausted`.
- **Former direct callers are on the policy.** The Article Editor's review (tool
  loop, structured and plain final review), the retrospective lesson and the
  semantic judge go through `invoke_with_rate_limit_handling`, keyed by their
  seat's display name, so they get the cap, the absolute deadline, per-dispatch
  sizing and the loop's retries; their seats are built with one SDK attempt. They
  were routed through the loop rather than given a bare `invocation_scope`,
  because inside a scope Google sends one SDK attempt per request: without the
  loop's retries these callers would have lost the resilience their SDK retries
  used to give. Each keeps its failure behaviour (editor: REVISE with
  `review_error`; lesson: `None`; judge: an exception, or HARD_FAIL for a
  refusal). The editor was uncapped; it now has the review-plane (`CROSS_CHECK`)
  budget, since it is bound to the same group as the Consultant and Auditor. The
  analyst structural retry reads its overall budget from the same resolver
  (`resolve_invocation_budget`) instead of recomputing it. Still direct: only
  the health check, which is bounded on its own (15 s overall, a 10 s client, one
  attempt) and runs at startup, outside every deadline.
- **The article workflow owns a deadline, and the writer runs on the loop.**
  `--article` runs after the analysis deadline's scope has closed, so the writer
  called its model with a synchronous `invoke` on the event loop, under no
  deadline, with SDK retries fixed on each client (three Anthropic requests of
  `API_TIMEOUT` each) and every fallback tier built and sent after any primary
  failure. Now `handle_article_generation` binds
  `ARTICLE_WORKFLOW_TIMEOUT_SECONDS` (900 s) around draft, review and
  revisions, and hard-bounds each step at it (`run_article_workflow_step`), so a
  step that overruns is cancelled and its result discarded, never written.
  Writing is a separate step after the bounded one (round 4): `ArticleWriter.draft`
  touches no file, and `publish_article` writes through
  `src/atomic_files.py::atomic_write_text` only once the draft has returned and
  the workflow deadline still holds. Before, `write(output_path=...)` and
  `generate_article` wrote from inside the bounded step, so a writer that
  swallowed cancellation could publish after its caller was told it timed out.
  The default is about twice the longest measured workflow: 333 to 444 s across 15
  June 2026 article runs (analysis save to article write). `write`, `revise` and
  the edit loop are async; each tier invokes through the runtime loop keyed by
  its seat (`Article Writer`, `Article Writer Fallback`). `TimeoutClass.EDITORIAL`
  is now runtime-owned (writer clients send one SDK attempt; the loop retries up
  to `LLM_ATTEMPTS`) and quick-invariant, with `API_TIMEOUT` per attempt; it was
  used only by the writer seats, so no other seat's policy moved. A fallback
  tier is not built once the allowance cannot fit a useful attempt. When the
  allowance runs out after a draft exists, the draft is saved as `.draft` with
  `editor_review_error` and `workflow_deadline_exceeded` set, and
  `writer_fell_back` kept; a review error never publishes. The workflow
  deadline is not a promise that an article finishes, and the pipeline never
  passes `--article`.
- **A direct caller under an enclosing deadline is sized from it.** With no
  invocation scope, a dispatch now admits and sizes its request against
  `current_enclosing_deadline()` under the model's class policy, as one SDK
  attempt like every request. With neither a scope nor an enclosing deadline
  nothing changes.
- **The portfolio macro classifier** (`scripts/portfolio_manager.py`) invokes
  through the runtime loop (its own `asyncio.run`, since the script's `main` is
  synchronous) and its seat sends one SDK attempt per request.
- **The seat, not just its class, resolves the policy.** Inside an invocation the
  runtime looks up the registered seat (`seat_for_agent`) and resolves through
  `policy_for_seat`, so a seat's explicit `client_timeout_seconds` (the Health
  Check's 10 s) is the request ceiling there too; resolving the class alone had
  dropped it for every scoped dispatch.
- **The ceiling follows the vendor that receives the request.** A dispatch keeps
  the invocation's deadline but re-derives the per-request ceiling for its own
  provider (`policy_for_provider`). Model failover is same-vendor by validation,
  so a delegate on another vendor is not reachable today; this keeps a future
  cross-vendor delegate from sending the original vendor's ceiling.
- **An explicit client timeout reaches the request (round 4).** The scope
  resolves the *registered* seat by display name, so a client built with an
  override (`seat_policy_with(..., client_timeout_seconds=111)`) carried 111 s
  only as its construction timeout: every dispatch re-sized from the scope and
  sent about 298 s; a direct caller under an enclosing deadline sent the class
  ceiling (300 s), and so did the Health Check's registered 10 s. Now the two
  construction choke points (`SeatModelFactory.build`, `build_legacy_model`)
  stamp the effective `SeatExecutionPolicy.client_timeout_seconds` on the
  transport as `request_cap_override`, and every dispatch resolves one chain in
  `policy_for_provider`: the model's override, then the canonical seat's (the
  scope), then the receiving vendor's ceiling -- always bounded by the inherited
  deadline. An override replaces the ceiling, as a registered one always did,
  so it may exceed the vendor's. A shorter ceiling lowers the minimum useful
  attempt to its share (`min_useful_seconds`). The field survives `model_copy`,
  the failover copy and every canonical writer fallback tier; settings are still
  read at dispatch, never frozen on the model. The legacy writer chain refuses
  any override but the temperature, and the writer seam no longer takes a
  `timeout` (no production caller passed one).
- **Every OpenAI-family and Anthropic client is deadline-aware, and names its
  vendor.** Until October 2026 only the flex-tier OpenAI client and Gemini went
  through the dispatch check. The canonical OpenAI adapter without a failover,
  every OpenAI-compatible client (DeepSeek, z.ai, Moonshot, xAI), the
  Consultant's alternate transport, the legacy review-plane factories, the APAC
  factory and both Anthropic writers were plain SDK classes that sent their
  construction-time timeout inside an invocation. Now every one is built by
  `new_openai_chat` / `new_anthropic_chat` in
  `src/llm_runtime/adapters/transports.py`, and the shared sizing lives in the
  SDK-free `src/llm_runtime/dispatch.py`. Each constructor passes
  `dispatch_provider` explicitly; a missing or blank one is a construction error.
  The factory's provider stamp arrives only after construction, and legacy
  clients never get it, so it cannot serve. The provider was hardcoded to
  `"openai"` in the one deadline-aware OpenAI class, so a compatible vendor was
  sized against the 120 s OpenAI request cap, not its
  `OPENAI_COMPATIBLE_CLIENT_TIMEOUT_SECONDS` ceiling, and keyed into OpenAI's flex
  health. A non-OpenAI host on the legacy path or the alternate transport is
  `openai_compatible`. A compatible vendor never carries a service tier: it
  cannot be built with one, its failover copy sends none, and a tier passed at
  invocation or through bound kwargs is refused before anything is sent (round
  4; it used to reach the vendor). An explicit `service_tier=None` sends no field. Anthropic takes the
  sized `timeout` per request on both its sync and async paths. **Behaviour change:** inside an invocation these clients
  now send a request sized from the deadline, and the Consultant's alternate
  transport on a compatible base is built with the compatible ceiling instead of
  the OpenAI one. Direct callers are unchanged: outside a scope a standard
  request keeps its built timeout.
- **The last policy constants are settings or derived (round 2, Step 7).**
  (Step 7 also derived direct-caller SDK attempts from `LLM_ATTEMPTS`; that was
  superseded by the single retry owner above.) The OpenAI standard request ceiling is
  `OPENAI_REQUEST_CAP_SECONDS` (120 s), read at the call. The APAC ceiling is the
  APAC node total, `APAC_NODE_TOTAL_TIMEOUT_SECONDS` (240 s), on the client (both
  schemas) and inside an invocation (`REGIONAL` on a compatible vendor): the
  client had a separate 240 s constant while the in-scope ceiling was the generic
  300 s compatible one. No request can usefully outlive the node's enclosing
  deadline, so the node total is the natural ceiling; changing it moves both.
  At shipped defaults no timeout value changes; inside the APAC node the 300 s
  in-scope ceiling never bound, because the 240 s node deadline was tighter.
- **Streaming is refused on the deadline-aware classes** (Gemini included). No
  application code streams, and a stream would bypass the per-dispatch sizing.
  `disable_streaming` routes `.stream()`/`.astream()` through the sized
  non-streaming path. `_stream`/`_astream`, a `stream=True` call argument and
  `streaming=True` at construction all raise. That last one matters because
  ChatOpenAI puts `stream: true` in its default request parameters, so even its
  non-streaming `_generate` would open a stream.
- **`--quick` no longer clamps `LLM_CALL_HARD_TIMEOUT_SECONDS`.** Every quick cap
  comes from the seat's class; the clamp's last readers (retrospective and judge)
  are on the resolver, so the full-mode value is read only in full mode.
- **Telemetry**: each `call_attempts` entry carries `dispatches` -- phase, tier,
  model, timeout sent, remaining allowance, outcome, elapsed. Each is one SDK
  attempt; no payloads.

**Attempt counts have one meaning: total attempts.** `API_RETRY_ATTEMPTS` became
`LLM_ATTEMPTS` (the old name is still read; the new one wins; the value carries over
unchanged because Google, its main consumer, already counted totals). It is the
runtime loop's budget only: no client takes an attempt count (see "One retry
owner" above).

**Verified against the installed SDKs** (google-genai 2.24.0, langchain-google-genai
4.4.0, langchain-openai 1.6.2, openai 3.14.1): a per-request `timeout` reaches the
request layer on Google (`X-Server-Timeout`, the ceiling of the seconds) and on both
OpenAI request paths (Responses and Chat Completions `create(timeout=...)`); OpenAI
has no per-request `max_retries`. `tests/llm_runtime/test_timeouts.py` pins both
through the SDKs' own request layers. With langchain-anthropic 1.7.2 and anthropic
1.6.0, a per-call `timeout` passes through `_get_request_payload` into
`messages.create(timeout=...)` on both paths; `create()` rejects `max_retries`.
`tests/llm_runtime/test_transport_deadlines.py` pins every seat on every vendor
its binding group allows, read from the real registries, at the request layer.

## Embedding collections

Embedding identity is independent from chat identity. Collection names include a
hash of provider, model, dimension, and schema version. A switch creates or reuses
that exact collection and leaves old and unstamped collections untouched.

```bash
# Read-only inventory
poetry run python scripts/embedding_collections.py

# Explicitly create an empty target for the configured identity
poetry run python scripts/embedding_collections.py --initialize-current lessons_learned
```

The tool never deletes or re-embeds. Historical `lessons_learned` content may not be
reconstructible, so any future migration command must require an explicit reviewed
source corpus and destination.
