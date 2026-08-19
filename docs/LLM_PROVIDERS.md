# LLM Provider Qualification and Operations

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

Transport capability and application qualification are separate contracts. For
example, current Claude profiles accurately record tool use, structured output,
and version-specific effort controls, but Anthropic remains writer-only until its
analytical seats pass the qualification procedure below. Binding validation checks
both layers and reports an unqualified provider/group assignment before model
construction.

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
host. Configure the native defaults with `GOOGLE_RPM_LIMIT`,
`OPENAI_RPM_LIMIT`, `ANTHROPIC_RPM_LIMIT`, `DEEPSEEK_RPM_LIMIT`, and
`ZAI_RPM_LIMIT`, plus `MOONSHOT_RPM_LIMIT` for Kimi review bindings. Every
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
