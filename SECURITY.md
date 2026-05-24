# Security Policy

This is a local research system that uses LLMs, market-data APIs, web search,
cached memory, saved analysis files, and optional broker/account context. That
is a lot of surface area for a personal tool. Treat it accordingly.

The short version:

- keep secrets out of git
- assume external text can be hostile
- assume market data can be stale or wrong
- do not expose local runtime services to the public internet
- verify important investment facts outside the system

## Supported Versions

| Version | Supported | Notes |
| --- | --- | --- |
| `main` | Yes | Active development. Use the latest commit. |

This project does not maintain long-lived release branches.

## Reporting a Vulnerability

Use GitHub Security Advisories if possible:

1. Open the repository's Security tab.
2. Choose "Report a vulnerability".
3. Include a clear description, reproduction steps, impact, affected commit or
   branch, and any suggested fix.

If GitHub advisories are not workable, open a public issue asking for secure
contact information. Do not put exploit details, secrets, or account-specific
data in a public issue.

Expected response:

- initial response within 72 hours
- status update within 7 days
- fix timing depends on severity

## Secrets

The normal local setup reads credentials from `.env`. The example file is
[`.env.example`](.env.example).

Required for the main analysis path:

```bash
GOOGLE_API_KEY=...
FINNHUB_API_KEY=...
TAVILY_API_KEY=...
```

Common optional keys:

```bash
EODHD_API_KEY=...
FMP_API_KEY=...
OPENAI_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGSMITH_API_KEY=...
EDINET_API_KEY=...
MCP server keys such as FMP_API_KEY or TWELVE_DATA_API_KEY
```

Do not commit `.env`, local MCP registries, IBKR credentials, private keys,
analysis outputs containing holdings, or copied terminal logs with secrets.

If a key leaks:

1. Revoke it at the provider.
2. Create a new key.
3. Review provider usage logs.
4. Check the repo and shell history for accidental copies.

## Local Data

The system writes useful local state. Some of it may be sensitive.

Important paths:

- `results/` - saved reports, analysis JSON, charts, macro-context cache
- `scratch/` - screening lists, quick-screen outputs, pipeline intermediates
- `chroma_db/` - local vector memory
- `data_cache/` - provider/cache data
- `runtime/` - dashboard job DB, MCP usage DB, settings
- `.results.latest_analyses_index.json` and lock files - latest-analysis cache
- `config/mcp_servers.json` - local MCP registry, intentionally not committed

The system also sends data outward when you use provider features:

- ticker symbols and queries to market-data providers
- web/search queries to Tavily or similar services
- prompt and output content to LLM providers
- optional traces to Langfuse or LangSmith
- optional MCP requests to configured MCP servers
- optional IBKR requests when broker workflows are enabled

Do not treat ticker interest, watchlists, holdings, or generated analysis as
private once you send them to a third-party API.

To remove local state:

```bash
rm -rf chroma_db/
rm -rf data_cache/
rm -rf results/
rm -rf scratch/
rm -rf runtime/
rm -f .results.latest_analyses_index.json .results.latest_analyses_index.lock
```

Do that only when you mean it. These paths are not just disposable logs.

## Untrusted Content

The agents read external text: news, search results, filings, social content,
financial API free-text fields, MCP output, cached macro context, and retrieved
memory. Any of that text can contain prompt-injection attempts.

The current codebase has an optional untrusted-content inspection layer. It can
inspect web/search output, memory writes, selected financial API fields, cached
context, MCP tool output, and other prompt-visible material.

Recommended first-run posture:

```bash
UNTRUSTED_CONTENT_INSPECTION_ENABLED=true
UNTRUSTED_CONTENT_BACKEND=python
UNTRUSTED_CONTENT_INSPECTION_MODE=warn
UNTRUSTED_CONTENT_FAIL_POLICY=fail_open
```

Start with `warn`. Read the logs. Move to stricter modes only after you
understand the false positives for your workflow.

Run the adversarial/security tests before changing inspection, tool execution,
memory ingress, or prompt-visible external content:

```bash
make security-tests
```

## AI Security Coverage

This repo is not a security product, but it deliberately addresses the main
application-layer risks in the
[OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/).
The short map:

- **Prompt injection**: optional ingress inspection, tool-output inspection,
  adversarial tests, and clear trust boundaries around external text.
- **Sensitive information disclosure**: `.env`-based secrets, ignored local MCP
  registries, sanitized logs/errors, and warnings around broker/account data.
- **Supply chain**: Poetry-managed dependencies, lockfile-based installs, and
  local checks with pytest, ruff, Gitleaks/Trivy config, and pre-commit hooks.
- **Data and model poisoning**: inspected memory writes, source-aware handling
  for cached/replayed context, and adversarial corpus tests for memory paths.
- **Improper output handling**: reports are research artifacts; deterministic
  validators, artifact-status metadata, and PM fast-fail paths keep some
  malformed or high-risk outputs from being treated as normal recommendations.
- **Excessive agency**: broker workflows are advisory, dashboard trading is
  read-only, MCP is disabled by default, and consultant MCP access is scoped and
  budgeted.
- **System prompt leakage**: prompts are not treated as secrets; logs and docs
  should still avoid exposing provider keys, local account details, and private
  operator context.
- **Vector and embedding weaknesses**: Chroma-backed memory is local,
  ticker-scoped where practical, and treated as untrusted when replayed into
  prompts.
- **Misinformation**: multi-agent disagreement, consultant/auditor paths,
  deterministic financial checks, caveats, and primary-source verification
  guidance all exist because confident prose is not proof.
- **Unbounded consumption**: quick mode, optional-agent switches, provider
  timeouts, MCP budgets, cached macro context, and cost/token reporting limit
  runaway local runs.

Other practical risks matter too: local files can contain holdings, provider
queries reveal interest, traces may contain prompt/output content, and a local
dashboard is still a data exposure if you bind it carelessly.

## Broker and Dashboard Workflows

The IBKR and dashboard paths are operator tools. Keep them local.

- The Flask dashboard is intended for local use, usually
  `http://127.0.0.1:5050`.
- Do not expose it directly to the internet.
- The dashboard is read-only for trading, but it can still reveal holdings,
  watchlists, cash state, order context, and recommendation history.
- `scripts/portfolio_manager.py` and `src/ibkr/` consume saved analysis JSONs
  and optional live broker context. Treat their outputs as sensitive.
- Queued dashboard refresh jobs live under `runtime/ibkr_dashboard/`.

Order execution is currently disabled; the system is advisory. That does not
make the data harmless.

## MCP

MCP support is disabled by default. When enabled, consultant MCP access is meant
to be narrow and vendor-specific, not a general-purpose tool tunnel.

Use `config/mcp_servers.example.json` as the template and keep the real
`config/mcp_servers.json` local. Give each server a narrow allowlist and a
reasonable per-run and daily budget. MCP output is external content and should
be inspected before it is shown to an LLM.

Use the smoke test when debugging MCP:

```bash
poetry run python scripts/mcp_smoke.py
```

## Financial and LLM Risks

This project is for research. It is not financial advice and it is not an
automated trading system.

Specific failure modes to expect:

- LLMs can hallucinate facts, citations, numbers, and causal explanations.
- Free or low-cost data providers can return stale, incomplete, or corrupt data.
- International coverage is uneven.
- Ticker symbols and currency units are easy to mishandle.
- Multi-agent debate reduces some bias, but it does not eliminate bias.
- Prompt injection can still slip through.
- Saved memory can preserve bad conclusions if bad content gets in.

Verify important claims against primary sources: filings, exchange pages,
company investor-relations material, and broker/platform data. Use the generated
analysis as a disciplined first pass, not as an authority.

## Dependencies and Local Runtime

Use Poetry and keep dependencies current:

```bash
poetry install
poetry show --outdated
```

Security-relevant local checks:

```bash
make security-tests
poetry run pytest tests/ -v
poetry run ruff check src/ tests/
git diff --check
```

The repo also includes Gitleaks/Trivy-related configuration and pre-commit
hooks, but local configuration varies. Do not assume a scanner has saved you.

## Contributor Checklist

Before sending a security-sensitive change, check:

- no secrets, tokens, private keys, or local account IDs are committed
- logs and errors do not expose raw secrets, raw provider payloads, or sensitive paths
- new external-content paths use the tool execution or inspection seams
- tests cover blocked, warned, sanitized, and failure cases where relevant
- broker/dashboard changes do not expose live account context accidentally
- file writes stay inside intended local paths
- public docs do not imply the tool is safe for automated trading

## Out of Scope

These are real risks, but they are not usually project security bugs:

- bad or delayed market data from third-party providers
- provider quota exhaustion
- investment losses
- model hallucinations that do not bypass a security boundary
- provider outages or pricing changes
- social engineering of the local operator
- vulnerabilities in your OS, browser, broker account, or API provider account

## License and Disclaimer

MIT license. See [LICENSE](LICENSE).

This policy is not a bug bounty program, a professional security audit, or a
promise of perfect security. Use the project with the same caution you would
use for any local tool that handles credentials, external content, and financial
research.

Last updated: 2026-05-24
