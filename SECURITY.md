# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          | Notes |
| ------- | ------------------ | ----- |
| main (latest) | :white_check_mark: | Active development - recommended |
| < 1.0 (dev) | :white_check_mark: | Pre-release - use with caution |

**Note:** This project is in active development. We recommend using the latest commit from `main` and monitoring releases.

---

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### Preferred Method: Private Security Advisory

1. Go to [Security Advisories](https://github.com/rgoerwit/ai-investment-agent/security/advisories)
2. Click "Report a vulnerability"
3. Provide detailed information about the issue
4. We aim to respond within 48-72 hours

### Alternative: Email Disclosure

If you prefer email or cannot use GitHub Security Advisories:
- **Contact:** Open an issue requesting secure contact information
- **PGP Key:** Available on request

**Please DO NOT:**
- Report security vulnerabilities through public GitHub issues
- Disclose vulnerabilities publicly before we've addressed them
- Exploit vulnerabilities for malicious purposes

### What to Include in Reports

- **Description:** Clear explanation of the vulnerability
- **Reproduction Steps:** How to reproduce the issue
- **Impact Assessment:** Potential consequences (data exposure, code execution, etc.)
- **Affected Versions:** Which versions/commits are vulnerable
- **Suggested Fix:** If you have one (optional)
- **Your Contact Info:** For follow-up questions

---

## Response Timeline

- **Initial Response:** Within 72 hours
- **Status Update:** Within 7 days
- **Fix Timeline:** Depends on severity
  - **Critical:** 7 days (API key exposure, remote code execution)
  - **High:** 14 days (authentication bypass, data leakage)
  - **Medium:** 30 days (denial of service, information disclosure)
  - **Low:** Next release cycle (minor issues)

---

## Security Considerations for Users

### 1. API Key Security ⚠️ **CRITICAL**

This application requires multiple API keys stored in a `.env` file. **Protecting these keys is YOUR responsibility.**

#### Required API Keys:

```bash
# .env file structure
GOOGLE_API_KEY=AIza...            # Google Gemini (required)
FINNHUB_API_KEY=...               # Market data (required)
TAVILY_API_KEY=tvly-...           # Web search (required)

# Optional but recommended
EODHD_API_KEY=...                 # High-quality international data
FMP_API_KEY=...                   # Financial Modeling Prep fallback
OPENAI_API_KEY=sk-...             # External consultant validation
LANGSMITH_API_KEY=lsv2_pt_...     # LangChain tracing/debugging
```

#### API Key Best Practices:

✅ **DO:**
- Keep `.env` file local (already in `.gitignore`)
- Use environment variables in production/CI
- Rotate API keys every 90 days
- Use API keys with minimum required permissions
- Monitor API usage dashboards for unauthorized activity
- Revoke keys immediately if compromised

❌ **DO NOT:**
- Commit `.env` file to version control
- Share API keys in public forums/chat/screenshots
- Use production keys for development/testing
- Store keys in application logs or error messages
- Use root/admin API keys (create restricted keys)

#### Key Compromise Response:

If you suspect API key compromise:
1. **Immediately revoke** the compromised key at the provider
2. **Generate a new key** with fresh credentials
3. **Review API usage logs** for unauthorized activity
4. **Check for data exfiltration** (unusual queries, high volume)
5. **Report to the provider** if fraudulent usage occurred

### 2. Data Privacy & Storage

#### Local Data Storage:

This application stores data **entirely on your local machine**:

- **ChromaDB vectors:** `chroma_db/` (agent memory, embeddings)
- **Analysis results:** `results/` (markdown reports)
- **Data cache:** `data_cache/` (temporary API responses)
- **Logs:** `stderr` output (structured logs, no secrets)

**No external data transmission** except:
- API calls to fetch financial data (ticker symbols visible to providers)
- LangSmith tracing (if explicitly enabled via `LANGSMITH_API_KEY`)

#### Privacy Recommendations:

- **Sensitive portfolios:** Do not share analysis results containing proprietary holdings
- **Confidential research:** Keep `results/` directory private (not in git)
- **Ticker privacy:** Be aware API providers log requested ticker symbols
- **Audit trail:** Review `.gitignore` to ensure no sensitive data is committed

#### Data Deletion:

To completely remove analysis data:
```bash
rm -rf chroma_db/       # Delete all agent memories
rm -rf data_cache/      # Delete cached API responses
rm -rf results/         # Delete analysis reports
```

### 3. Financial Data Disclaimer ⚠️ **IMPORTANT**

This software is for **research and educational purposes ONLY**.

#### Critical Warnings:

❌ **NOT financial advice** - Do not treat AI-generated analysis as professional investment recommendations
❌ **NOT suitable for automated trading** - Requires human review and verification
❌ **NOT guaranteed accurate** - AI can hallucinate data, misinterpret news, or make calculation errors
❌ **NOT real-time** - Free API data may be delayed 15+ minutes
❌ **NOT comprehensive** - Free APIs have coverage gaps for international stocks

#### AI/LLM-Specific Risks:

- **Hallucination:** LLMs can fabricate financial metrics, news events, or reasoning
- **Recency Bias:** Training data cutoffs mean LLMs lack knowledge of very recent events
- **Confirmation Bias:** Multi-agent debate helps but doesn't eliminate bias
- **Context Confusion:** Ticker-specific memory isolation prevents but doesn't eliminate cross-contamination
- **Prompt Injection:** Malicious news articles or company filings could influence analysis

#### Your Responsibilities:

✅ **Always verify** critical facts with authoritative sources (SEC filings, Bloomberg, company IR)
✅ **Cross-check metrics** using multiple data providers
✅ **Review reasoning** - don't blindly accept BUY/SELL recommendations
✅ **Understand limitations** - this tool generates research ideas, not final decisions
✅ **Comply with regulations** - consult tax/legal advisors for cross-border investing

### 4. Third-Party Dependency Vulnerabilities

#### Known High-Risk Dependencies:

**ChromaDB (~61 vulnerabilities as of Dec 2024):**
- **Risk:** Vector database with known security issues
- **Mitigation:**
  - Runs locally with no network exposure
  - Isolated in virtual environment (`poetry` sandboxing)
  - No public internet access to ChromaDB port
  - We monitor for updates and apply patches
- **Status:** Accepted risk (benefits outweigh for local-only use)

**Other Dependencies:**
- **LangChain/LangGraph:** Actively maintained, security patches applied
- **Google/OpenAI SDKs:** Official SDKs, vendor-maintained
- **Financial data libraries:** Free APIs have limited security guarantees

#### Mitigation Strategies:

```bash
# Keep dependencies updated
poetry update

# Review security advisories
poetry show --outdated

# Check for known vulnerabilities
# (Consider adding: pip install safety && safety check)
```

**OpenSSF Scorecard Results:** See latest scan at repository homepage for dependency health scores.

### 5. External API Security Risks

#### Data Provider Risks:

- **Request Logging:** API providers may log ticker symbols, timestamps, source IPs
- **Rate Limiting:** Aggressive usage can trigger account suspension
- **Data Integrity:** Free APIs may return stale, incomplete, or incorrect data
- **Service Outages:** Providers can go offline, change pricing, or deprecate endpoints
- **Terms of Service:** Ensure your usage complies with provider ToS (especially for commercial use)

#### API-Specific Considerations:

| Provider | Risk Level | Notes |
|----------|-----------|-------|
| Google Gemini | Low-Medium | Requests logged by Google, subject to usage policies |
| OpenAI (optional) | Low-Medium | Similar logging, rate limits, content policy |
| yfinance | Low | Unofficial Yahoo Finance scraper, no ToS guarantee |
| EODHD/FMP | Low | Paid APIs with SLAs, better reliability |
| Tavily | Medium | Web search may return malicious/misleading content |
| StockTwits | Medium | Social media API, content moderation varies |

### 6. Running Untrusted Code & LLM Output

#### AI-Generated Content Risks:

⚠️ This application executes reasoning from large language models. While we use structured prompts and validation:

- **Unexpected patterns:** LLMs can generate surprising or malicious-looking text
- **Data parsing errors:** Malformed output could cause application crashes
- **Prompt injection:** Malicious news articles could manipulate LLM reasoning
- **Tool misuse:** LLMs decide which tools to call (validated but not foolproof)

#### Protection Mechanisms:

✅ **Structured output formats:** Agents must follow strict markdown templates
✅ **Input validation:** Ticker symbols sanitized in `ticker_utils.py`
✅ **Output sanitization:** Markdown escaping prevents code injection in reports
✅ **No code execution:** Analysis results are text reports, not executable code
✅ **Tool whitelisting:** Only approved functions callable by agents

#### User Precautions:

- Review analysis results before acting on recommendations
- Be skeptical of extreme/sensational claims in agent reasoning
- Verify financial metrics against primary sources
- Report suspicious LLM behavior (hallucinations, biases, errors)

---

## Secure Development Practices

### For Contributors:

#### Pre-Commit Security Checks:

```bash
# Run full quality suite before committing
poetry run ruff check src/          # Linting (security rules enabled)
poetry run black --check src/       # Formatting
poetry run mypy src/                # Type checking (catches type confusion)
poetry run pytest tests/ -v         # Security-critical test coverage
git diff --check                    # Detect merge conflicts
```

#### Code Review Checklist:

- [ ] No hardcoded API keys, passwords, or secrets
- [ ] Input validation for user-provided data (tickers, file paths)
- [ ] Output sanitization for LLM-generated content
- [ ] Dependency updates reviewed for breaking changes
- [ ] Tests cover security-critical code paths
- [ ] No use of `eval()`, `exec()`, or dynamic imports
- [ ] File operations use safe paths (no directory traversal)

#### Security Guidelines:

1. **Input Validation:** Always validate ticker symbols, API responses, user inputs
2. **Secrets Management:** Never hardcode credentials - use environment variables
3. **Dependency Hygiene:** Pin versions in `poetry.lock`, review updates carefully
4. **Error Handling:** Don't leak sensitive info in error messages or logs
5. **Least Privilege:** Use minimal API permissions, restricted file paths
6. **Defense in Depth:** Multiple validation layers (prompts + code + tests)

---

## Disclosure Policy

When a security vulnerability is reported:

1. **Acknowledgment:** Receipt confirmed within 48-72 hours
2. **Triage:** Severity assessment and validation (timeline varies)
3. **Fix Development:** Patch created in private branch
4. **Testing:** Security fix verified with tests
5. **Coordinated Disclosure:** Public disclosure coordinated with reporter
6. **Credit:** Reporter credited in release notes (unless anonymity requested)
7. **Advisory:** GitHub Security Advisory published with CVE (if applicable)

### Severity Classification:

| Level | Examples | Response Time |
|-------|----------|---------------|
| **Critical** | API key exposure, remote code execution, data exfiltration | 7 days |
| **High** | Authentication bypass, privilege escalation, significant data leak | 14 days |
| **Medium** | Denial of service, information disclosure, minor data leak | 30 days |
| **Low** | Configuration issues, minor bugs with security implications | Next release |

---

## Out of Scope

The following are **NOT considered security vulnerabilities**:

❌ Issues in third-party APIs (yfinance, FMP, EODHD, Tavily)
❌ Rate limiting or API quota exhaustion
❌ Market data inaccuracies or delays
❌ Investment losses (this is a research tool, not financial advice)
❌ Performance issues or high token costs
❌ LLM hallucinations (expected behavior - user must verify)
❌ Social engineering attacks (user responsibility to protect API keys)
❌ Browser/OS vulnerabilities (use updated systems)

---

## Security Features

### Current Protections:

✅ **API Key Isolation:** `.env` file (gitignored) for credentials
✅ **Input Validation:** Ticker sanitization in `ticker_utils.py`
✅ **Output Sanitization:** Markdown escaping, no code execution in reports
✅ **Rate Limiting:** Built-in API rate limit handling
✅ **Memory Isolation:** Ticker-specific ChromaDB collections
✅ **Error Handling:** Graceful degradation when APIs fail
✅ **Logging:** Structured logs to stderr (no secrets)
✅ **Virtual Environment:** Poetry isolation from system Python
✅ **GitLeaks:** Secret scanning in CI/CD (when configured)
✅ **Dependabot:** Automated dependency updates (GitHub)

### Security Tooling:

Enabled in repository:
- **GitLeaks:** Pre-commit hook scans for secrets
- **Trivy:** Docker/dependency vulnerability scanning
- **Dependabot:** Automated security update PRs
- **Branch Protection:** Required checks before merge (when configured)

### Roadmap (Future Enhancements):

- [ ] **SAST:** CodeQL static analysis in GitHub Actions
- [ ] **Secrets Scanning:** GitHub Advanced Security
- [ ] **Vulnerability Scanning:** Integrate `safety` or `pip-audit`
- [ ] **API Key Rotation:** Automated rotation reminders
- [ ] **Audit Logging:** Track sensitive operations (ticker analyses, API calls)
- [ ] **Sandboxing:** Docker-based isolation for production deployments
- [ ] **MFA for Deployment:** Require 2FA for release/publish

---

## Resources

### Security Best Practices:

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/)
- [Python Security Warnings](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [LangChain Security Guidelines](https://python.langchain.com/docs/security)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

### AI/LLM Security:

- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Google Responsible AI Practices](https://ai.google/responsibility/responsible-ai-practices/)
- [OpenAI API Security Best Practices](https://platform.openai.com/docs/guides/safety-best-practices)

### Financial Data Security:

- [SEC Cybersecurity Guidance](https://www.sec.gov/cybersecurity)
- [FINRA Technology & Cybersecurity](https://www.finra.org/rules-guidance/key-topics/cybersecurity)

---

## Acknowledgments

We appreciate responsible security researchers who help make this project safer. Contributors who report valid vulnerabilities will be credited in:

- This SECURITY.md file (Hall of Fame section - to be added)
- Release notes for security fixes
- GitHub Security Advisories

### Hall of Fame:

*No security vulnerabilities reported yet. Be the first!*

---

## License & Disclaimer

**License:** MIT - See [LICENSE](LICENSE) for full terms.

**Security Disclaimer:** This security policy does not constitute:
- A bug bounty program (no financial rewards)
- A guarantee of perfect security (no software is 100% secure)
- Legal liability for damages (use at your own risk)
- Professional security audit (community-driven best effort)

**Use at Your Own Risk:** This project is provided "AS IS" without warranty. Users assume all risks associated with financial analysis, data accuracy, and investment decisions.

---

**Last Updated:** December 14, 2024
**Policy Version:** 2.0
**Changes from v1.0:** Added comprehensive API key security, AI/LLM risks, dependency vulnerabilities, financial disclaimer, and expanded best practices.
