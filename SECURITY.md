# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, report them using one of these methods:

### Private Vulnerability Reporting (Preferred)

Use GitHub's [private vulnerability reporting](https://github.com/rgoerwit/investment-agent-public/security/advisories/new) feature. This allows you to privately disclose security issues directly to the maintainers.

### Email Disclosure

Alternatively, email security concerns to the repository owner. Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Response Timeline

- **Initial Response**: Within 72 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: 7 days
  - High: 14 days
  - Medium/Low: 30 days

## Security Best Practices

When using this system:

1. **Protect API Keys**: Never commit API keys to version control
2. **Use Environment Variables**: Store sensitive data in `.env` files (gitignored)
3. **Review Dependencies**: Dependabot monitors vulnerabilities automatically
4. **Update Regularly**: Keep dependencies current with security patches
5. **Sandbox Execution**: Run in isolated environments for production use

## Disclosure Policy

Once a vulnerability is fixed:

1. A security advisory will be published on GitHub
2. The fix will be released in a patch version
3. Credit will be given to the reporter (unless anonymity requested)

## Out of Scope

The following are **not** considered security vulnerabilities:

- Issues in third-party APIs (yfinance, FMP, EODHD, Tavily)
- Rate limiting or API quota exhaustion
- Market data inaccuracies
- Investment losses (this is a research tool, not financial advice)
- Performance issues or high token costs

## Security Features

This repository includes:

- GitLeaks secret scanning (CI/CD)
- Trivy vulnerability scanning (dependencies, Docker)
- Dependabot automated security updates
- GitHub secret scanning (when enabled)
- Branch protection with required security checks
