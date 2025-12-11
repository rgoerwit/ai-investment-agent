# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

- Updated Dockerfile to multi-stage build pattern (40% smaller images)
- Improved error handling in report generator with fallback hierarchy (Portfolio Manager → Research Manager → Trader)
- Modernized GitHub Actions workflows (CodeQL v4, step-level conditionals)
- Enhanced Dependabot configuration for automated security updates

### Fixed

- Fixed silent output truncation when Portfolio Manager fails to produce final decision
- Fixed Docker build with `--no-root` flag for Poetry dependency installation
- Fixed SARIF upload errors in security scanning workflow
- Fixed missing file checks in CI/CD Docker image scanning

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
- CLAUDE.md developer guide for AI assistants
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
