# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
