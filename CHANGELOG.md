# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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