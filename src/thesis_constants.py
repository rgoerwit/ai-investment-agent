"""Canonical numeric thresholds for the GARP investment thesis.

Single source of truth for thresholds that also appear as prose inside
prompts/*.json. tests/prompts/test_threshold_parity.py asserts the prompt
text stays in sync with these values — update both together.
"""

# Liquidity (USD average daily turnover)
LIQUIDITY_MIN_USD = 100_000  # below: hard fail
LIQUIDITY_PASS_USD = (
    250_000  # at/above: full pass; between = MARGINAL (max 3% position)
)

# Valuation
PE_MAX = 18.0
PEG_MAX = 1.2
PB_MAX = 1.4
SECTOR_MEDIAN_PE: dict[str, float] = {
    "Energy": 12.0,
    "Materials": 14.0,
    "Industrials": 17.0,
    "Consumer Discretionary": 18.0,
    "Consumer Staples": 20.0,
    "Health Care": 22.0,
    "Financials": 12.0,
    "Information Technology": 25.0,
    "Communication Services": 20.0,
    "Utilities": 16.0,
    "Real Estate": 18.0,
}
PE_VS_SECTOR_RICH = 1.30

# Business-model-aware margin floors for the distribution model (thin margin,
# high asset turnover). The standard scoring bars (operating margin >12%, gross
# margin >30%) penalize structurally-thin distributors that earn returns through
# turnover, not margin (APR.WA). These relaxed floors apply ONLY to the
# distribution-prone GICS sectors below AND ONLY when asset turnover confirms the
# model (>= ASSET_TURNOVER_DISTRIBUTION_MIN); every other name keeps the standard bar.
ASSET_TURNOVER_DISTRIBUTION_MIN = 1.5
# Distribution-prone GICS sectors = the keys of the margin dicts below.
SECTOR_OPERATING_MARGIN_MIN: dict[str, float] = {
    "Consumer Discretionary": 6.0,
    "Industrials": 6.0,
}
SECTOR_GROSS_MARGIN_MIN: dict[str, float] = {
    "Consumer Discretionary": 22.0,
    "Industrials": 22.0,
}

# Quality floors (percent scores from Senior Fundamentals DATA_BLOCK)
HEALTH_MIN_PCT = 50.0
GROWTH_MIN_PCT = 50.0

# Senior Fundamentals scoring rubric totals (prompt: "FINANCIAL HEALTH SCORE
# (12 Points Total)" / "GROWTH TRANSITION SCORE (6 Points Total)"). N/A criteria
# may legitimately shrink the *available* denominator below these totals, so the
# code-side consistency check treats them as ceilings, not exact requirements.
HEALTH_RUBRIC_POINTS = 12.0
GROWTH_RUBRIC_POINTS = 6.0
FINANCIALS_HEALTH_REMOVED_POINTS = 1.0  # D/E point removed for Financials
SCORE_PCT_TOLERANCE = 1.5  # pct-points; absorbs LLM rounding (83% vs 83.3%)

# Discovery / diversification
ANALYST_COVERAGE_MAX = 15  # at/above: "discovered", hard fail
US_REVENUE_MAX_PCT = 35.0

# Portfolio Manager qualitative risk-tally zones
RISK_ZONE_HIGH = 2.0  # >=: default SELL
RISK_ZONE_MODERATE = 1.0  # >=: default HOLD; below: default BUY

# Large-drawdown triggers (shared by context_flags classification and the
# pre-graph news-analyst drawdown-investigation injection)
DRAWDOWN_52WK_RATIO = 0.60  # current/52wk-high at/below: large drawdown
DRAWDOWN_SMA200_RATIO = 0.80  # current below 0.8×SMA200: large drawdown
