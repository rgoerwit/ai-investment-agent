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

# Quality floors (percent scores from Senior Fundamentals DATA_BLOCK)
HEALTH_MIN_PCT = 50.0
GROWTH_MIN_PCT = 50.0

# Discovery / diversification
ANALYST_COVERAGE_MAX = 15  # at/above: "discovered", hard fail
US_REVENUE_MAX_PCT = 35.0

# Portfolio Manager qualitative risk-tally zones
RISK_ZONE_HIGH = 2.0  # >=: default SELL
RISK_ZONE_MODERATE = 1.0  # >=: default HOLD; below: default BUY
