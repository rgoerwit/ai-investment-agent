"""Single source of truth for portfolio-reconciliation operational defaults.

Each knob below was previously re-declared as a literal at several layers — the
CLI signature default in ``cli_options.py``, a per-script CLI override in
``scripts/portfolio_manager.py``, the dashboard ``DashboardSettings``, and a
number of function signatures (``reconcile``, ``read_portfolio``,
``IbkrPortfolioDataService``, ``check_staleness``, …). Because the same value
lived in ~5–9 places, the layers drifted: the cash buffer read 0.03 in some
signatures and 0.05 in the operative CLI/dashboard defaults at the same time.

Reference these constants everywhere a default is needed. Changing a knob is then
a one-line edit here, and no layer can silently shadow another.
"""

# Fraction of net liquidation value held back as a cash buffer (never deployed
# into new BUYs). Operative default for both the CLI and the dashboard.
DEFAULT_CASH_BUFFER_PCT = 0.03

# Max analysis age (days) before a saved verdict is treated as stale.
DEFAULT_MAX_AGE_DAYS = 14

# Price drift (%) vs analysis entry that flags a stale verdict for refresh.
DEFAULT_DRIFT_PCT = 15.0

# Max number of stale analyses refreshed in a single reconciliation run.
DEFAULT_REFRESH_LIMIT = 10

# Concentration ceilings (% of portfolio) that trigger TRIM recommendations.
DEFAULT_SECTOR_LIMIT_PCT = 30.0
DEFAULT_EXCHANGE_LIMIT_PCT = 40.0

# Position-weight bands (% of portfolio) for ADD/TRIM rebalancing.
DEFAULT_OVERWEIGHT_PCT = 20.0
DEFAULT_UNDERWEIGHT_PCT = 20.0
