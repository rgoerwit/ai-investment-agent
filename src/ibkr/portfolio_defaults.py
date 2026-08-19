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

# Price drift (%) vs the analysis-time spot price that flags a stale verdict
# for refresh. The entry price is a trade instruction, not a price anchor.
DEFAULT_DRIFT_PCT = 15.0

# Max number of stale analyses refreshed in a single reconciliation run.
DEFAULT_REFRESH_LIMIT = 10

# Refresh scheduling is weighted fair rather than strict-priority: when urgent
# and normal-cycle work are both pending, the default 2:1 pattern guarantees
# normal-cycle service across repeated bounded runs without making urgent work
# wait for an entire portfolio cycle.
DEFAULT_REFRESH_URGENT_WEIGHT = 2
DEFAULT_REFRESH_CYCLE_WEIGHT = 1

# A failed full analysis is retried on a later run, but not immediately on every
# reconciliation invocation. The scheduler state records the exact retry time.
DEFAULT_REFRESH_FAILURE_BACKOFF_HOURS = 24

# Concentration ceilings (% of portfolio) that trigger TRIM recommendations.
DEFAULT_SECTOR_LIMIT_PCT = 30.0
DEFAULT_EXCHANGE_LIMIT_PCT = 40.0

# Position-weight bands (% of portfolio) for ADD/TRIM rebalancing.
DEFAULT_OVERWEIGHT_PCT = 20.0
DEFAULT_UNDERWEIGHT_PCT = 20.0

# Positions below this USD value are de-minimis: never surfaced as executable
# actions or attention-level reviews (urgent price reviews, mandatory exits,
# and compliance-class flags like PFIC are exempt from suppression).
DEFAULT_MIN_ACTIONABLE_POSITION_USD = 300.0

# A held-position thesis-failure SELL requires CONFIRMATION: the most recent
# prior full-mode analysis must also reject, at least this many days before the
# current one (one bad data day re-analyzed twice must not self-confirm).
DEFAULT_SELL_CONFIRMATION_MIN_SPACING_DAYS = 7

# How far back to scan same-ticker history for the confirming prior verdict.
DEFAULT_SELL_CONFIRMATION_LOOKBACK_DAYS = 60
