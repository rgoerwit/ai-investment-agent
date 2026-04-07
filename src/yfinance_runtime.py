"""Shared yfinance runtime defaults and helpers."""

import yfinance as yf
import yfinance.exceptions as yf_exceptions

YFRateLimitError = yf_exceptions.YFRateLimitError


def configure_yfinance_defaults() -> None:
    """Apply process-wide yfinance defaults with conservative retry behavior."""
    retries = getattr(yf.config.network, "retries", 0)
    if retries < 3:
        yf.config.network.retries = 3
    yf.config.debug.hide_exceptions = False
