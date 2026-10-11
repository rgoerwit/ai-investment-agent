"""Shared yfinance runtime defaults and helpers."""

import yfinance as yf
import yfinance.exceptions as yf_exceptions

YFRateLimitError = yf_exceptions.YFRateLimitError

# yfinance's own retry loop sleeps 1 + 2 + 4 s between attempts of one request,
# invisibly to every deadline and with no knowledge of the source fallbacks. The
# application owns retries: a transient Yahoo failure is answered by the next
# source in the fetcher's chain, not by seven seconds of backoff inside one call.
YFINANCE_NETWORK_RETRIES = 0


def configure_yfinance_defaults() -> None:
    """Apply process-wide yfinance defaults: no library retries, loud errors."""
    try:
        yf.config.network.retries = YFINANCE_NETWORK_RETRIES
        yf.config.debug.hide_exceptions = False
    except AttributeError:
        pass
