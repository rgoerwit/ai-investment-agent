"""
Custom exceptions for IBKR integration.

Clean error hierarchy for distinct failure modes.
"""


class IBKRError(Exception):
    """Base exception for all IBKR-related errors."""


class IBKRAuthError(IBKRError):
    """OAuth failure, session expired, or invalid credentials."""


class IBKRAPIError(IBKRError):
    """HTTP errors, rate limits (429), server errors from IBKR API."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class IBKRTickerResolutionError(IBKRError):
    """conid lookup failed for a given yfinance ticker."""

    def __init__(self, ticker: str, message: str = ""):
        self.ticker = ticker
        super().__init__(
            message or f"Failed to resolve IBKR conid for ticker: {ticker}"
        )


class IBKRSessionConflictError(IBKRError):
    """Brokerage session active elsewhere (only one session per username)."""
