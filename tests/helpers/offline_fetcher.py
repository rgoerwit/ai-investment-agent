"""Cut a ``SmartMarketDataFetcher`` off from Yahoo and Tavily at its own seams.

The fetcher builds a real Tavily client from the session's dummy key and repairs a
missing 52-week range from ``get_price_history`` (a Yahoo chart request). A test
that mocks the *sources* but not these two still sent Yahoo and Tavily requests
from the merge step. Both are advisory and answer "unavailable" offline: no Tavily
client (gap filling returns nothing) and an empty price history (the range is
blanked). A test that exercises either path sets its own after calling this.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pandas as pd

from src.data.fetcher import SmartMarketDataFetcher


def offline_market_fetcher(fetcher: SmartMarketDataFetcher) -> SmartMarketDataFetcher:
    """Return ``fetcher`` with no Tavily client and an empty price history."""
    fetcher.tavily_client = None
    fetcher.get_price_history = AsyncMock(return_value=pd.DataFrame())  # type: ignore[method-assign]
    return fetcher
