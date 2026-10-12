"""Take the live FX tier out of play so FX lookups answer from the fallback table.

``get_fx_rate`` tries yfinance first and falls back to ``FALLBACK_RATES_TO_USD``.
A unit test that normalises a non-USD position without pinning the live tier
sends Yahoo requests; and the process-wide ``FxRateCache`` would carry one test's
rate into the next.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src import fx_normalization


def pin_live_fx_unavailable(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Make the live tier return None (as an unreachable Yahoo does); reset the cache."""
    live = AsyncMock(return_value=None)
    monkeypatch.setattr(fx_normalization, "get_fx_rate_yfinance", live)
    monkeypatch.setattr(fx_normalization, "_fx_rate_cache", None)
    return live
